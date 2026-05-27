from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import PyPDF2
import google.genai as genai
import anthropic
from pinecone import Pinecone
from google.cloud import vision
from pdf2image import convert_from_bytes
from docx import Document
import os
import io
import base64

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\shola\vision-key.json"
POPPLER_PATH = r"C:\Users\shola\poppler\poppler-26.02.0\Library\bin"

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("knowledge-assistant")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
vision_client = vision.ImageAnnotatorClient()

app = Flask(__name__)
CORS(app)

def ocr_image_bytes(image_bytes):
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def extract_from_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    pages = len(reader.pages)
    for i, page in enumerate(reader.pages):
        t = page.extract_text()
        if t:
            text += f"\n[Page {i+1}]\n{t}"
    return text, pages

def extract_from_pdf_ocr(file_bytes):
    images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
    text = ""
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        ocr_text = ocr_image_bytes(buf.getvalue())
        text += f"\n[Page {i+1}]\n{ocr_text}"
    return text, len(images)

def extract_from_image(file_bytes):
    return ocr_image_bytes(file_bytes), 1

def extract_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text, 1

def extract_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore"), 1

def extract_text(filename, file_bytes):
    name = filename.lower()
    if name.endswith(".txt"):
        return extract_from_txt(file_bytes)
    elif name.endswith(".docx"):
        return extract_from_docx(file_bytes)
    elif name.endswith((".jpg", ".jpeg", ".png")):
        return extract_from_image(file_bytes)
    elif name.endswith(".pdf"):
        text, pages = extract_from_pdf_text(file_bytes)
        if len(text.strip()) < 100:
            return extract_from_pdf_ocr(file_bytes)
        return text, pages
    return "", 0

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append({"text": chunk, "chunk_index": len(chunks)})
    return chunks

def embed_and_store(chunks, source_name):
    try:
        index.delete(delete_all=True)
    except Exception:
        pass
    for chunk in chunks:
        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"]
        )
        embedding = result.embeddings[0].values
        index.upsert(vectors=[{
            "id": f"{source_name}_chunk_{chunk['chunk_index']}",
            "values": embedding,
            "metadata": {"text": chunk["text"], "source": source_name}
        }])

def generate_summary_and_questions(text):
    preview = text[:3000]
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Based on this document content, do two things:
1. Write a 2-3 sentence summary of what this document is about.
2. Generate exactly 3 specific questions a user might want to ask about this document.

Format your response exactly like this:
SUMMARY: [your summary here]
Q1: [question 1]
Q2: [question 2]
Q3: [question 3]

Document content:
{preview}"""
        }]
    )
    raw = response.content[0].text
    summary = ""
    questions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("Q1:"):
            questions.append(line.replace("Q1:", "").strip())
        elif line.startswith("Q2:"):
            questions.append(line.replace("Q2:", "").strip())
        elif line.startswith("Q3:"):
            questions.append(line.replace("Q3:", "").strip())
    return summary, questions

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    file_bytes = file.read()
    text, pages = extract_text(file.filename, file_bytes)
    if not text.strip():
        return jsonify({"error": "Could not extract text from this file"}), 400
    chunks = chunk_text(text)
    embed_and_store(chunks, file.filename)
    summary, questions = generate_summary_and_questions(text)
    return jsonify({
        "filename": file.filename,
        "pages": pages,
        "chunks": len(chunks),
        "summary": summary,
        "questions": questions
    })

@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )
    question_embedding = result.embeddings[0].values
    search_results = index.query(
        vector=question_embedding,
        top_k=5,
        include_metadata=True
    )
    context = "\n\n".join([match.metadata['text'] for match in search_results.matches])
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""You are a precise document assistant. Answer the question using ONLY the context below.
- Answer directly and clearly without mentioning source numbers or references.
- Combine information from all relevant parts into one clean answer.
- If the answer is not in the context, say: "Hmm, I looked everywhere in this document and came up empty. Try asking something else or upload a different document!"
- Never make up information.

Context:
{context}

Question: {question}

Answer:"""
        }]
    )
    return jsonify({"answer": response.content[0].text})

@app.route("/clear", methods=["POST"])
def clear():
    try:
        index.delete(delete_all=True)
    except Exception:
        pass
    return jsonify({"status": "cleared"})

@app.route("/")
def serve_html():
    return send_from_directory(r"C:\Users\shola", "docbot.html")

if __name__ == "__main__":
    app.run(port=5000, debug=False)
