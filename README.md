# DocBot — AI Document Assistant

I built DocBot because I wanted to prove a point: you should be able to upload any document — a scanned invoice, a Word report, a photo of a handwritten note — and just ask it questions in plain English. No formatting tricks, no copy-pasting, no searching through pages manually.

That's what DocBot does.

---

## What it does

Upload a document and DocBot reads it, summarizes it, suggests questions you might want to ask, and then answers whatever you type — pulling answers directly from the document content, not from the AI's general knowledge.

It works across five file types:

- **PDF** — text-based and scanned (image-based)
- **DOCX** — Word documents
- **TXT** — plain text files
- **JPG / PNG** — photos, screenshots, scanned images

For scanned documents and images, Google Cloud Vision OCR extracts the text first — so even a photo of a printed page works.

---

## How it works

Standard RAG pipeline, built from scratch without LangChain or any abstraction layer:

1. Document uploaded → text extracted (PyPDF2 for text PDFs, Google Vision for scanned files, python-docx for Word)
2. Text split into overlapping 1000-character chunks
3. Each chunk converted into a vector embedding using Google Gemini
4. Embeddings stored in Pinecone vector database
5. User asks a question → question gets embedded the same way
6. Pinecone finds the 5 most relevant chunks via similarity search
7. Those chunks sent to Claude as context → Claude generates a grounded answer

Claude is instructed to only answer from the provided context. If the answer isn't in the document, it says so — no hallucination.

---

## Tech stack

| Layer | Tool |
|---|---|
| Backend | Python, Flask |
| AI — answers | Claude (Anthropic) |
| AI — embeddings | Gemini (Google) |
| Vector database | Pinecone |
| OCR | Google Cloud Vision |
| PDF parsing | PyPDF2, pdf2image |
| Word parsing | python-docx |
| Frontend | HTML, CSS, JavaScript |

---

## Running it locally

**1. Clone the repo**
```bash
git clone https://github.com/praneeth-sholapur/docbot-ai.git
cd docbot-ai
```

**2. Install dependencies**
```bash
pip install flask flask-cors PyPDF2 google-generativeai anthropic pinecone google-cloud-vision pdf2image python-docx pillow python-dotenv
```

**3. Create a `.env` file**
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/vision-key.json

You will need:
- [Google AI Studio](https://aistudio.google.com) — Gemini embeddings
- [Pinecone](https://pinecone.io) — free tier, create index with 3072 dimensions
- [Anthropic](https://console.anthropic.com) — Claude answers
- [Google Cloud](https://console.cloud.google.com) — enable Cloud Vision API, download service account key

**4. Run**
```bash
python server.py
```

Open `http://localhost:5000` in your browser.

---

## What I learned building this

The hardest part wasn't the RAG pipeline itself. The harder parts were getting scanned PDF support working reliably across platforms, figuring out that Pinecone's dimension count needs to exactly match the embedding model output, and building a frontend that feels like a real product without using any framework.

The part I am most proud of is the multi-format support. Most RAG demos only handle clean text PDFs. This one handles whatever you throw at it — including a 26-page scanned car rental damage claim document, which is what prompted me to add OCR support in the first place.

---

## Author

Praneeth Sholapur — [Portfolio](https://praneeth-sholapur.github.io) · [LinkedIn](https://linkedin.com/in/praneeth-sholapur-1b89062a3)
