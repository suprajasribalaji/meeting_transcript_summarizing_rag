AI Meeting Transcription Summarizer (RAG + Memory)

An AI-powered Meeting Transcript Summarizer built using:

- Chainlit (Interactive Chat UI)
- Google Gemini 2.5 Flash (LLM)
- Qdrant Cloud (Vector Database)
- HuggingFace Sentence Transformers (Embeddings)
- Retrieval Augmented Generation (RAG)

------------------------------------------------------------

FEATURES

- PDF Upload & Chunking
- Embedding using all-MiniLM-L6-v2 (384 dimension)
- Vector Storage in Qdrant Cloud
- Top-5 Similarity Retrieval
- Strict Context-Based LLM Response
- Session-Based Chat History
- Automatic Same-Day Session Recovery
- Anti-Hallucination Enforcement

------------------------------------------------------------

ARCHITECTURE

User (Browser)
      ↓
Chainlit (UI + Backend)
      ↓
Gemini 2.5 Flash (LLM)
      ↓
Qdrant Cloud (Vector DB)

------------------------------------------------------------

PROJECT STRUCTURE

meeting_transcription_summarizer/

- rag_chain_memory.py   (Main Chainlit app)
- requirements.txt
- README.txt
- .env (Not committed)

------------------------------------------------------------

ENVIRONMENT VARIABLES

GOOGLE_API_KEY=your_google_api_key
QDRANT_CLUSTER_ENDPOINT=https://your-qdrant-url
QDRANT_API_KEY=your_qdrant_api_key

NOTE: Do NOT commit .env to GitHub.

------------------------------------------------------------

LOCAL SETUP

1. Create Virtual Environment:

python -m venv .venv
.venv\\Scripts\\activate

2. Install Dependencies:

pip install -r requirements.txt

3. Run Application:

chainlit run rag_chain_memory.py

App runs at:
http://localhost:8000

------------------------------------------------------------

DEPLOYMENT (Render)

Build Command:
pip install -r requirements.txt

Start Command:
chainlit run rag_chain_memory.py --host 0.0.0.0 --port 10000

Add environment variables in Render dashboard.

------------------------------------------------------------

HOW IT WORKS

1. Upload transcript PDF
2. Text is chunked using RecursiveCharacterTextSplitter
3. Each chunk embedded using MiniLM (384 dimension)
4. Stored in Qdrant Cloud
5. User query embedded
6. Top-5 similar chunks retrieved
7. Gemini generates answer using ONLY retrieved context
8. Chat history stored with:
   - session_id
   - session_date
   - role
   - content
   - created_at

------------------------------------------------------------

ANTI-HALLUCINATION RULES

- Uses ONLY retrieved context
- Returns 5 concise bullet points by default
- Says "I don't know" if information not found
- No hallucination allowed

------------------------------------------------------------

TECH STACK

UI: Chainlit
LLM: Google Gemini 2.5 Flash
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector DB: Qdrant Cloud
Backend: Python
Retrieval: Similarity Search (k=5)

------------------------------------------------------------

AUTHOR

Supraja Sri R B
AI Engineer | RAG Systems | LLM Applications

output_path
