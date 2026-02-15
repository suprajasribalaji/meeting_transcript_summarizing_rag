from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from pypdf import PdfReader
import chainlit as cl
import os
from dotenv import load_dotenv
import uuid
import hashlib
import time
import logging
import json
from logging.handlers import RotatingFileHandler

load_dotenv()


# LOGGING SETUP
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })

retrieval_logger = logging.getLogger("retrieval")
retrieval_logger.setLevel(logging.INFO)

retrieval_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "retrieval.log"),
    maxBytes=5_000_000,
    backupCount=3
)
retrieval_handler.setFormatter(JsonFormatter())
retrieval_logger.addHandler(retrieval_handler)

error_logger = logging.getLogger("error")
error_logger.setLevel(logging.ERROR)

error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"),
    maxBytes=5_000_000,
    backupCount=3
)
error_handler.setFormatter(JsonFormatter())
error_logger.addHandler(error_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
retrieval_logger.addHandler(console_handler)
error_logger.addHandler(console_handler)


# Qdrant Collection Config
TRANSCRIPT_DOCUMENT_COLLECTION = "transcript_document"
SIZE = 384


# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)


# Initialize the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Initialize the Qdrant Client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)


# Ensure Collection exists
def ensure_collection_exists(collection_name, size):
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )


# Ensure Payload Indexes setted
def ensure_payload_indexes():
    # Index for file_hash (deduplication)
    qdrant_client.create_payload_index(
        collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
        field_name="file_hash",
        field_schema="keyword",
    )

    # Index for document_id (query filtering)
    qdrant_client.create_payload_index(
        collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
        field_name="document_id",
        field_schema="keyword",
    )


ensure_collection_exists(TRANSCRIPT_DOCUMENT_COLLECTION, SIZE)
ensure_payload_indexes()


# Retriever + Answer Question
async def answer_question(user_input, active_document_id):

    start_time = time.time()

    retrieval_logger.info(f"User Question: {user_input}")
    retrieval_logger.info(f"Active Document ID: {active_document_id}")

    query_vector = embeddings.embed_query(user_input)

    search_result = qdrant_client.query_points(
        collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=active_document_id),
                )
            ]
        ),
        limit=7,
    )

    if not search_result.points:
        return "I don't know from this document."

    retrieved_context = "\n\n".join(
        p.payload["text"] for p in search_result.points
    )

    page_numbers = sorted(
        list({p.payload["page_number"] for p in search_result.points})
    )

    system_prompt = f"""
        You are a strict internal document assistant.

        Rules:
        - STRICTLY USE ONLY the provided context.
        - Do NOT use outside knowledge.
        - Do NOT speculate.
        - If answer is not fully supported, say:
            "I don't know from this document. Ask anything from the document."

        Context:
        {retrieved_context}
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])
    except Exception as e:
        error_logger.error(f"LLM Error: {str(e)}")
        return "An internal error occurred."

    generated_answer = response.content

    # Hallucination Validation
    validation_prompt = f"""
        Check if the ANSWER is fully supported by the CONTEXT.

        If unsupported, respond ONLY with:
        INVALID

        If fully supported, respond ONLY with:
        VALID

        CONTEXT:
        {retrieved_context}

        ANSWER:
        {generated_answer}
    """

    validation_response = await llm.ainvoke([
        HumanMessage(content=validation_prompt)
    ])

    if "INVALID" in validation_response.content:
        retrieval_logger.warning("Hallucination detected.")
        return "I don't know from this document."

    forbidden_phrases = [
        "according to general knowledge",
        "it is widely known",
        "in general",
        "from my opinion"
    ]

    if any(p in generated_answer.lower() for p in forbidden_phrases):
        retrieval_logger.warning("Guardrail triggered.")
        return "I don't know from this document. Ask anything from this document."

    latency = round(time.time() - start_time, 2)
    retrieval_logger.info(f"Pages Used: {page_numbers}")
    retrieval_logger.info(f"Latency: {latency}s")
    retrieval_logger.info("-" * 60)

    citation_text = f"\n\n(Source: Page {', '.join(map(str, page_numbers))})"
    return generated_answer + citation_text


# Chat start
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("active_document_id", None)
    await cl.Message(content="Hey, I'm your Meeting Transcript Summarizing Assistant.\nUpload a PDF to begin.").send()


# Message Handler
@cl.on_message
async def handle_message(message: cl.Message):

    # ---------------- PDF Upload ----------------
    if message.elements:

        for element in message.elements:

            reader = PdfReader(element.path)

            full_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

            file_hash = hashlib.sha256(full_text.encode()).hexdigest()

            existing = qdrant_client.scroll(
                collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_hash",
                            match=MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            if existing[0]:
                existing_doc_id = existing[0][0].payload["document_id"]
                cl.user_session.set("active_document_id", existing_doc_id)

                await cl.Message(
                    content=f"⚡ '{element.name}' already indexed. Using existing document."
                ).send()

                if message.content:
                    answer = await answer_question(
                        message.content.strip(),
                        existing_doc_id
                    )
                    await cl.Message(content=answer).send()

                return

            # New document
            document_id = str(uuid.uuid4())
            cl.user_session.set("active_document_id", document_id)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
            )

            batch = []

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                chunks = splitter.split_text(page_text)
                vectors = embeddings.embed_documents(chunks)

                for chunk, vector in zip(chunks, vectors):
                    batch.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "text": chunk,
                                "document_id": document_id,
                                "file_hash": file_hash,
                                "page_number": page_number,
                            }
                        )
                    )

            qdrant_client.upsert(
                collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
                points=batch
            )

            await cl.Message(
                content=f"✅ '{element.name}' indexed successfully."
            ).send()

            if message.content:
                answer = await answer_question(
                    message.content.strip(),
                    document_id
                )
                await cl.Message(content=answer).send()

        return

    # ---------------- Question Only ----------------
    user_input = message.content.strip()

    active_document_id = cl.user_session.get("active_document_id")

    if not active_document_id:
        await cl.Message(content="⚠ Please upload a PDF first.").send()
        return

    answer = await answer_question(user_input, active_document_id)
    await cl.Message(content=answer).send()

@cl.on_chat_end
async def on_chat_end():
    print("User disconnected! Either closed the session by closing the window!")