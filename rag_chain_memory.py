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
from datetime import datetime

load_dotenv()

TRANSCRIPT_DOCUMENT_COLLECTION = "transcript_document"
CHAT_HISTORY_COLLECTION = "chat_history"
SIZE = 384

# Initialize the Gemini LLm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

# Initialize the HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize the Qdrant Client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Ensure Collections are exists
def ensure_collection_exists(collection_name, size):
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )

ensure_collection_exists(TRANSCRIPT_DOCUMENT_COLLECTION, SIZE)
ensure_collection_exists(CHAT_HISTORY_COLLECTION, SIZE)

# Ensure Payload indexes exists
def ensure_payload_indexes():
    qdrant_client.create_payload_index(
        collection_name=CHAT_HISTORY_COLLECTION,
        field_name="session_date",
        field_schema="keyword",
    )

    qdrant_client.create_payload_index(
        collection_name=CHAT_HISTORY_COLLECTION,
        field_name="session_id",
        field_schema="keyword",
    )

ensure_payload_indexes()

# Get or create session id for today
def get_or_create_session_id():
    today = datetime.now().strftime("%Y-%m-%d")

    response = qdrant_client.scroll(
        collection_name=CHAT_HISTORY_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="session_date",
                    match=MatchValue(value=today),
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )

    records = response[0]

    if records:
        return records[0].payload["session_id"]

    return str(uuid.uuid4())

# Save chat message to qdrant collection
def save_chat_message(session_id, role, content):

    now = datetime.now()
    session_date = now.strftime("%Y-%m-%d")
    created_at = now.isoformat()

    vector = embeddings.embed_query(content)

    qdrant_client.upsert(
        collection_name=CHAT_HISTORY_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "session_id": session_id,
                    "session_date": session_date,
                    "role": role,
                    "content": content,
                    "created_at": created_at,
                },
            )
        ],
    )

# Load chat history from the qdrant collection
async def load_chat_history(session_id):
    response = qdrant_client.scroll(
        collection_name=CHAT_HISTORY_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        ),
        limit=200,
        with_payload=True,
    )

    records = response[0]
    records.sort(key=lambda x: x.payload["created_at"])

    for record in records:
        role = record.payload["role"]
        content = record.payload["content"]

        msg = cl.Message(content=content)
        msg.author = role
        await msg.send()


# Chat start
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Loading session...").send()
    session_id = get_or_create_session_id()
    cl.user_session.set("session_id", session_id)

    print("ACTIVE SESSION:", session_id)

    await load_chat_history(session_id)

    await cl.Message(
        content="Hey, I'm Meeting Transcript Summarizer Assistant.\nUpload the meeting transcript PDF to help you!"
    ).send()

# Message handler
@cl.on_message
async def handle_message(message: cl.Message):

    session_id = cl.user_session.get("session_id")

    # -------- PDF Upload --------
    if message.elements:
        documents = []

        for element in message.elements:
            if element.mime == "application/pdf":
                reader = PdfReader(element.path)

                for page_number, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    documents.append({
                        "text": text,
                        "page_number": page_number + 1
                    })

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
        )

        chunks = []

        for doc in documents:
            page_chunks = splitter.split_text(doc["text"])
            for chunk in page_chunks:
                chunks.append({
                    "text": chunk,
                    "page_number": doc["page_number"]
                })

        texts = [chunk["text"] for chunk in chunks]
        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page_number": chunk["page_number"]
                }
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        qdrant_client.upsert(
            collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
            points=points
        )

        await cl.Message(
            content="✅ PDF indexed successfully! Ask me anything."
        ).send()

        return

    # -------- Normal Chat --------
    user_input = (message.content or "").strip()

    if not user_input:
        user_input = "Provide a concise summary in 5 bullet points."

    save_chat_message(session_id, "user", user_input)

    query_vector = embeddings.embed_query(user_input)

    search_result = qdrant_client.query_points(
        collection_name=TRANSCRIPT_DOCUMENT_COLLECTION,
        query=query_vector,
        limit=5
    )

    if not search_result.points:
        fallback = "I can't find this, so please ask something else."
        save_chat_message(session_id, "assistant", fallback)
        await cl.Message(content=fallback).send()
        return

    retrieved_context = "\n\n".join(
        [point.payload["text"] for point in search_result.points]
    )

    system_prompt = f"""
You are an AI Meeting Transcription Assistant using RAG.

Rules:
- Use ONLY the retrieved context below.
- If format not specified, give 5 concise bullet points.
- If not found, say "I don't know."
- No hallucination.

Retrieved Context:
{retrieved_context}
"""

    async with cl.Step(name="🔎 Retrieving & Analyzing...", type="llm") as step:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])
        step.output = response.content

    assistant_reply = response.content

    save_chat_message(session_id, "assistant", assistant_reply)

    await cl.Message(content=assistant_reply).send()

@cl.on_chat_end
async def on_chat_end():
    print("User disconnected!")
