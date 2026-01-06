from fastapi import FastAPI, APIRouter, UploadFile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

app = FastAPI()
router = APIRouter()

# ------------------ Embeddings ------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key,
)

# ------------------ Upload & Embed Docs ------------------
@router.post("/doc_embeddings")
async def doc_embeddings(file: UploadFile):
    content_bytes = await file.read()
    text = content_bytes.decode("utf-8", errors="ignore")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_text(text)

    docs = [
        Document(
            page_content=t,
            metadata={"source": file.filename}
        )
        for t in texts
    ]

    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name="math_uploaded_doc",
        force_recreate=False,
    )

    return {
        "filename": file.filename,
        "chunks": len(texts),
        "collection": "math_uploaded_doc"
    }

# ------------------ Query Vector DB ------------------
@router.post("/query")
async def user_query(query: str):
    vector_store = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        embedding=embeddings,
        collection_name="math_uploaded_doc",
    )

    search_results = vector_store.similarity_search(
        query=query,
        k=3
    )

    context = "\n\n".join(
        f"Chunk:\n{doc.page_content}"
        for doc in search_results
    )

    return {
        "query": query,
        "results_found": len(search_results),
        "context": context
    }

# ------------------ Register Router ------------------
app.include_router(router)
