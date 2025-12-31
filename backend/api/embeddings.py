from fastapi import FastAPI, APIRouter, UploadFile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()
google_api_key=os.getenv("GOOGLE_API_KEY")

app=FastAPI()
router=APIRouter()




@router.post("/doc_embeddings")
async def doc_embeddings(file: UploadFile):
    content_bytes = await file.read()
    text = content_bytes.decode("utf-8", errors="ignore")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
    )

    docs = [Document(page_content=t) for t in texts]

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name="math_uploaded_doc",
        force_recreate=False,
    )

    return {"filename": file.filename, "chunks": len(texts), "collection": "math_uploaded_doc"}

# Ensure router is registered so endpoints show in docs
app.include_router(router, prefix="/api")
    