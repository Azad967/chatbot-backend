from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_services import (
    extract_text,
    split_text,
    store_chunks,
    retrieve_context,
    get_llm_response,
    get_latest_pdf,
    download_pdf,
)
from database import supabase
class ChatRequest(BaseModel):
    question: str
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔹 Chat endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        question = request.question

        context = retrieve_context(question)
        answer = get_llm_response(context, question)

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "Server error occurred"}


# 🔹 Clear old embeddings (FIXED)
def clear_old_embeddings():
    supabase.table("rag_documents") \
        .delete() \
        .neq("content", "") \
        .execute()


# 🔹 Load latest PDF from Supabase on startup
# @app.on_event("startup")
# def load_pdf_on_start():
#     print("🚀 Loading PDF from Supabase...")

#     pdf_url = get_latest_pdf()

#     if not pdf_url:
#         print("❌ No PDF found in database")
#         return

#     file_path = download_pdf(pdf_url)

#     text = extract_text(file_path)
#     chunks = split_text(text)

#     # 🔥 Clear old data safely
#     clear_old_embeddings()

#     # 🔥 Store new embeddings
#     store_chunks(chunks)

#     print("✅ PDF processed and embeddings stored")

@app.post("/process-pdf")
async def process_pdf():
    print("📥 Processing latest PDF...")

    pdf_url = get_latest_pdf()

    if not pdf_url:
        return {"status": "error", "message": "No PDF found"}

    file_path = download_pdf(pdf_url)

    text = extract_text(file_path)
    chunks = split_text(text)

    # Clear old embeddings
    clear_old_embeddings()

    # Store new embeddings
    store_chunks(chunks)

    print("✅ PDF processed successfully")

    return {"status": "success"}
