import os
import requests
from openai import OpenAI
from database import supabase
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 🔹 Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 LLM client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# 🔹 Get latest uploaded PDF
def get_latest_pdf():
    response = supabase.table("chatbot_documents") \
        .select("*") \
        .eq("action", "uploaded") \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()

    if not response.data:
        return None

    return response.data[0]["file_url"]


# 🔹 Download PDF
def download_pdf(url, save_path="temp.pdf"):
    response = requests.get(url)

    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path


# 🔹 Extract text
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


# 🔹 Split text into chunks
def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# 🔹 Generate embeddings
def get_embedding(text):
    return model.encode(text).tolist()


# 🔹 Store chunks
def store_chunks(chunks):
    print("🧹 Clearing old embeddings...")

    # 🔥 Always delete old data
    supabase.table("rag_documents") \
        .delete() \
        .neq("content", "") \
        .execute()

    print("📦 Storing new chunks...")

    data = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        data.append({
            "content": chunk,
            "embedding": embedding
        })

    supabase.table("rag_documents").insert(data).execute()

    print("✅ New chunks stored successfully")


# 🔹 Retrieve relevant context
def retrieve_context(question):
    query_embedding = get_embedding(question)

    response = supabase.rpc("match_rag_documents", {
        "query_embedding": query_embedding
    }).execute()

    texts = [doc["content"] for doc in response.data]
    return "\n".join(texts)


# 🔹 Generate answer
def get_llm_response(context, question):
    prompt = f"""
    Note**= You are a helpful chatbot , you have to always greet the user if they greet you and Answer ONLY from the given context.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content