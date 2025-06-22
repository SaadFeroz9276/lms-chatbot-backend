import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware

# Load FAISS vector store
DB_PATH = "./faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load Mistral model via Ollama
llm = Ollama(model="mistral")

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
    return_source_documents=True
)

# FastAPI setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Greeting phrases (multilingual)
GREETINGS = {
    "hello", "hi", "hey", "salam", "hola", "bonjour", "ciao", "namaste",
    "السلام عليكم", "안녕하세요", "こんにちは", "guten tag", "hallo", "ola",
    "good morning", "good evening", "good afternoon", "howdy"
}

# University context keywords (expand as needed)
UNIVERSITY_TOPICS = {
    "course", "exam", "attendance", "gpa", "grades", "university", "semester",
    "prerequisite", "enrollment", "credit hour", "schedule", "department", "subject"
}

# Input schema
class ChatRequest(BaseModel):
    query: str

# Helper functions
def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETINGS

def is_university_related(text: str) -> bool:
    return any(keyword in text.lower() for keyword in UNIVERSITY_TOPICS)

@app.post("/chat")
async def chat_endpoint(data: ChatRequest):
    query = data.query.strip()

    # Handle greetings
    if is_greeting(query):
        return {"response": "👋 Hello! I'm your university assistant. Feel free to ask me anything about courses, exams, or policies."}

    # Use RAG if university-related
    if is_university_related(query):
        try:
            result = rag_chain.invoke({"query": query})
            return {"response": result["result"]}
        except Exception as e:
            return {"response": "Something went wrong while processing your academic question. Please try again later."}

    # Fallback for irrelevant prompts
    return {
        "response": "🤖 I'm here to assist with your university-related questions such as courses, exams, or attendance. Please rephrase your question if needed."
    }

# Run with: uvicorn rag_api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
