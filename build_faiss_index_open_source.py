
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Directory containing your documents

DOC_DIR = "./documents"

# Load documents
def load_documents(doc_dir):
    docs = []
    for fname in os.listdir(doc_dir):
        path = os.path.join(doc_dir, fname)
        try:
            if fname.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif fname.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    return docs

# Chunk documents
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# Main build function
def build_faiss_index():
    print("Loading documents...")
    documents = load_documents(DOC_DIR)
    print(f"Loaded {len(documents)} documents")

    print("Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Loading HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding and building index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Saving index to ./faiss_index")
    vectorstore.save_local("faiss_index")
    print("FAISS index successfully created!")

if __name__ == "__main__":
    build_faiss_index()
