from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# NEW imports
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 1. Load PDF
loader = PyPDFLoader("data/udemy.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Vector store (FAISS)
vector_db = FAISS.from_documents(docs, embeddings)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 5. BM25 (keyword search)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

# 6. HYBRID RETRIEVER (key part)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # you can tune this
)

# 7. Load environment variables
load_dotenv()

# 8. LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

print("🤖 Hybrid PDF Chatbot Ready (type 'exit' to quit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # 9. HYBRID SEARCH
    docs = hybrid_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # 10. Prompt
    prompt = f"""
    Answer ONLY from the context below.
    If answer is not present, say "Not found in document".

    Context:
    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print("AI:", response.content)