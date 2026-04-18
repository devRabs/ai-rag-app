from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load PDF
loader = PyPDFLoader("data/udemy.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings (local)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Store in vector DB
db = FAISS.from_documents(docs, embeddings)

# 5. Load environment variables
load_dotenv()

# 6. Load  LLM with Groq API key from .env
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

print("🤖 PDF Chatbot Ready (type 'exit' to quit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # 7. Retrieve relevant chunks
    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    # 8. Ask LLM with context
    prompt = f"""
    Answer based only on the context below:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print("AI:", response.content)