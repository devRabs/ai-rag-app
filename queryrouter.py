from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

# ---------------------------------------
# 0. Load environment variables
# ---------------------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

# ---------------------------------------
# 1. Load PDF
# ---------------------------------------
loader = PyPDFLoader("data/udemy.pdf")
documents = loader.load()

# Add metadata
for i, doc in enumerate(documents):
    doc.metadata["page"] = i

# ---------------------------------------
# 2. Split into chunks
# ---------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)

# ---------------------------------------
# 3. Embeddings
# ---------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ---------------------------------------
# 4. Vector DB
# ---------------------------------------
db = FAISS.from_documents(split_docs, embeddings)

# ---------------------------------------
# 5. LLM (Groq)
# ---------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ---------------------------------------
# 6. Helper: Extract city
# ---------------------------------------
def extract_city(query):
    words = query.lower().split()

    if "in" in words:
        idx = words.index("in")
        if idx + 1 < len(words):
            return words[idx + 1]

    return words[-1]  # fallback

# ---------------------------------------
# 7. Get coordinates (Open-Meteo)
# ---------------------------------------
def get_coords(city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    res = requests.get(url).json()

    if res.get("results"):
        return res["results"][0]["latitude"], res["results"][0]["longitude"]

    return None, None

# ---------------------------------------
# 8. Weather function (NO API KEY needed)
# ---------------------------------------
def get_weather(query):
    city = extract_city(query)

    lat, lon = get_coords(city)

    if not lat:
        return "City not found."

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"

    try:
        data = requests.get(url).json()
        temp = data["current"]["temperature_2m"]
        return f"{city.capitalize()} temperature: {temp}°C"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# ---------------------------------------
# 9. Query Router
# ---------------------------------------
def route_query(query):
    q = query.lower()

    if any(word in q for word in ["weather", "temperature", "climate"]):
        return "weather"

    if any(word in q for word in ["course", "pdf", "enroll", "student"]):
        return "rag"

    return "general"

# ---------------------------------------
# 10. Chat loop
# ---------------------------------------
print("🤖 Smart RAG Assistant Ready (type 'exit' to quit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    route = route_query(query)

    # ---------------------------------------
    # 🔹 WEATHER (Real-time)
    # ---------------------------------------
    if route == "weather":
        answer = get_weather(query)
        print("AI:", answer)
        continue

    # ---------------------------------------
    # 🔹 RAG (PDF)
    # ---------------------------------------
    if route == "rag":
        retrieved_docs = db.similarity_search(query, k=3)

        if not retrieved_docs:
            # fallback to LLM
            response = llm.invoke(query)
            print("AI:", response.content)
            continue

        context = "\n\n".join([
            f"(Page {doc.metadata.get('page')}) {doc.page_content}"
            for doc in retrieved_docs
        ])

        prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question: {query}
"""

        response = llm.invoke(prompt)

        print("\n📄 Sources:", [doc.metadata.get("page") for doc in retrieved_docs])
        print("AI:", response.content)
        continue

    # ---------------------------------------
    # 🔹 GENERAL LLM
    # ---------------------------------------
    response = llm.invoke(query)
    print("AI:", response.content)