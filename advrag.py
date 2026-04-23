import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# =========================
# 1. LOAD ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found")

# =========================
# 2. LOAD PDF
# =========================
loader = PyPDFLoader("data/udemy.pdf")
documents = loader.load()

# =========================
# 3. BETTER CHUNKING
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)

# =========================
# 4. EMBEDDINGS (UPGRADED)
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vector_db = FAISS.from_documents(docs, embeddings)

# =========================
# 5. BM25
# =========================
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 15

# =========================
# 6. LLM
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# =========================
# 7. QUERY EXPANSION
# =========================
def expand_query(query):
    expansions = {
        "mcp": "model context protocol MCP architecture components hosts clients servers tools resources connections system design",
        "price": "price cost fee subscription pricing amount ₹",
        "instructor": "instructor teacher author trainer background experience profile",
        "course": "course training program syllabus content details",
    }

    for key in expansions:
        if key in query.lower():
            query += " " + expansions[key]

    return query

# =========================
# 8. MULTI-QUERY GENERATION
# =========================
def generate_queries(query):
    return [
        query,
        f"Explain {query}",
        f"List details about {query}",
        f"{query} architecture components",
    ]

# =========================
# 9. SEMANTIC SCORING
# =========================
def semantic_score(doc, query):
    q_emb = embeddings.embed_query(query)
    d_emb = embeddings.embed_query(doc.page_content)

    return sum(q * d for q, d in zip(q_emb, d_emb))

# =========================
# 10. RAG PIPELINE
# =========================
def run_rag_query(query):

    queries = generate_queries(query)

    all_docs = []

    # --- MULTI-QUERY RETRIEVAL ---
    for q in queries:
        q = expand_query(q)

        vector_docs = vector_db.similarity_search(q, k=10)
        bm25_docs = bm25.invoke(q)

        all_docs.extend(vector_docs + bm25_docs)

    # --- DEDUP ---
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # --- FALLBACK ---
    if len(unique_docs) < 5:
        fallback_docs = vector_db.similarity_search(query, k=20)
        unique_docs.extend(fallback_docs)

    # --- SEMANTIC RANKING ---
    ranked_docs = sorted(
        unique_docs,
        key=lambda d: semantic_score(d, query),
        reverse=True
    )

    # --- FINAL CONTEXT ---
    final_docs = ranked_docs[:10]

    print("\n--- TOP CHUNKS ---")
    for i, d in enumerate(final_docs):
        print(f"\nChunk {i+1}:\n{d.page_content[:200]}")

    context = "\n\n".join([doc.page_content for doc in final_docs])

    prompt = f"""
    Answer ONLY from the context below.
    If not found say 'Not found in document'.

    Context:
    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)
    return response.content

# =========================
# 11. TEST CASES
# =========================
test_cases = [
    {
        "name": "Pricing Test",
        "query": "What is the price of the course and subscription options?",
        "expected_keywords": ["₹", "subscription", "monthly", "cost"],
        "should_fail": False
    },
    {
        "name": "Prerequisite Test",
        "query": "Is programming experience required?",
        "expected_keywords": ["software engineering", "must"],
        "should_fail": False
    },
    {
        "name": "MCP Components Test",
        "query": "List all MCP architecture components",
        "expected_keywords": ["hosts", "clients", "servers"],
        "should_fail": False
    },
    {
        "name": "Instructor Test",
        "query": "Who is the instructor and their background?",
        "expected_keywords": ["instructor", "experience"],
        "should_fail": False
    },
    {
        "name": "Hallucination Test",
        "query": "Does the course provide job placement?",
        "expected_keywords": ["not found"],
        "should_fail": True
    }
]

# =========================
# 12. EVALUATION
# =========================
def evaluate_response(response, expected_keywords, should_fail):
    response_lower = response.lower()

    match_score = sum(
        1 for kw in expected_keywords if kw.lower() in response_lower
    )

    keyword_score = match_score / len(expected_keywords)

    # partial credit
    if keyword_score == 0 and len(response) > 50:
        keyword_score = 0.5

    hallucination = False
    if should_fail and "not found" not in response_lower:
        hallucination = True

    return {
        "keyword_score": round(keyword_score, 2),
        "hallucination": hallucination
    }

# =========================
# 13. RUN TESTS
# =========================
results = []

print("\n🚀 Running ADVANCED RAG Test Suite...\n")

for test in test_cases:
    print(f"\n=== Running: {test['name']} ===")

    answer = run_rag_query(test["query"])

    evaluation = evaluate_response(
        answer,
        test["expected_keywords"],
        test["should_fail"]
    )

    results.append({
        "name": test["name"],
        "query": test["query"],
        "answer": answer,
        **evaluation
    })

# =========================
# 14. REPORT
# =========================
print("\n\n===== FINAL RAG TEST REPORT =====\n")

total_score = 0
hallucinations = 0

for r in results:
    print(f"Test: {r['name']}")
    print(f"Score: {r['keyword_score']}")
    print(f"Hallucination: {r['hallucination']}")
    print(f"Answer: {r['answer'][:200]}")
    print("-" * 50)

    total_score += r["keyword_score"]

    if r["hallucination"]:
        hallucinations += 1

avg_score = total_score / len(results)

print("\n===== SUMMARY =====")
print(f"Average Score: {round(avg_score, 2)}")
print(f"Hallucinations: {hallucinations}/{len(results)}")