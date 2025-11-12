from src.graph.state import RobotDogState

def rag_pipeline(state: RobotDogState) -> RobotDogState:
    query = state["query"]
    print(f"[RAG] Searching vector DB for '{query}' ... (stub)")
    # Normally you'd use: Chroma/FAISS + SentenceTransformerEmbeddings
    results = ["Mock result about university policy."]
    return {"response": f"RAG result: {results[0]}"}
