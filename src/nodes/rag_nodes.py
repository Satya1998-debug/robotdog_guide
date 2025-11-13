from src.graph.state import RobotDogState

def rag_pipeline(state: RobotDogState) -> RobotDogState:
    query = state.get("original_query", "")
    print(f"[RAG] Searching vector DB for '{query}' ... (stub)")
    results = ["Mock result about university policy."]
    return {"final_response": f"RAG result: {results[0]}"}
