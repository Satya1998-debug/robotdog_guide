from src.graph.state import RobotDogState
from src.graph.schemas import RAGNodeOutput

def rag_pipeline(state: RobotDogState) -> RobotDogState:
    """
    Retrieve and generate response using RAG with structured output.
    Uses LLM-3 for RAG generation.
    """
    query = state.get("original_query", "")
    print(f"[RAG] Searching vector DB for '{query}' ... (stub)")
    
    # Mock retrieval
    results = ["Mock result about university policy."]
    rag_response = f"Based on the university policy: {results[0]}"
    
    # Create structured output
    rag_output = RAGNodeOutput(
        retrieved_docs=results,
        rag_result=rag_response,
        final_response=rag_response,
        sources=["university_policy_db"],
        confidence=0.88
    )
    
    return {
        "retrieved_docs": rag_output.retrieved_docs,
        "rag_result": rag_output.rag_result,
        "final_response": rag_output.final_response,
        "rag_output": rag_output
    }
