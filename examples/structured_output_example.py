"""
Example: Using Structured Output with LLM in LangGraph Nodes

This example demonstrates how to use Pydantic schemas with LLM.with_structured_output()
to ensure LLM responses match your node's expected output format.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.graph.state import RobotDogState
from src.graph.schemas import (
    DecisionNodeOutput,
    ActionPlannerOutput,
    RAGNodeOutput,
    ConversationNodeOutput
)


# ============================================
# Example 1: Decision Node with Structured LLM Output
# ============================================

def decision_node_with_llm(state: RobotDogState) -> RobotDogState:
    """
    Use LLM with structured output for intent classification.
    """
    query = state.get("original_query", "")
    
    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for a robot assistant.
        Classify the user's intent into one of these categories:
        - conversation: General chat, greetings, small talk
        - functional: Robot actions like move, turn, pick up
        - institutional: Questions about university, courses, locations
        - ambiguous: Unclear or needs clarification
        
        Provide a confidence score (0.0 to 1.0) and reasoning."""),
        ("user", "{query}")
    ])
    
    # Create LLM with structured output matching our schema
    llm = ChatOllama(model="llama3.2", temperature=0)
    structured_llm = llm.with_structured_output(DecisionNodeOutput)
    
    # Create chain
    chain = prompt | structured_llm
    
    # Get structured response - guaranteed to match DecisionNodeOutput schema
    decision_output: DecisionNodeOutput = chain.invoke({"query": query})
    
    print(f"[DecisionNode] Intent: {decision_output.intent} ({decision_output.confidence:.2f})")
    print(f"[DecisionNode] Reasoning: {decision_output.reasoning}")
    
    return {
        "intent": decision_output.intent,
        "confidence": decision_output.confidence,
        "ambiguous": decision_output.ambiguous,
        "decision_output": decision_output
    }


# ============================================
# Example 2: Action Planner with Structured LLM Output
# ============================================

def action_planner_with_llm(state: RobotDogState) -> RobotDogState:
    """
    Use LLM to generate structured action plans.
    """
    query = state.get("original_query", "")
    context_tags = state.get("context_tags", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a robot action planner. Create detailed action plans.
        
        Return a structured plan with:
        - action_intent: High-level description of the action
        - plan: Dictionary with keys 'action', 'target', 'method', 'parameters'
        - action_sequence: List of step-by-step actions
        - estimated_duration: Time in seconds
        - requires_confirmation: Boolean if action needs approval
        
        Context tags: {context_tags}"""),
        ("user", "{query}")
    ])
    
    llm = ChatOllama(model="llama3.2", temperature=0)
    structured_llm = llm.with_structured_output(ActionPlannerOutput)
    
    chain = prompt | structured_llm
    
    # Get structured action plan
    planner_output: ActionPlannerOutput = chain.invoke({
        "query": query,
        "context_tags": ", ".join(context_tags)
    })
    
    print(f"[ActionPlanner] Plan: {planner_output.plan}")
    print(f"[ActionPlanner] Steps: {planner_output.action_sequence}")
    print(f"[ActionPlanner] Duration: {planner_output.estimated_duration}s")
    
    return {
        "action_intent": planner_output.action_intent,
        "plan": planner_output.plan,
        "action_sequence": planner_output.action_sequence,
        "needs_confirmation": planner_output.requires_confirmation,
        "action_planner_output": planner_output
    }


# ============================================
# Example 3: RAG Pipeline with Structured LLM Output
# ============================================

def rag_pipeline_with_llm(state: RobotDogState) -> RobotDogState:
    """
    Use LLM to generate RAG responses with structured output.
    """
    query = state.get("original_query", "")
    
    # Step 1: Retrieve documents (mock for example)
    retrieved_docs = [
        "The library is located in Building A, 2nd floor.",
        "Library hours: Monday-Friday 8am-8pm, Saturday 10am-6pm."
    ]
    
    # Step 2: Generate answer with structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions using provided documents.
        
        Context documents:
        {documents}
        
        Provide:
        - rag_result: Your generated answer based on documents
        - final_response: A formatted, conversational response
        - sources: List of source identifiers used
        - confidence: How confident you are (0.0 to 1.0)
        """),
        ("user", "{query}")
    ])
    
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    structured_llm = llm.with_structured_output(RAGNodeOutput)
    
    chain = prompt | structured_llm
    
    # Get structured RAG response
    rag_output: RAGNodeOutput = chain.invoke({
        "query": query,
        "documents": "\n".join(f"[{i}] {doc}" for i, doc in enumerate(retrieved_docs))
    })
    
    # Override retrieved_docs with actual retrieval
    rag_output.retrieved_docs = retrieved_docs
    
    print(f"[RAG] Answer: {rag_output.rag_result}")
    print(f"[RAG] Confidence: {rag_output.confidence:.2f}")
    print(f"[RAG] Sources: {rag_output.sources}")
    
    return {
        "retrieved_docs": rag_output.retrieved_docs,
        "rag_result": rag_output.rag_result,
        "final_response": rag_output.final_response,
        "rag_output": rag_output
    }


# ============================================
# Example 4: Conversation Node with Structured LLM Output
# ============================================

def conversation_node_with_llm(state: RobotDogState) -> RobotDogState:
    """
    Generate conversational responses with structured output.
    """
    query = state.get("original_query", "")
    chat_history = state.get("chat_history", [])
    
    # Format chat history
    history_text = "\n".join(chat_history[-5:]) if chat_history else "No previous conversation"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly robot assistant. Generate natural, helpful responses.
        
        Previous conversation:
        {history}
        
        Provide:
        - final_response: Your conversational reply
        - chat_history_entry: Formatted entry for history (User: X, Bot: Y)
        - metadata: Any relevant metadata
        """),
        ("user", "{query}")
    ])
    
    llm = ChatOllama(model="llama3.2", temperature=0.7)
    structured_llm = llm.with_structured_output(ConversationNodeOutput)
    
    chain = prompt | structured_llm
    
    # Get structured conversation response
    conversation_output: ConversationNodeOutput = chain.invoke({
        "query": query,
        "history": history_text
    })
    
    print(f"[Conversation] Response: {conversation_output.final_response}")
    
    return {
        "final_response": conversation_output.final_response,
        "chat_history": [conversation_output.chat_history_entry],
        "conversation_output": conversation_output
    }


# ============================================
# Example 5: Complete Workflow Test
# ============================================

def test_structured_workflow():
    """
    Test the complete workflow with structured outputs.
    """
    from src.graph.workflow import build_robotdog_workflow_graph
    
    # Build the workflow
    workflow = build_robotdog_workflow_graph()
    
    # Initial state
    initial_state = {
        "original_query": "Where is the library?",
        "exit": False
    }
    
    # Run the workflow
    print("\n" + "="*60)
    print("Running RobotDog Workflow with Structured Outputs")
    print("="*60 + "\n")
    
    result = workflow.invoke(initial_state)
    
    # Inspect structured outputs
    print("\n" + "="*60)
    print("Structured Output Results")
    print("="*60 + "\n")
    
    # Decision output
    if decision := result.get("decision_output"):
        print(f"Decision Intent: {decision.intent}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}\n")
    
    # RAG output
    if rag := result.get("rag_output"):
        print(f"RAG Result: {rag.rag_result}")
        print(f"Sources: {rag.sources}")
        print(f"Confidence: {rag.confidence:.2f}\n")
    
    # Action output
    if action := result.get("action_planner_output"):
        print(f"Action Plan: {action.plan}")
        print(f"Steps: {action.action_sequence}")
        print(f"Duration: {action.estimated_duration}s\n")
    
    # Final response
    print(f"Final Response: {result.get('final_response', 'N/A')}")
    
    return result


# ============================================
# Example 6: Error Handling with Structured Output
# ============================================

from pydantic import ValidationError

def safe_structured_node(state: RobotDogState) -> RobotDogState:
    """
    Example of handling validation errors in structured output.
    """
    query = state.get("original_query", "")
    
    try:
        # Attempt to create structured output
        decision_output = DecisionNodeOutput(
            intent="conversation",
            confidence=0.85,
            ambiguous=False,
            reasoning="General conversation detected"
        )
        
        return {
            "intent": decision_output.intent,
            "confidence": decision_output.confidence,
            "decision_output": decision_output
        }
        
    except ValidationError as e:
        # Handle validation errors gracefully
        print(f"[ERROR] Validation failed: {e}")
        
        # Return safe defaults
        fallback_output = DecisionNodeOutput(
            intent="ambiguous",
            confidence=0.0,
            ambiguous=True,
            reasoning="Validation error occurred"
        )
        
        return {
            "intent": fallback_output.intent,
            "confidence": fallback_output.confidence,
            "ambiguous": True,
            "decision_output": fallback_output
        }


# ============================================
# Usage Instructions
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  LangGraph Structured Output Examples                     ║
    ╚═══════════════════════════════════════════════════════════╝
    
    This file demonstrates how to use structured outputs with LLMs.
    
    Key Benefits:
    ✓ Type-safe outputs validated by Pydantic
    ✓ LLM responses match expected schema
    ✓ Better error handling and debugging
    ✓ Self-documenting code
    
    To use in your nodes:
    
    1. Import the schema:
       from src.graph.schemas import YourNodeOutput
    
    2. Create structured LLM:
       structured_llm = llm.with_structured_output(YourNodeOutput)
    
    3. Use in chain:
       output = chain.invoke(inputs)
    
    4. Return both raw and structured:
       return {
           "field": output.field,
           "your_node_output": output
       }
    
    Run individual examples:
    - test_structured_workflow()
    - Or integrate into your workflow nodes
    """)
    
    # Uncomment to run test:
    # test_structured_workflow()
