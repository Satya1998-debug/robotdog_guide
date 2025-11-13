from langgraph.graph import END
from src.graph.state import RobotDogState
from typing import Literal

def context_processor(state: RobotDogState) -> RobotDogState:
    # use LLM-1
    text = state.get("original_query", "")
    print(f"[ContextProcessor] Received text: {text}")
    return {"original_query": text}

def exit_check(state: RobotDogState) -> RobotDogState:
    query = state.get("original_query", "").lower()
    if any(x in query for x in ["exit", "quit", "stop"]):
        print("[ExitCheck] Exit command detected.")
        return {"chat_history": ["exit command detected"]}
    return {}

def decision_node(state: RobotDogState) -> RobotDogState:
    query = state.get("original_query", "").lower()
    if any(x in query for x in ["where", "info", "course", "university"]):
        intent = "institutional"
    elif any(x in query for x in ["walk", "move", "go", "turn"]):
        intent = "functional"
    else:
        intent = "conversation"
    print(f"[DecisionNode] Intent detected: {intent}")
    return {"node_sequence": ["decision_node"]}

def conversation_node(state: RobotDogState) -> RobotDogState:
    # use LLM-2
    q = state.get("original_query", "")
    print(f"[ConversationNode] Handling conversation: {q}")
    response = f"Conversational reply to '{q}'"
    return {"final_response": response, "chat_history": [response]}

def clarification_node(state: RobotDogState) -> RobotDogState:
    q = state.get("original_query", "")
    print(f"[ClarificationNode] Asking for clarification on: {q}")
    response = "Could you please clarify your request?"
    return {"final_response": response, "chat_history": [response]}

def decide_query_intention(state: RobotDogState) -> Literal["rag_node", "action_planner_node", "conversation_node", "clarification_node"]:
    intent = state["intent"]
    if intent == "institutional":
        return "rag_node"
    elif intent == "functional":
        return "action_planner_node"
    elif intent == "ambiguous":
        return "clarification_node"
    else:  # general conversation
        return "conversation_node"
    
    
def should_continue(state: RobotDogState) -> Literal["listen_to_human_node", END]:
    """Decide if we should continue the loop or stop based upon whether human said to quit."""

    if state.get("exit", False):
        print("[ShouldContinue] Exit signal received. Ending interaction.")
        return END
    # Otherwise, we continue listening to the human
    return "listen_to_human_node"