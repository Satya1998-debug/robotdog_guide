from src.graph.state import RobotDogState
from typing import Literal

def context_processor(state: RobotDogState) -> RobotDogState:
    # use LLM-1
    text = state["input_text"]
    print(f"[ContextProcessor] Received text: {text}")
    return {"query": text, "context_tags": []}

def exit_check(state: RobotDogState) -> RobotDogState:
    query = state["query"].lower()
    if any(x in query for x in ["exit", "quit", "stop"]):
        print("[ExitCheck] Exit command detected.")
        return {"exit": True}
    return {"exit": False}

def decision_node(state: RobotDogState) -> RobotDogState:
    query = state["query"].lower()
    if any(x in query for x in ["where", "info", "course", "university"]):
        intent = "institutional"
    elif any(x in query for x in ["walk", "move", "go", "turn"]):
        intent = "functional"
    else:
        intent = "conversation"
    print(f"[DecisionNode] Intent detected: {intent}")
    return {"intent": intent}

def conversation_node(state: RobotDogState) -> RobotDogState:
    # use LLM-2
    q = state["query"]
    print(f"[ConversationNode] Handling conversation: {q}")
    return {"response": f"Conversational reply to '{q}'"}

def clarification_node(state: RobotDogState) -> RobotDogState:
    q = state["query"]
    print(f"[ClarificationNode] Asking for clarification on: {q}")
    return {"response": "Could you please clarify your request?"}


def decide_query_intention(state: RobotDogState) -> Literal["rag_node", "action_planner_node", 
                                                            "conversation_node", "clarification_node"]:
    
    intent = state["intent"]
    if intent == "institutional":
        return "rag_node"
    elif intent == "functional":
        return "action_planner_node"
    elif intent == "ambiguous":
        return "clarification_node"
    else:  # general conversation
        return "conversation_node"
