from src.graph.state import RobotDogState

# MCP
def mcp_client(state: RobotDogState) -> RobotDogState:
    pass

# uses LLM-3

def action_planner(state: RobotDogState) -> RobotDogState:
    q = state["original_query"]
    print(f"[ActionPlanner] Creating action plan for: {q}")
    plan = {"action": "move_forward"}  # simple placeholder
    return {"plan": plan}

def action_classifier(state: RobotDogState) -> RobotDogState:
    response = state["response"]
    print(f"[ActionClassifier] Classifying response: {response}")
    if "move" in response:
        action_type = "navigation"
    else:
        action_type = "other_tools"
    return {"action_type": action_type}
