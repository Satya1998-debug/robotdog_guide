from src.graph.state import RobotDogState

def perception_feedback(state: RobotDogState) -> RobotDogState:
    print("[Feedback] Processing sensor feedback ...")
    return {"feedback": "Robot reached target."}

def summarizer_node(state: RobotDogState) -> RobotDogState:
    feedback = state.get("feedback", "")
    summary = f"Status update: {feedback}"
    print(f"[Summarizer] {summary}")
    return {"final_response": summary}
