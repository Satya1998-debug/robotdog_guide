
def perception_feedback(state):
    print("[Feedback] Processing sensor feedback ...")
    return {"feedback": "Robot reached target."}

def summarizer_node(state):
    feedback = state.get("feedback", "")
    summary = f"Status update: {feedback}"
    print(f"[Summarizer] {summary}")
    return {"response": summary}
