from src.graph.state import RobotDogState
from src.graph.schemas import PerceptionFeedbackOutput, SummarizerNodeOutput

def perception_feedback(state: RobotDogState) -> RobotDogState:
    """
    Process sensor feedback with structured output.
    """
    print("[Feedback] Processing sensor feedback ...")
    
    robot_status = state.get("robot_status", "unknown")
    tool_result = state.get("tool_result", {})
    
    # Create structured output
    perception_output = PerceptionFeedbackOutput(
        feedback="Robot reached target successfully.",
        sensor_data={
            "position": "target_location",
            "status": robot_status,
            "tool_status": tool_result.get("status", "unknown")
        },
        anomalies_detected=False
    )
    
    return {
        "feedback": perception_output.feedback,
        "sensor_feedback": str(perception_output.sensor_data),
        "perception_output": perception_output
    }

def summarizer_node(state: RobotDogState) -> RobotDogState:
    """
    Summarize execution results with structured output.
    Uses LLM-4 for summarization.
    """
    feedback = state.get("feedback", "")
    tool_result = state.get("tool_result", {})
    action_intent = state.get("action_intent", "action")
    
    summary = f"Status update: {feedback}"
    final_response = f"I have completed the {action_intent}. {feedback}"
    
    print(f"[Summarizer] {summary}")
    
    # Create structured output
    summarizer_output = SummarizerNodeOutput(
        final_response=final_response,
        summary=summary,
        success=(tool_result.get("status") == "success"),
        metadata={
            "action": action_intent,
            "robot_status": state.get("robot_status", "unknown")
        }
    )
    
    return {
        "final_response": summarizer_output.final_response,
        "summary": summarizer_output.summary,
        "summarizer_output": summarizer_output
    }
