from src.graph.state import RobotDogState
from src.graph.schemas import ActionClassifierOutput, ActionPlannerOutput, MCPToolOutput

def action_classifier(state: RobotDogState) -> RobotDogState:
    """
    Classify action type with structured output.
    """
    response = state.get("final_response", "")
    query = state.get("original_query", "")
    print(f"[ActionClassifier] Classifying response: {response}")
    
    # Classification logic
    if "move" in response.lower() or "go" in response.lower():
        action_type = "navigation"
        action_intent = "navigate_to_location"
        confidence = 0.90
    elif "pick" in response.lower() or "grab" in response.lower():
        action_type = "manipulation"
        action_intent = "manipulate_object"
        confidence = 0.85
    elif "look" in response.lower() or "see" in response.lower():
        action_type = "perception"
        action_intent = "perceive_environment"
        confidence = 0.80
    else:
        action_type = "other_tools"
        action_intent = "use_tool"
        confidence = 0.70
    
    # Create structured output
    classifier_output = ActionClassifierOutput(
        action_intent=action_intent,
        action_type=action_type,
        confidence=confidence
    )
    
    return {
        "action_intent": classifier_output.action_intent,
        "action_classifier_output": classifier_output
    }

def action_planner(state: RobotDogState) -> RobotDogState:
    """
    Create detailed action plan with structured output.
    Uses LLM-3 for action planning.
    """
    q = state.get("original_query", "")
    print(f"[ActionPlanner] Creating action plan for: {q}")
    
    # Simple placeholder plan
    plan = {
        "action": "move_forward",
        "target": "destination",
        "method": "autonomous_navigation"
    }
    action_sequence = ["initialize", plan["action"], "verify_completion"]
    
    # Create structured output
    planner_output = ActionPlannerOutput(
        action_intent="navigate_forward",
        plan=plan,
        action_sequence=action_sequence,
        estimated_duration=5.0,
        requires_confirmation=False
    )
    
    return {
        "action_intent": planner_output.action_intent,
        "plan": planner_output.plan,
        "action_sequence": planner_output.action_sequence,
        "needs_confirmation": planner_output.requires_confirmation,
        "action_planner_output": planner_output
    }

def mcp_client(state: RobotDogState) -> RobotDogState:
    """
    Execute MCP tool calls with structured output.
    """
    action_intent = state.get("action_intent", "unknown")
    action_sequence = state.get("action_sequence", [])
    
    print(f"[MCPClient] Executing tool for: {action_intent}")
    print(f"[MCPClient] Action sequence: {action_sequence}")
    
    # Mock tool execution
    tool_name = "robot_control_tool"
    result = {
        "status": "success",
        "message": f"Executed {action_intent}",
        "actions_completed": str(len(action_sequence))
    }
    
    # Create structured output
    mcp_output = MCPToolOutput(
        tool_called=tool_name,
        tool_result=result,
        robot_status="idle",
        execution_success=True,
        error_message=None
    )
    
    return {
        "tool_called": mcp_output.tool_called,
        "tool_result": mcp_output.tool_result,
        "robot_status": mcp_output.robot_status,
        "mcp_output": mcp_output
    }
