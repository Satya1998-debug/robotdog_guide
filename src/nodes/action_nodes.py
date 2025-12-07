from src.graph.state import RobotDogState
from src.graph.schemas import ActionInputToToolsLLM
from src.tools_servers.tools import get_all_tools
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.config import ollama_base_url, action_planner_LLM_model, tool_LLM_model, ACTION_CONFIDENCE_THRESHOLD
from src.logger import logger

# LLM with tools
# Create LLM with tools bound (LangGraph pattern) once here
llm = ChatOllama(
        model=tool_LLM_model,  # LLM-5
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.2,
    )
llm_with_tools = llm.bind_tools(get_all_tools())


def action_classifier(state: RobotDogState) -> RobotDogState:
    """
    Classify action type based on RAG output and determine if MCP execution is needed.
    Checks action threshold from RAG node to decide on robot action.
    """
    logger.info("[Node] -> action_classifier_node")
    rag_node_output = state.get("rag_node_output", {})
    
    # Extract action decision from RAG
    requires_robot_action = rag_node_output.get("requires_robot_action", False)
    action_confidence = rag_node_output.get("action_confidence", 0.0)
    target_location = rag_node_output.get("target_location", None)
    target_person = rag_node_output.get("target_person", None)
    probable_actions = rag_node_output.get("probable_actions", [])
    
    query_lower = rag_node_output.get("rag_modified_query", "").lower()
    
    logger.info(f"[action_classifier] Threshold check: confidence={action_confidence:.2f} vs {ACTION_CONFIDENCE_THRESHOLD} | Requires action: {requires_robot_action}")
    
    # if no acton needed, then route to tts node
    if not requires_robot_action or action_confidence < ACTION_CONFIDENCE_THRESHOLD:
        # no robot action needed - just informational response to the converation node
        action_input_to_tools_llm = ActionInputToToolsLLM(
            rag_modified_query=query_lower,
            action_intent="no_action_needed",
            action_type="",
            requires_robot_action=False,
            action_confidence=action_confidence,
            target_location=target_location,
            target_person=target_person,
            probable_actions=[]
        )
        return {"informational_response": rag_node_output.get("informational_response", ""),
                "action_input_to_tools_llm": dict(action_input_to_tools_llm)}
    
    else:
    
        # Robot action is needed - classify action type
                
        # fallback filters for action type classification
        if target_location or "navigation" in probable_actions or \
            any(keyword in query_lower for keyword in ["take me", "go to", "navigate", "find room", "bring me"]):
            action_type = "navigation"
            action_intent = f"navigate_to_{target_location or 'location'}"
            
        elif target_person or "navigation" in probable_actions or \
            any(keyword in query_lower for keyword in ["find person", "locate", "where is", "follow"]):
            action_type = "navigation"
            action_intent = f"find_person_{target_person or 'unknown'}"
            
        else:  # default to other tools
            action_type = "other_tools"
            action_intent = "execute_basic_tools"
        
        logger.info(f"[action_classifier] Action type: {action_type} | Intent: {action_intent} | Location: {target_location} | Person: {target_person}")
                
        # Create ActionInput with all RAG data
        action_input_to_tools_llm = ActionInputToToolsLLM(
            rag_modified_query=query_lower,
            action_intent=action_intent,
            action_type=action_type,
            requires_robot_action=True,
            action_confidence=action_confidence,
            target_location=target_location,
            target_person=target_person,
            probable_actions=probable_actions
        )
        
        return {"action_input_to_tools_llm": action_input_to_tools_llm}

def action_planner(state: RobotDogState) -> RobotDogState:
    """
    Create detailed action plan with structured output using LLM.
    This node is for direct 'functional' intent queries (bypasses RAG and action_classifier).
    Uses LLM-4 for action planning and outputs ActionInput for tools node.
    """
    logger.info("[Node] -> action_planner_node")
    query = state.get("original_query", "")
    context_output = state.get("context_proc_node_output", {})
    context_tags = context_output.get("context_tags", {})
    intent_reasoning = state.get("decision_node_output", {}).get("intent_reasoning", "")

    messages = []

    if state.get("summary", ""):  # insert the summary first
        summary_system_msg = f"Previous conversation summary: {state['summary']}"
        messages.append(SystemMessage(content=summary_system_msg))
    
    messages.extend(state.get("chat_history", [])) # include prior chat history after previous session's summary
    
    # Use LLM to generate action input with structured output
    system_prompt = """You are a robot action planner that analyzes user commands and creates action inputs for robot execution.

        Your role is to:
        1. Analyze the user's direct command/request
        2. Determine the high-level action intent (e.g., "follow_human", "navigate_to_location")
        3. Classify the action type: navigation, or other_tools
        4. Extract target location or person if mentioned
        5. Generate list of probable robot actions needed
        6. Provide a brief informational response summarizing the action
        7. Set action confidence (0.0-1.0) based on clarity of command

        For functional/direct commands, always set requires_robot_action=True and action_confidence >= 0.8

        Action type categories:
        - manipulation: picking, placing, grabbing, holding objects  
        - perception: scanning, looking, inspecting, observing
        - other_tools: speak, stand, sit, crawl, dance, etc.

        Be specific and decisive for direct robot commands."""

    user_prompt = f"""User command: "{query}"
        Context tags: {context_tags}
        Intent reasoning: {intent_reasoning}

        Analyze this direct robot command and provide:
        1. action_intent: High-level intent (e.g., "follow_human", "navigate_to_kitchen")
        2. action_type: Category from [navigation, manipulation, perception, other_tools]
        3. requires_robot_action: True (this is a direct command)
        4. action_confidence: 0.8-1.0 (high confidence for direct commands)
        5. target_location: Extract location if mentioned (or null)
        6. target_person: Extract person name if mentioned (or null)
        7. informational_response: Brief summary of what robot will do
        8. List probable robot actions based on the query: (keep the list empty if no action needed)
           - "navigation", or "others" etc. 
           - "other tools" like stand, sit, crawl, speak, etc.
        """

    messages.extend([  # include current node's msgs
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Use LLM-4 (action planning model) to generate ActionInputToMCP
    action_llm = ChatOllama(
        model=state.get("action_planner_LLM_model", action_planner_LLM_model),  # LLM-4
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.3,  # Moderate temperature for creative but reliable planning
    )

    structured_llm = action_llm.with_structured_output(ActionInputToMCP)
    action_input_to_mcp = structured_llm.invoke(messages)

    response_content = f"""action inetnt: {action_input_to_mcp.action_intent}\n\
        action type: {action_input_to_mcp.action_type}\n"""
    
    return {"action_input_to_mcp": dict(action_input_to_mcp), 
            "chat_history": [SystemMessage(content="Action planner node: You are a helpful assistant that plans robot actions."),
                             HumanMessage(content="Analyze the user command and plan the robot actions accordingly."), 
                             AIMessage(content=response_content)]}  

def call_llm_with_tools(state: RobotDogState) -> RobotDogState:
    """
    MCP model node following LangGraph documentation pattern.
    
    Pattern:
    1. Get MCP tools from global cache (initialized in workflow)
    2. Bind tools to LLM using bind_tools()
    3. Build context-rich messages from action_input_to_mcp
    4. LLM decides which tools to call (or none)
    5. Returns with tool calls in messages
    6. tools_condition router will direct to ToolNode if tools needed
    7. ToolNode executes tools and returns here for next iteration
    
    This is the LangGraph recommended pattern for MCP integration.
    """
    logger.info("[Node] -> llm_tools_node")
        
    # Get action context
    action_input_data = state.get("action_input_to_mcp", {})
    original_query = state.get("original_query", "")
    
    # Extract action details
    action_intent = action_input_data.get("action_intent", "")
    action_type = action_input_data.get("action_type", "")
    target_location = action_input_data.get("target_location")
    target_person = action_input_data.get("target_person")
    probable_actions = action_input_data.get("probable_actions", [])
    rag_modified_query = action_input_data.get("rag_modified_query", original_query)

    messages = state.get("chat_history", [])
    
    # Build context-rich system message
    system_msg = f"""You are a RobotDog assistant executing physical robot actions.
        Current Action Context:
        - Action Intent: {action_intent}
        - Action Type: {action_type}
        - Target Location: {target_location or 'Not specified'}
        - Target Person: {target_person or 'Not specified'}
        - Probable Actions: {', '.join(probable_actions) if probable_actions else 'None'}

        Execute the appropriate LangChain tools to complete this action. Guidelines:
        - For navigation: call stand_up() first, then navigate_to(x, y), then get_sensor_data()
        - For sit/stand commands: call the respective tool directly
        - For emergency: call emergency_stop() immediately
        - Always be sequential - one tool at a time

        Available LangChain tools: stand_up, sit_down, navigate_to, emergency_stop, get_sensor_data"""
    
    # Build user message
    user_msg = f"""Execute this robot action: {rag_modified_query}
        Action Details:
        - Original query: {original_query}
        - Action intent: {action_intent}
        - Target: {target_location or target_person or 'Not specified'}

        Use the available tools to complete this action step by step."""
    
    messages.extend([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ])
    
    # Invoke LLM with tools already bound, LLM will decide which tools to call
    response = llm_with_tools.invoke(messages)
    
    # Check what tools LLM decided to call
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc.name for tc in response.tool_calls]
        logger.info(f"[llm_tools_node] LLM decided tools: {tool_names}")
    else:
        logger.info(f"[llm_tools_node] LLM decided: NO TOOLS (end of loop)")

    # response_content = f"""Tool response: {response.toolcall_response}\nTools called: {response.tools_called}"""
    
    # Return response with messages
    # tools_condition will check if there are tool_calls and route accordingly
    return {"messages": response,  # this message field is only the tool node to check the last message and execute tools if any call is made
            "chat_history": [SystemMessage(content="You are a helpful assistant that calls tools for robot actions."),
                             HumanMessage(content="Do the tool calls accordingly."), 
                             response]}  # chat history will already have all previous messages and will add the tool call response