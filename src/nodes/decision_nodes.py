from langgraph.graph import END
from langchain_ollama import ChatOllama
from src.graph.state import RobotDogState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Literal
from src.config import context_LLM_model, conversation_LLM_model, ollama_base_url
from src.graph.schemas import ContextProcessorOutput, DecisionNodeOutput, ConversationNodeOutput, ClarificationNodeOutput

def context_processor(state: RobotDogState) -> RobotDogState:
    """
    Process and normalize user input with structured output.
    Makes LLM-1 call to extract both context and intent classification.
    """

    query = state.get("original_query", "")

    messages = state.get("chat_history", []) # this messages is a local variable that stores the chat history for LLM calls

    # Combined prompt that extracts context AND classifies intent in ONE LLM call
    system_prompt = """You are an expert assistant that performs comprehensive query analysis:
        1. Extract context tags (keywords related to university services, robot functions, location info, person names, etc.)
        2. Classify the intent into one of: institutional, functional, ambiguous, or conversation
        3. Provide confidence scores for each intent type (must sum to ~1.0)
        4. Explain your reasoning for both context extraction and intent classification

        Intent definitions:
        - 'institutional': Query involves a person/entity that requires authentication/verification (e.g., "Is Dr. Smith available?")
        - 'functional': Query is a direct action command for the robot (e.g., "Take me to room 305", "Follow me", "Sit", "Stand", etc.)
        - 'ambiguous': Query is confusing, unclear, or lacks sufficient context
        - 'conversation': General conversational query or small talk (e.g., "How are you?", "What's your name?", "Tell me a joke", "what is the weather like?")"""

    user_prompt = f"""Analyze this user query: "{query}"

        Extract and provide:
        1. Context tags as a dictionary (e.g., {{"person": "Dr. Smith", "action": "navigation", "location": "room 305"}})
        2. Primary intent classification
        3. Confidence scores for all 4 intent types
        4. Reasoning explaining both your context extraction and intent classification"""

    messages.extend([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    # single LLM call that returns context + intent together
    llm = ChatOllama(
        model=state.get("context_LLM_model", context_LLM_model),
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.2,
    )

    structured_llm = llm.with_structured_output(ContextProcessorOutput)
    res = structured_llm.invoke(messages)

    # context data to the chat history
    response_content = f"""Context Tags: {res.context_tags}\nIntent: {res.intent}\nReasoning: {res.intent_reasoning}"""

    return {"context_proc_node_output": dict(res), 
            "chat_history": [SystemMessage(content=system_prompt), 
                             HumanMessage(content=user_prompt), 
                             AIMessage(content=response_content)]}

def exit_check(state: RobotDogState) -> RobotDogState:
    query = state.get("original_query", "").lower()
    if any(x in query for x in ["exit", "quit", "stop"]):
        print("[ExitCheck] Exit command detected.")
        return {"chat_history": ["exit command detected"]}
    return {}

def decision_node(state: RobotDogState) -> RobotDogState:
    """
    Extract decision from context processor output, just reads from context_proc_node_output.
    Performs additional rule-based logic or filtering if needed.
    """
    context_output = state.get("context_proc_node_output", {})
    
    # Extract intent-related fields from context processor output
    intent = context_output.get("intent", "ambiguous")
    confidence = context_output.get("confidence", {})
    intent_reasoning = context_output.get("intent_reasoning", "")
    
    # Optional: Add rule-based overrides or validation logic here
    # For example, if confidence is too low, override to ambiguous
    max_confidence = confidence.get(intent, 0.0)
    if max_confidence < 0.5:
        print(f"[DecisionNode] Low confidence ({max_confidence:.2f}), overriding to ambiguous")
        intent = "ambiguous"
        intent_reasoning += " | Overridden to ambiguous due to low confidence."
    
    # Optional: Add exit detection logic
    original_query = state.get("original_query", "").lower()
    if any(keyword in original_query for keyword in ["exit", "quit", "stop", "goodbye", "bye"]):
        print("[DecisionNode] Exit keyword detected in query")
        state["exit"] = True
    
    print(f"[DecisionNode] Final intent: {intent} | Confidence: {confidence}")
    
    # decision node output
    decision_output = DecisionNodeOutput(intent=intent, 
                                         confidence=confidence, 
                                         intent_reasoning=intent_reasoning)
    
    return {"decision_node_output": dict(decision_output)}

def conversation_node(state: RobotDogState) -> RobotDogState:
    """
    Generate conversational response with structured output.
    Uses LLM-2 for natural conversation.
    """
    query = state.get("original_query", "")
    context = state.get("context_proc_node_output", {})
    context_tags = context.get("context_tags", {})
    messages = state.get("chat_history", []) # this messages is a local variable that stores the chat history for LLM calls
    
    # Use LLM-2 to generate natural conversational response
    system_prompt = """You are a friendly, helpful robot assistant that engages in natural conversation.
        You can discuss various topics, answer general questions, make small talk, and be personable.

        Your personality:
        - Friendly and approachable
        - Professional but warm
        - Helpful and informative
        - Can handle casual conversation, jokes, weather talk, greetings, etc.

        Keep responses concise but engaging. If the user asks about your capabilities, mention you can help with navigation, finding people, and institutional information."""

    
    user_prompt = f"""User just said: "{query}"
        Context tags: {context_tags}
        Generate a natural, conversational response. Be friendly and engaging. Follow history for context."""

    messages.extend([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Use LLM-2 (conversation model) to generate response
    llm = ChatOllama(
        model=state.get("conversation_LLM_model", "qwen2.5-coder:1.5b"),
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.7,  # higher temperature for more natural conversation
    )

    structured_llm = llm.with_structured_output(ConversationNodeOutput)
    conversation_output = structured_llm.invoke(messages)
    
    # add response to history
    response_content = conversation_output.conversation_reply

    return {"conversation_node_output": dict(conversation_output),
            "chat_history": [SystemMessage(content=system_prompt),
                             HumanMessage(content=user_prompt), 
                             AIMessage(content=response_content)]}

def clarification_node(state: RobotDogState) -> RobotDogState:
    """
    Ask for clarification with structured output to the user. Formulate the question using LLM-1. 
    This node is called when the intent is "ambiguous".
    """
    query = state.get("original_query", "")
    context = state.get("context_proc_node_output", {})
    context_tags = context.get("context_tags", {})
    intent_reasoning = context.get("intent_reasoning", "")
    
    messages = state.get("chat_history", [])
    # Use LLM to generate a clarification question
    system_prompt = """You are a helpful robot assistant that generates clarification questions.
        When a user's query is ambiguous or unclear, you need to ask a specific, helpful question to understand their intent better.

        Your task is to:
        1. Generate a polite, specific clarification question
        2. Identify the type of clarification needed (e.g., "missing_location", "unclear_intent", "ambiguous_person", "multiple_possible_actions")

        Be conversational and helpful in your clarification question."""

    user_prompt = f"""The user said: "{query}"

        Context extracted: {context_tags}
        Why it's ambiguous: {intent_reasoning}

        Generate:
        1. A clarification question that will help me understand what the user wants
        2. The type of clarification needed (e.g., "missing_location", "unclear_intent", "ambiguous_person", etc.)

        Be specific and helpful."""

    messages.extend([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Use same LLM (LLM-1) to generate clarification question
    llm = ChatOllama(
        model=state.get("context_LLM_model", context_LLM_model),
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.3,  # slightly more creative than context processing
    )

    structured_llm = llm.with_structured_output(ClarificationNodeOutput)
    clarification_output = structured_llm.invoke(messages)

    response_content = f"""Clarification Question: {clarification_output.question}\nType: {clarification_output.clarify_type}"""

    
    return {"clarification_node_output": dict(clarification_output),
            "chat_history": [SystemMessage(content=system_prompt),
                             HumanMessage(content=user_prompt), 
                             AIMessage(content=response_content)]}

def decide_query_intention(state: RobotDogState) -> Literal["rag_node", "action_planner_node", "conversation_node", "clarification_node"]:
    intent = state.get("decision_node_output", {}).get("intent", "")
    if intent == "institutional":  # will have RAG
        return "rag_node"
    elif intent == "functional":  # will have MCP
        return "action_planner_node"
    elif intent == "ambiguous":
        return "clarification_node"  # will call LLM then TTS module (with clarify question)
    else:  # general conversation
        return "conversation_node"  # will call LLM then TTS module (with conversational reply)

def decide_mcp_execution(state: RobotDogState) -> Literal["mcp_llm_node", "speak_to_human_node"]:
    """
    Decide whether to execute MCP based on action_input from action_classifier.
    If action is not needed (requires_robot_action=False), skip MCP and go directly to speaking.
    """
    action_input_data_mcp = state.get("action_input_to_mcp", {}) # most of this is empty if the robot action is not needed
    requires_action = action_input_data_mcp.get("requires_robot_action", False)
    action_intent = action_input_data_mcp.get("action_intent", "no_action_needed")
    
    if not requires_action or action_intent == "no_action_needed":
        return "speak_to_human_node"
    else:
        return "mcp_llm_node"
    
def should_continue(state: RobotDogState) -> Literal["listen_to_human_node", END]:
    """Decide if we should continue the loop or stop based upon whether human said to quit."""

    if state.get("exit", False):
        return END
    # Otherwise, we continue listening to the human
    return "listen_to_human_node"