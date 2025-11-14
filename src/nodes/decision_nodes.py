from langgraph.graph import END
from langchain_ollama import ChatOllama
from src.graph.state import RobotDogState
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Literal
from src.config import context_LLM_model, ollama_base_url
from src.graph.schemas import ContextProcessorOutput, DecisionNodeOutput, ConversationNodeOutput, ClarificationNodeOutput

def context_processor(state: RobotDogState) -> RobotDogState:
    """
    Process and normalize user input with structured output.
    Uses LLM-1 for context extraction.
    """
    chat = state.get("stt_node_output", {}).get("original_query", "")
    context_proc_node_output = state.get("context_proc_node_output")

    # LLM message prompt, each LLM may have different system instructions
    messages = [
        SystemMessage(content="You are an expert assistant that extracts context from user queries, normalize the query, derive context tags."),
        HumanMessage(content=f"User query: {chat}\nExtract the main intent and relevant context tags. Provide a normalized version of the query. \
                     Context tags means keywords related to university services, robot functions, location info, person names, etc."),
    ]

    # call LLM-1 to process text and extract context
    llm = ChatOllama(
        model=state.get("context_LLM_model", context_LLM_model),
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.2,
    )

    structured_llm = llm.with_structured_output(ContextProcessorOutput)
    res = structured_llm.invoke(messages)
    return {"context_proc_node_output": dict(res)}

def exit_check(state: RobotDogState) -> RobotDogState:
    query = state.get("original_query", "").lower()
    if any(x in query for x in ["exit", "quit", "stop"]):
        print("[ExitCheck] Exit command detected.")
        return {"chat_history": ["exit command detected"]}
    return {}

def decision_node(state: RobotDogState) -> RobotDogState:
    """
    Classify user intent with structured output and confidence scoring.
    """
    normalized_query = state.get("context_proc_node_output", {}).get("normalized_query", "")
    context_tags = state.get("context_proc_node_output", {}).get("context_tags", {})

    decision_node_output = state.get("decision_node_output")

    # LLM message prompt, each LLM may have different system instructions
    messages = [
        SystemMessage(content="You are an expert assistant that extracts intent (e.g., institutional, functional, ambiguous, conversation) from normalized queries and context tags. \
                      Provide confidence scores for each intent. If unsure, classify as ambiguous. Also do reasoning for your classification."),
        HumanMessage(content=f"Normalized User query: {normalized_query}\nExtract the main intent and relevant context tags. \
                     Context tags: {context_tags}. Context tags means keywords related to university, robot functions, location info, person names, etc. \
                        \n NOTE: \n intent is 'institutional' if its a person, so that we can check its authenticity, 'functional' if the query is a direct action only, 'ambiguous' if the query is confusion and unclear, 'conversation' if its a general talk."),
    ]

    # call LLM-1 to process text and extract context
    llm = ChatOllama(
        model=state.get("context_LLM_model", context_LLM_model),
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.3,
    )

    structured_llm = llm.with_structured_output(DecisionNodeOutput)
    res = structured_llm.invoke(messages)
    return {"decision_node_output": dict(res)}

def conversation_node(state: RobotDogState) -> RobotDogState:
    """
    Generate conversational response with structured output.
    Uses LLM-2 for natural conversation.
    """
    q = state.get("original_query", "")
    print(f"[ConversationNode] Handling conversation: {q}")
    
    response = f"Conversational reply to '{q}'"
    
    # Create structured output
    conversation_output = ConversationNodeOutput(
        final_response=response,
        chat_history_entry=f"User: {q}\nBot: {response}",
        metadata={"timestamp": "now", "node": "conversation"}
    )
    
    return {
        "final_response": conversation_output.final_response,
        "chat_history": [conversation_output.chat_history_entry],
        "conversation_output": conversation_output
    }

def clarification_node(state: RobotDogState) -> RobotDogState:
    """
    Ask for clarification with structured output.
    """
    q = state.get("original_query", "")
    print(f"[ClarificationNode] Asking for clarification on: {q}")
    
    response = "Could you please clarify your request?"
    
    # Create structured output
    clarification_output = ClarificationNodeOutput(
        final_response=response,
        clarification_type="ambiguous_intent",
        original_query=q
    )
    
    return {
        "final_response": clarification_output.final_response,
        "chat_history": [response],
        "clarification_output": clarification_output
    }

def decide_query_intention(state: RobotDogState) -> Literal["rag_node", "action_planner_node", "conversation_node", "clarification_node"]:
    intent = state.get("decision_node_output", {}).get("intent", "")
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