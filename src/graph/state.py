from typing import Any, TypedDict, List, Dict, Optional, Annotated
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


from src.graph.schemas import (
    ActionInputToToolsLLM,
    SpeechToTextOutput,
    ContextProcessorOutput,
    DecisionNodeOutput,
    ConversationNodeOutput,
    ClarificationNodeOutput,
    RAGNodeOutput,
)


class RobotDogState(TypedDict):
    """
    LangGraph state with structured outputs for each node, each field is either a single structured output or a list that accumulates outputs.
    """

    start_conversation: bool  # true if starting new conversation, of pursuit existing
    original_query: str       # raw user input query

    # Speech Processing 
    stt_node_output: Optional[SpeechToTextOutput]         # structured ASR output, has original_query
    
    # Context & Decision 
    # context_LLM_model: str                           # LLM-1 model used for context processing
    context_proc_node_output: Optional[ContextProcessorOutput]    # has normalized query, context tags
    decision_node_output: Optional[DecisionNodeOutput]       # has intent, confidence, intent_reasoning
    
    # Conversation
    # conversation_LLM_model: str                      # LLM-2 model used for conversation 
    conversation_node_output: Optional[ConversationNodeOutput]     # structured conversation response
    clarification_node_output: Optional[ClarificationNodeOutput]   # structured clarification output
    
    # RAG 
    # rag_LLM_model: str                               # LLM-3 model used for RAG
    rag_node_output: Optional[RAGNodeOutput]                 # structured RAG output
    
    # # Action Planning & Execution 
    # action_planner_LLM_model: str                    # LLM-4 model used for action planning
    
    # action input to tools
    action_input_to_tools_llm: Optional[ActionInputToToolsLLM]                      # structured action classification
    informational_response: Optional[str]  # direct response if no action needed
    
    # MCP execution (LangGraph pattern with messages)
    # toolcall_output: Optional[ToolCallOutput]
    # for tool calling with llm_tools_node (it needs a specific format)
    messages: Annotated[List[BaseMessage], add_messages]  # messages for LLM with tools (accumulated)
    llm_tool_call_once: bool  # flag to indicate if LLM tool call has been made in this session
    
    # Memory & History for all chats fro all nodes
    chat_history: Annotated[List[BaseMessage], add_messages]  # conversation history (accumulated)
    
    # Control Flags 
    needs_confirmation: bool # true if action requires human input
    exit: bool  # alternative exit flag

    # final response
    final_response: str  # structured final response output

    # summary,  this is compted once at the end of one loop of conversation just before speaking to human
    summary: str  # conversation summary of previous interactions,
