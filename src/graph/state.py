from typing import Literal, TypedDict, List, Dict, Optional, Annotated
from operator import add
from src.graph.schemas import (
    SpeechToTextOutput,
    ContextProcessorOutput,
    DecisionNodeOutput,
    ConversationNodeOutput,
    ClarificationNodeOutput,
    RAGNodeOutput,
    ActionClassifierOutput,
    ActionPlannerOutput,
    MCPToolOutput,
    PerceptionFeedbackOutput,
    SummarizerNodeOutput,
    MemoryUpdate,
    NodeSequenceEntry
)


class RobotDogState(TypedDict):
    """
    LangGraph state with structured outputs for each node, each field is either a single structured output or a list that accumulates outputs.
    """

    start_conversation: bool  # true if starting new conversation, of pursuit existing
    
    # Speech Processing 
    stt_node_output: Optional[SpeechToTextOutput]         # structured ASR output, has original_query
    
    # Context & Decision 
    context_LLM_model: str                           # LLM-1 model used for context processing
    context_proc_node_output: Optional[ContextProcessorOutput]    # has normalized query, context tags
    decision_node_output: Optional[DecisionNodeOutput]       # has intent, confidence, intent_reasoning
    
    nodewise_chat_history: List[str]          # intermediate chat history per node, need to add redce logic

    # Conversation
    conversation_LLM_model: str                      # LLM-2 model used for conversation 
    conversation_node_output: Optional[ConversationNodeOutput]     # structured conversation response
    clarification_node_output: Optional[ClarificationNodeOutput]   # structured clarification output
    
    # RAG 
    rag_node_output: Optional[RAGNodeOutput]                 # structured RAG output

    #-------------------------------------------------- edited till here --------------------------------
    
    # Action Planning & Execution 
    action_classifier_output: Optional[ActionClassifierOutput]  # structured action classification
    action_planner_output: Optional[ActionPlannerOutput]        # structured action plan
    mcp_output: Optional[MCPToolOutput]                         # structured MCP tool result
    
    action_intent: str                                  # high-level action intent
    plan: Dict[str, str]                               # structured plan
    action_sequence: List[str]                         # sequence of planned robot actions
    tool_called: str                                    # name of tool invoked
    tool_result: Dict[str, str]                        # response from MCP tool
    robot_status: str                                   # "running" | "idle" | "error"
    
    # Feedback & Perception 
    perception_output: Optional[PerceptionFeedbackOutput]       # structured perception feedback
    summarizer_output: Optional[SummarizerNodeOutput]           # structured summary
    
    feedback: str                                       # processed feedback text
    summary: str                                        # summarized status for TTS
    sensor_feedback: str                                # raw robot feedback
    
    # Memory & History 
    memory_state: Dict[str, str]                       # persistent short-term memory
    memory_updates: Annotated[List[MemoryUpdate], add] # accumulated memory updates
    chat_history: Annotated[List[str], add]            # conversation history (accumulated)
    node_sequence: Annotated[List[NodeSequenceEntry], add]  # sequence of visited nodes (accumulated)
    
    # Control Flags 
    needs_confirmation: bool                            # true if action requires human input
    exit_flag: bool                                    # true if user said bye/exit
    exit: bool                                          # alternative exit flag