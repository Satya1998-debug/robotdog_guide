from typing import TypedDict, List, Dict, Optional

class RobotDogState(TypedDict, total=False):
    input_text: str                     # From ASR or text input
    final_response: str                 # Final output to TTS

    # for Context Processor
    original_query: str                 # Raw user input before normalization
    query: str                          # Normalized/cleaned form
    context_tags: List[str]             # Tags like ["location:lab"]

    # for Decision Node
    intent: str                         # "conversation" | "functional" | "institutional" | ...
    confidence: float                   # Confidence score for safety logic
    ambiguous: bool                    # Whether query needs clarification

    # for RAG node
    retrieved_docs: List[str]           # Retrieved docs from vector DB (summaries)
    rag_result: str                     # RAG output text

    # for Action Planner
    action_intent: str                  # High-level action (move, stand, open door, etc..)
    plan: Dict[str, str]                # Structured plan: {"action": "...", "target": "..."}
    action_sequence: List[str]          # Sequence of planned robot actions

    # for MCP Tool Execution
    tool_called: str                    # Name of tool invoked
    tool_result: Dict[str, str]         # Response from MCP tool

    # for Robot Execution
    robot_status: str                   # "running" | "idle" | "error"
    sensor_feedback: str                # Raw robot feedback

    # for Memory for Feedback
    feedback: str                       # Processed feedback text
    summary: str                        # Summarized status for TTS
    memory_state: Dict[str, str]        # Persistent short-term memory

    # for Conversation Management
    chat_history: List[str]             # conversation history
    node_sequence: List[str]            # Sequence of visited nodes for context tracking in history

    # for Control Flags
    needs_confirmation: bool            # True if action requires human input
    exit_flag: bool                    # True if user said bye/exit
