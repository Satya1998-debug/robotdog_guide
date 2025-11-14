"""
Structured output schemas for LangGraph nodes using Pydantic models.
Each node should return a properly typed output that matches these schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# this indicates the object/state variabe within each node's output

class SpeechToTextOutput(BaseModel):
    original_query: str = Field(..., description="Transcribed text from audio input")


class TextToSpeechOutput(BaseModel):
    audio_output: bool = Field(default=False, description="Whether audio was successfully generated")


class ContextProcessorOutput(BaseModel):
    normalized_query: str = Field(..., description="Normalized or clean query by the context processor")
    original_query: str = Field(..., description="Raw user input")
    context_tags: Dict[str, str] = Field(default_factory=dict, description="Extracted context tags like location, time, etc.")


class DecisionNodeOutput(BaseModel):
    intent: Literal["conversation", "functional", "institutional", "ambiguous"] = Field(..., description="Detected intent type")
    confidence: Dict[str, float] = Field(..., description="Confidence score for each of the intent")
    intent_reasoning: Optional[str] = Field(default=None, description="Explanation of intent classification")


class ConversationNodeOutput(BaseModel):
    response_conversation: str = Field(..., description="Generated conversational response")

class ClarificationNodeOutput(BaseModel):
    clarification_question: str = Field(..., description="Clarification question to ask user")
    clarification_type: str = Field(..., description="Type of clarification needed")
    original_query: str = Field(..., description="The query that needs clarification")


class RAGNodeOutput(BaseModel):
    retrieved_docs: List[str] = Field(..., description="Retrieved document snippets from vector DB")
    rag_result: str = Field(..., description="Generated response using RAG")
    sources: Optional[List[str]] = Field(default=None, description="Source identifiers for retrieved docs")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in RAG result")

#-------------------------------------------------------------------------------------- schema edited till here -------
class ActionClassifierOutput(BaseModel):
    action_intent: str = Field(..., description="High-level action intent (navigation, manipulation, etc.)")
    action_type: Literal["navigation", "manipulation", "perception", "other_tools"] = Field(..., description="Classified action type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")


class ActionPlannerOutput(BaseModel):
    action_intent: str = Field(..., description="High-level action description")
    plan: Dict[str, str] = Field(..., description="Structured action plan")
    action_sequence: List[str] = Field(..., description="Ordered list of robot actions to execute")
    estimated_duration: Optional[float] = Field(default=None, description="Estimated execution time in seconds")
    requires_confirmation: bool = Field(default=False, description="Whether action needs user confirmation")


class MCPToolOutput(BaseModel):
    tool_called: str = Field(..., description="Name of the MCP tool invoked")
    tool_result: Dict[str, str] = Field(..., description="Result from MCP tool execution")
    robot_status: str = Field(..., description="Current robot status after tool execution")
    execution_success: bool = Field(..., description="Whether tool execution succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")


class PerceptionFeedbackOutput(BaseModel):
    feedback: str = Field(..., description="Processed sensor/perception feedback")
    sensor_data: Optional[Dict[str, str]] = Field(default=None, description="Raw or processed sensor readings")
    anomalies_detected: bool = Field(default=False, description="Whether any anomalies were detected")


class SummarizerNodeOutput(BaseModel):
    final_response: str = Field(..., description="Summarized status for TTS output")
    summary: str = Field(..., description="Detailed summary of action execution and feedback")
    success: bool = Field(..., description="Whether overall action was successful")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Additional summary metadata")


class MemoryUpdate(BaseModel):
    key: str = Field(..., description="Memory key")
    value: str = Field(..., description="Memory value")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of memory update")


class NodeSequenceEntry(BaseModel):
    node_name: str = Field(..., description="Name of the executed node")
    timestamp: Optional[str] = Field(default=None, description="Execution timestamp")
    status: Literal["success", "error", "skipped"] = Field(default="success", description="Execution status")
