"""
Structured output schemas for LangGraph nodes using Pydantic models.
Each node should return a properly typed output that matches these schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from langchain_core.messages import BaseMessage


# this indicates the object/state variabe within each node's output

class SpeechToTextOutput(BaseModel):
    original_query: str = Field(..., description="Transcribed text from audio input")


class TextToSpeechOutput(BaseModel):
    audio_output: bool = Field(default=False, description="Whether audio was successfully generated")


class ContextProcessorOutput(BaseModel):
    """Combined output from context processor that includes BOTH context extraction AND intent classification"""
    context_tags: Dict[str, str] = Field(default_factory=dict, description="Extracted context tags like location, time, person names, etc.")
    # Intent classification fields (computed in same LLM call)
    intent: Literal["conversation", "functional", "institutional", "ambiguous"] = Field(..., description="Detected intent type")
    confidence: Dict[str, float] = Field(..., description="Confidence score for each intent type")
    intent_reasoning: str = Field(..., description="Explanation of intent classification and context extraction")


class DecisionNodeOutput(BaseModel):
    """Output from decision node - this is now derived from ContextProcessorOutput without additional LLM calls"""
    intent: Literal["conversation", "functional", "institutional", "ambiguous"] = Field(..., description="Detected intent type")
    confidence: Dict[str, float] = Field(..., description="Confidence score for each of the intent")
    intent_reasoning: Optional[str] = Field(default=None, description="Explanation of intent classification")


class ConversationNodeOutput(BaseModel):
    conversation_reply: str = Field(..., description="Generated conversational response")


class ClarificationNodeOutput(BaseModel):
    question: str = Field(..., description="Clarification question to ask user")
    clarify_type: str = Field(..., description="Type of clarification needed")


class RAGNodeOutput(BaseModel):
    retrieved_context: str = Field(..., description="Retrieved and summarized context from knowledge base")
    rag_modified_query: str = Field(..., description="Modified query with specific details (full names, room numbers, locations)")
    requires_robot_action: bool = Field(..., description="Whether the query requires physical robot action (navigation, manipulation)")
    action_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence that robot action is needed")
    target_location: Optional[str] = Field(default=None, description="Specific location/destination extracted from context (e.g., 'Room 305', 'Building A')")
    target_person: Optional[str] = Field(default=None, description="Full name of person from context if applicable")
    probable_actions: List[str] = Field(default_factory=list, description="List of probable robot actions based on the query, such as navigation etc.")
    informational_response: str = Field(default="", description="Direct answer to user if no action needed, or context summary if action needed")


class ActionInputToMCP(BaseModel):
    rag_modified_query: str = Field(default="", description="Modified query with specific details (full names, room numbers, locations)")
    action_intent: str = Field(..., description="High-level action intent (navigation, manipulation, etc.)")
    action_type: Literal["navigation", "manipulation", "perception", "other_tools"] = Field(..., description="Classified action type")
    requires_robot_action: bool = Field(default=False, description="Whether the query requires physical robot action (navigation, manipulation)")
    action_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence that robot action is needed")
    target_location: Optional[str] = Field(default=None, description="Specific location/destination extracted from context (e.g., 'Room 305', 'Building A')")
    target_person: Optional[str] = Field(default=None, description="Full name of person from context if applicable")
    probable_actions: List[str] = Field(default_factory=list, description="List of probable robot actions based on the query, such as navigation etc.")


# class ToolCallOutput(BaseModel):
#     messages: List[BaseMessage] = Field(default_factory=list, description="Messages for LLM with tools (accumulated)")
#     toolcall_response: str = Field(..., description="Raw response from tool model")
#     tools_called: List[str] = Field(default_factory=list, description="List of tools that were called during tool execution")    