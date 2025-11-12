from pydantic import Field, BaseModel

# state of the graph 
class RobotDogState(BaseModel):
    orginial_query: str = Field(..., description="The command given to the robot dog.")
    feedback: str = Field(None, description="Feedback from the robot dog's sensors.")
    final_response: str = Field(None, description="Final response summarizing the robot dog's status.")
    node_sequence: list[str] = Field(default_factory=list, description="Sequence of nodes executed in the workflow.")
    action_sequence: list[str] = Field(default_factory=list, description="Sequence of actions taken by the robot dog.")
    chat_history: list[str] = Field(default_factory=list, description="Chat history for context.")