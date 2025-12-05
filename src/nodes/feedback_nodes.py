from src.graph.state import RobotDogState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from src.config import summarizer_LLM_model, ollama_base_url
from src.logger import logger

def summarizer_node(state: RobotDogState) -> RobotDogState:
    """
    Summarize the chat history at the start of the conversation with structured output.
    """
    logger.info("[Node] -> summarizer_node")
    if state.get("chat_history", ""):
        messages = []

        if state.get("summary", ""):
            prompt = """ This is an ongoing conversation with prior summary provided. 
            Extend the summary with new messages given now."""

        else:
            prompt = """Create a summary of the following conversation between the user and RobotDog assistant in a concise manner, focusing on key points discussed."""

        messages.extend([
            *state.chat_history,
            HumanMessage(content=prompt),
        ])

                
        llm = ChatOllama(
                model=summarizer_LLM_model,  # LLM-6
                base_url=ollama_base_url,
                validate_model_on_init=True,
                temperature=0.4,
            )   
            
        response = llm.invoke(messages)

        delete_ops = [RemoveMessage(id=msg.id) for msg in state.chat_history]  # keep last 2 messages for context
    
    return {"summary": response.content, "chat_history": delete_ops}