from src.graph.state import RobotDogState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from src.config import summarizer_LLM_model, ollama_base_url, ENABLE_SUMMARY
from src.logger import logger

def summarizer_node(state: RobotDogState) -> RobotDogState:
    """
    Summarize the chat history at the start of the conversation with structured output.
    """
    logger.info("[Node] -> summarizer_node")
    
    chat_history = state.get("chat_history", [])
    
    if ENABLE_SUMMARY or len(chat_history) > 20:  # enable summary node if configured or chat history is long
        logger.info("[summarizer_node] Summary node is enabled.")
        if chat_history:
            messages = []
            if state.get("summary", ""):
                prompt = """ This is an ongoing conversation with prior summary provided. 
                Extend the summary with new messages given now."""
                summary_system_msg = f"Previous conversation summary: {state['summary']}"
                messages.append(SystemMessage(content=summary_system_msg))

            else:
                prompt = """Create a summary of the historical conversation between the user and RobotDog assistant in a concise manner, 
                focusing on key points discussed. Also keep information that may be relevant for future context, keep info of the node execution sequence in the history.
                Donot use any speciial formatting, just plain text."""

            messages.extend(chat_history)  # include all messages for context
            messages.append(HumanMessage(content=prompt))

            llm = ChatOllama(
                    model=summarizer_LLM_model,  # LLM-6
                    base_url=ollama_base_url,
                    validate_model_on_init=True,
                    temperature=0.4,
                )   
                
            response = llm.invoke(messages)

            delete_ops = [RemoveMessage(id=msg.id) for msg in chat_history]  # keep last 2 messages for context
        
            return {"summary": response.content, "chat_history": delete_ops}
    
    else:
        logger.info("[summarizer_node] Summary node is disabled.")
        return {}