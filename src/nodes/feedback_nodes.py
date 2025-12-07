from src.graph.state import RobotDogState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from src.config import summarizer_LLM_model, ollama_base_url, ENABLE_SUMMARY
from src.logger import logger

# Initialize summarization LLM
summary_llm = ChatOllama(
                    model=summarizer_LLM_model,  # LLM-6
                    base_url=ollama_base_url,
                    validate_model_on_init=True,
                    temperature=0.4,
                )   


def summarizer_node(state: RobotDogState) -> RobotDogState:
    """
    Summarize the chat history at the start of the conversation with structured output.
    """
    logger.info("[Node] -> summarizer_node")
    
    chat_history = state.get("chat_history", [])
    messages_from_state = state.get("messages", [])
    
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
            
            # Invoke summarization LLM with error handling
            try:
                logger.info("[summarizer_node] Invoking summary LLM...")
                response = summary_llm.invoke(messages)
                summary_content = response.content
                logger.info("[summarizer_node] Summary LLM completed successfully")
            except Exception as e:
                logger.error(f"[summarizer_node] Error invoking summary LLM: {e}")
                # Fallback: Keep existing summary or create basic one
                summary_content = state.get("summary", "Conversation in progress.")
            
            # clear the "messages" field from the state as tool executions for this session are finished 
            delete_ops_msg = [RemoveMessage(id=msg.id) for msg in messages_from_state if messages_from_state]
        
            # delete all chat history 
            delete_ops = [RemoveMessage(id=msg.id) for msg in chat_history]  # keep last 2 messages for context
        
            return {"summary": summary_content, 
                    "chat_history": delete_ops, 
                    "messages": delete_ops_msg}
    
    else:
        # clear the "messages" field from the state as tool executions for this session are finished 
        delete_ops_msg = [RemoveMessage(id=msg.id) for msg in messages_from_state if messages_from_state]
        
        logger.info("[summarizer_node] Summary node is disabled.")
        return {"messages": delete_ops_msg}