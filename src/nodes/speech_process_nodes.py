from typing import Literal
from src.graph.state import RobotDogState
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import logger
from src.rag_server.voiceAssistant import get_voice_assistant

voice_assistant = get_voice_assistant()

def speech_to_text(enable_audio=True) -> str:
    """
    Convert audio to text using ASR model.
    """
    
    converted_text = voice_assistant.get_voice_input(timeout_sec=10) if enable_audio else ""
    if not converted_text or not enable_audio:
        logger.warning("[speech_to_text] Empty input received from ASR. or Audio disabled, returning empty string.")        
        converted_text = input("You (type your query): ")
        
    logger.info(f"[speech_to_text] Converted text: {converted_text}")
    
    return converted_text

def text_to_speech(text: str) -> str:
    """
    Convert text to audio using TTS model.
    """
    # uses TTS model
    #audio_data = "this is audio data generated from text"
    logger.info(f"[text_to_speech] Speaking: {text}")
    voice_assistant.speak(text)

def listen_to_human(state: RobotDogState) -> RobotDogState:
    """
    Listen to human speech and transcribe with structured output.
    """
    logger.info("[Node] -> listen_to_human_node")
    
    try:
        converted_text = speech_to_text()
        logger.info(f"[listen_to_human] Converted text: {converted_text}")
        
    except Exception as e:
        logger.error(f"[listen_to_human] Error in speech-to-text: {e}")
        converted_text = ""
        
    # Set both the structured output AND the parent state variable
    return {"original_query": converted_text,
            "chat_history": HumanMessage(content=converted_text)}

def speak_to_human(state: RobotDogState) -> RobotDogState:
    """
    Speak response to human with structured output tracking.
    """
    logger.info("[Node] -> speak_to_human_node")
    response_text = state.get("final_response", "No response to speak.")
    
    try:
        text_to_speech(response_text)
        logger.info(f"[speak_to_human] Speaking out: {response_text}")
    except Exception as e:
        logger.error(f"[speak_to_human] Error in text-to-speech: {e}")
        response_text = ""
        # Still print the text even if audio fails
    
    return {"final_response": response_text, 
            "chat_history": AIMessage(content=response_text)}