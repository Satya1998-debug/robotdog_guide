import random
from typing import Literal
from src.graph.state import RobotDogState
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import logger
from src.rag_server.voiceAssistant import get_voice_assistant
from src.rag_server.config import NARRATE_NODES

voice_assistant = get_voice_assistant()

# Short, conversational fillers spoken at long-running node entries so the
# user perceives activity. Phrases are intentionally generic (no tech terms).
_NARRATION_PHRASES = {
    "thinking": [
        "Let me think about that.",
        "One moment, please.",
        "Give me a second.",
    ],
    "looking_up": [
        "Let me check that for you.",
        "Looking that up now.",
        "Give me a moment while I check.",
    ],
    "acting": [
        "Alright, working on it.",
        "Okay, on it.",
        "Got it, let me handle that.",
    ],
}


def narrate(key: str) -> None:
    """Speak a short filler phrase for the given node key.

    Safe to call from any node; no-op if narration is disabled or the key
    is unknown. Errors during TTS are swallowed so they never break the graph.
    """
    if not NARRATE_NODES:
        return
    phrases = _NARRATION_PHRASES.get(key)
    if not phrases:
        return
    try:
        voice_assistant.speak(random.choice(phrases))
    except Exception as e:
        logger.warning(f"[narrate] Filler speech failed for '{key}': {e}")

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