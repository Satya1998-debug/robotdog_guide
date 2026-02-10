from typing import Literal
from src.graph.state import RobotDogState
from src.graph.schemas import SpeechToTextOutput, TextToSpeechOutput
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import logger

def speech_to_text(audio_data: str) -> str:
    """
    Convert audio to text using ASR model.
    """
    # uses ASR model
    converted_text = input("You (type your query): ")
    return converted_text

def text_to_speech(text: str) -> str:
    """
    Convert text to audio using TTS model.
    """
    # uses TTS model
    #audio_data = "this is audio data generated from text"
    audio_data = text  # Placeholder for actual audio data
    print(f"{text}")
    return audio_data

def listen_to_human(state: RobotDogState) -> RobotDogState:
    """
    Listen to human speech and transcribe with structured output.
    """
    logger.info("[Node] -> listen_to_human_node")
    
    try:
        audio_data = "audio data from human"  # TODO: integrate actual audio input
        converted_text = speech_to_text(audio_data)
        
        if not converted_text or converted_text.strip() == "":
            logger.warning("[listen_to_human] Empty input received")
            converted_text = "hello"  # Default fallback
    except Exception as e:
        logger.error(f"[listen_to_human] Error in speech-to-text: {e}")
        converted_text = "hello"  # Fallback to safe default
    
    # Create structured output
    text_input_from_speech = SpeechToTextOutput(original_query=converted_text)
    
    # Set both the structured output AND the parent state variable
    return {"stt_node_output": dict(text_input_from_speech),
            "original_query": converted_text,
            "chat_history": HumanMessage(content=converted_text)}

def speak_to_human(state: RobotDogState) -> RobotDogState:
    """
    Speak response to human with structured output tracking.
    """
    logger.info("[Node] -> speak_to_human_node")
    response_text = state.get("final_response", "No response to speak.")
    
    try:
        audio_data = text_to_speech(response_text)
        logger.info(f"[speak_to_human] Speaking out: {response_text}")
        audio_generated = True
    except Exception as e:
        logger.error(f"[speak_to_human] Error in text-to-speech: {e}")
        audio_data = None
        audio_generated = False
        # Still print the text even if audio fails
        print(f"[RobotDog Response]: {response_text}")
    
    # Create structured output
    tts_output = TextToSpeechOutput(
        audio_generated=audio_generated,
        audio_metadata={
            "text_length": str(len(response_text)),
            "audio_format": "wav" if audio_generated else "none",
            "duration_estimate": f"{len(response_text) * 0.1}s" if audio_generated else "0s"
        }
    )
    
    return {"final_response": response_text, 
            "chat_history": AIMessage(content=response_text)}