
from src.graph.state import RobotDogState

def speech_to_text(audio_data: str) -> str:
    # uses ASR model
    converted_text = "This is a transcribed text from audio."
    return converted_text

def text_to_speech(text: str) -> str:
    # uses TTS model
    audio_data = "this is audio data generated from text"
    return audio_data

def listen_to_human(state: RobotDogState) -> RobotDogState:
    audio_data = "audio data from human"
    converted_text = speech_to_text(audio_data)
    return {"original_query": converted_text}

def speak_to_human(state: RobotDogState) -> RobotDogState:
    response_text = state.get("final_response", "No response to speak.")
    audio_data = text_to_speech(response_text)
    print(f"[SpeakToHuman] Speaking out: {response_text} (stub for audio data: {audio_data})")
    return {}