"""Rosbridge-backed voice assistant (thin wrapper).

Voice I/O now lives inside ``door_navigation/scripts/voice_assistant_node.py``
on the robot. This module keeps the same public entry point --
``get_voice_assistant()`` returning something that exposes ``.speak(...)``
and ``.get_voice_input(...)`` -- but the implementation is just an alias for
:class:`src.tools_servers.ros_client.RosCommandClient`, which already talks
to the robot over rosbridge for navigation and now also exposes the voice
methods.

Why: two processes can't own the same ALSA capture device. Before, both the
robotdog guide and the door coordinator opened a local ``VoiceAssistant``
and one of them failed with a channel error. Now every consumer routes
through the single voice node on the robot.
"""

from src.tools_servers.ros_client import RosCommandClient
from src.logger import logger


# Backwards-compat alias so ``from src.rag_server.voiceAssistant import VoiceAssistant``
# keeps working. The class *is* the rosbridge client.
VoiceAssistant = RosCommandClient


_voice_assistant_instance = None


def get_voice_assistant(enable_listening=True):
    """Return a process-wide shared voice assistant.

    The ``enable_listening`` argument is accepted for API compatibility with
    the old local implementation but is a no-op here: whether ASR works
    depends on the ``/voice/listen`` service being up on the robot, not on
    anything we do in this process.

    The instance is created lazily on the first call so importing this
    module does not open a rosbridge connection.
    """
    global _voice_assistant_instance
    if _voice_assistant_instance is None:
        _voice_assistant_instance = RosCommandClient(logger=logger)
    return _voice_assistant_instance


if __name__ == "__main__":
    va = get_voice_assistant()
    try:
        va.speak("Hello from the robotdog guide over rosbridge.")
        if va.recognizer is None:
            print("Listening service unavailable.")
        else:
            while True:
                text = va.get_voice_input(timeout_sec=10)
                if not text:
                    print("(no speech within timeout)")
                    continue
                print(f"You said: {text}")
                if "exit" in text or "quit" in text:
                    break
                va.speak(f"You said: {text}")
    finally:
        va.close()
