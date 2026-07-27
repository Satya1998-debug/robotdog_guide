import os
import time
import logging
import roslibpy

# small pause after navigation completes so the door coordinator can finish
POST_NAV_SPEAK_BUFFER_SEC = float(os.environ.get("POST_NAV_SPEAK_BUFFER_SEC", "1.0"))
# fallback wait when the caller doesn't pass an explicit timeout to get_voice_input().
DEFAULT_LISTEN_TIMEOUT_SEC = float(os.environ.get("VOICE_LISTEN_TIMEOUT_SEC", "10.0"))

# placeholder / no-op strings we don't send to TTS.
_PLACEHOLDER_TEXTS = ("", "NA", "N/A", "NONE", "NULL")


def _is_placeholder(text):
    if text is None:
        return True
    s = str(text).strip()
    return (not s) or s.upper() in _PLACEHOLDER_TEXTS


class RosCommandClient:
    """Rosbridge client for robot-side services.

    Currently exposes:
      - ``start_navigation``  -> ``/agent/start_navigation``
      - ``speak``             -> ``/voice/speak``
      - ``get_voice_input``   -> ``/voice/listen``

    The last two intentionally reuse the same rosbridge websocket connection
    so the guide process doesn't open multiple sockets to the same server.
    They also mirror the ``VoiceAssistant.speak`` / ``VoiceAssistant.get_voice_input``
   
    All robot-side bringup (navigation stack, door pipeline, coordinator,
    voice_assistant_node, rosbridge, robot_command_bridge) is expected to
    already be running on the robot.
    """

    def __init__(self, host=None, port=None, logger=None):
        self.host = host or os.environ.get("ROSBRIDGE_HOST", "0.0.0.0")
        self.port = int(port or os.environ.get("ROSBRIDGE_PORT", 9090))
        self.logger = logger
        self.ros = None
        self.navigation_srv = None
        self.speak_srv = None
        self.listen_srv = None
        self.ros_bridge_connected = self._connect()

    def _connect(self):
        try:
            self.ros = roslibpy.Ros(self.host, self.port)
            self.ros.run()
            self.navigation_srv = roslibpy.Service(self.ros, "/agent/start_navigation", "door_navigation/StartNavigation")
            self.speak_srv = roslibpy.Service(self.ros, "/voice/speak", "door_navigation/Speak")
            self.listen_srv = roslibpy.Service(self.ros, "/voice/listen", "door_navigation/Listen")
            self.logger.info(f"Connected to ROS Bridge at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ROS Bridge: {e}")
            return False

    def _ensure_connection(self):
        # ensure connection to ros bridge, checks and establishes if not already connected
        if self.ros is None or not self.ros.is_connected:
            return self._connect()
        return True

    def start_navigation(self, goal=None, timeout=100):
        """Trigger navigation on Jetson via service. Blocks until it completes."""
        try:
            if goal is None:
                goal = {}
            self.logger.info(f"Received navigation goal: {goal}")

            if not goal:
                return {"success": False, "reason": "goal_for_navigation_cannot_be_empty"}

            if not self._ensure_connection():
                return {"success": False, "reason": "failed_to_connect_to_ros_bridge"}

            request = roslibpy.ServiceRequest(goal)
            response = self.navigation_srv.call(request, timeout=timeout)

            if not response.get("success", False):
                self.logger.error(f"Navigation failed: {response.get('reason', 'unknown')}")

            self.logger.info(f"Navigation response: {response}")

            if POST_NAV_SPEAK_BUFFER_SEC > 0:
                time.sleep(POST_NAV_SPEAK_BUFFER_SEC)

            return response

        except Exception as e:
            self.logger.error(f"Exception occurred in start_navigation: {e}")
            return {"success": False, "reason": str(e)}

    @property
    def recognizer(self):
        # return the audio recognizer service from the ros bridge
        return self.listen_srv

    def speak(self, text="", blocking=True):
        """Route TTS through /voice/speak on the robot.

        Empty and placeholder strings ("", "NA", "N/A", ...) are silently
        dropped so the TTS engine never mispronounces field-defaults. Set
        ``blocking=False`` for fire-and-forget narration: the voice node
        queues the utterance and returns immediately.
        """
        if _is_placeholder(text):
            return
        stripped = str(text).strip()

        if not self._ensure_connection() or self.speak_srv is None:
            self.logger.warning(f"[SPEAK-fallback] {stripped}")
            return
        try:
            req = roslibpy.ServiceRequest({"text": stripped, "blocking": bool(blocking)})
            self.speak_srv.call(req, timeout=180)
        except Exception as e:
            self.logger.error(f"speak failed: {e}; fallback: [SPEAK] {stripped}")

    def get_voice_input(self, timeout_sec=None):
        """Return the next transcribed utterance (lowercased) or "".

        Kept lenient: returns "" (rather than raising) when the service is
        unreachable so the guide falls back to keyboard input via
        ``speech_to_text``.
        """
        if not self._ensure_connection() or self.listen_srv is None:
            self.logger.warning("[VoiceAssistant] /voice/listen unavailable; returning empty string.")
            return ""

        req_timeout = float(timeout_sec) if timeout_sec else 0.0
        # Give the transport a bit more time than the service-side wait so
        # the service replies "timed_out" instead of us tripping the RPC timeout.
        rpc_timeout = (req_timeout if req_timeout > 0 else DEFAULT_LISTEN_TIMEOUT_SEC) + 5.0
        try:
            req = roslibpy.ServiceRequest({"timeout_sec": req_timeout, "grammar": ""})
            resp = self.listen_srv.call(req, timeout=rpc_timeout)
            if resp is None or resp.get("timed_out"):
                return ""
            return (resp.get("text") or "").lower()
        except Exception as e:
            self.logger.error(f"listen failed: {e}")
            return ""

    def get_speech_input(self):
        """Legacy stub kept for API compatibility with the old VoiceAssistant."""
        return "Hi"

    def get_text_input(self):
        """Blocking keyboard fallback, unchanged."""
        text = input("Type your query: ")
        return text.strip().lower()

    def close(self):
        # close the ros bridge connection
        if self.ros is not None:
            self.ros.terminate()


if __name__ == "__main__":
    ros_client = RosCommandClient(logger=logging.getLogger(__name__))
    try:
        s_time = time.time()
        ros_client.logger.info("Starting navigation...")
        result = ros_client.start_navigation(
            goal={"person": "satya", "room": "pos3"}, timeout=500
        )
        ros_client.logger.info(f"Navigation Result: {result}")
        ros_client.logger.info(f"Total time taken: {time.time() - s_time:.2f} seconds")
    finally:
        ros_client.close()
