import roslibpy
import threading
import time

class RosCommandClient:
    def __init__(self, host="localhost", port=9090):
        self.host = host
        self.port = port
        self._lock = threading.Lock() # to ensure thread safety
        self._connect()
        
        # navigation result state
        self._nav_result = None
        self._nav_error = None

    def _connect(self):
        # initialize ros connection
        self.ros = roslibpy.Ros(self.host, self.port)
        self.ros.run()

        # initiate door coordinator service
        self.door_coordinator_srv = roslibpy.Service(self.ros, "/agent/start_door_coordinator", "std_srvs/Trigger")  # roslibpy instance, service, type

        # initiate navigation action client
        self.nav_action = roslibpy.ActionClient(self.ros, "/agent/start_navigation", "door_navigation/NavigateTaskAction")  # roslibpy instance, action, type

    def _ensure_connection(self):
        # ensure ros connection is active
        if not self.ros.is_connected:
            self._connect()

    def start_navigation(self, timeout=900): # 15 minutes timeout
        """
        Triggers navigation Action on Jetson.
        This Action internally triggers UDP-based navigation.
        Blocks until navigation completes or fails.
        """

        with self._lock:
            self._ensure_connection()

            self._nav_result = None
            self._nav_error = None

            goal = {}  # empty goal = "navigate to door"

            goal_id = self.nav_action.send_goal(
                goal=goal,
                resultback=self._on_nav_result,
                feedback=self._on_nav_feedback, 
                errback=self._on_nav_error
            )

            if goal_id is None:
                raise RuntimeError("Failed to send navigation goal")

        try:
            self.nav_action.wait_goal(goal_id, timeout) # blocks until goal completes or timeout
        except Exception:
            self.nav_action.cancel_goal(goal_id)
            raise RuntimeError("Navigation timed out")

        if self._nav_error is not None:
            raise RuntimeError(f"Navigation failed: {self._nav_error}")
        
        # explicit failure from result
        if not self._nav_result.get("success", False):
            raise RuntimeError(
                f"Navigation failed: {self._nav_result.get('reason', 'unknown')}"
            )

        return self._nav_result # {"success": True, "reason": "arrived"}
    
    def _on_nav_feedback(self, feedback):
        # progress, pose, etc.
        pass

    def _on_nav_result(self, result):
        self._nav_result = result

    def _on_nav_error(self, error):
        self._nav_error = error


    def start_door_coordinator(self):
        with self._lock:
            self._ensure_connection()
            req = roslibpy.ServiceRequest()
            return self.door_coordinator_srv.call(req)

    def close(self):
        self.ros.terminate()
