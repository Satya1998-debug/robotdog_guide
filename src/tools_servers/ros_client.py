import os
import time
import logging

import roslibpy


# Small pause after navigation completes so the door coordinator can finish
POST_NAV_SPEAK_BUFFER_SEC = float(os.environ.get("POST_NAV_SPEAK_BUFFER_SEC", "1.0"))


class RosCommandClient:
    """Thin rosbridge client that only invokes /agent/start_navigation.

    All robot side bringup (navigation stack, door pipeline, coordinator,
    state estimator, rosbridge, robot_command_bridge) is expected to already
    be running on the robot. This client only sends navigation requests.
    """

    def __init__(self, host=None, port=None, logger=None):
        self.host = host or os.environ.get("ROSBRIDGE_HOST", "0.0.0.0")
        self.port = int(port or os.environ.get("ROSBRIDGE_PORT", 9090))
        self.logger = logger
        self.ros = None
        self.navigation_srv = None
        self.ros_bridge_connected = self._connect()

    def _connect(self):
        try:
            self.ros = roslibpy.Ros(self.host, self.port)
            self.ros.run()
            self.navigation_srv = roslibpy.Service(self.ros, "/agent/start_navigation", "door_navigation/StartNavigation",)
            self.logger.info(f"Connected to ROS Bridge at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ROS Bridge: {e}")
            return False

    def _ensure_connection(self):
        if self.ros is None or not self.ros.is_connected:
            return self._connect()
        return True

    def start_navigation(self, goal=None, timeout=900):
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

    def close(self):
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
