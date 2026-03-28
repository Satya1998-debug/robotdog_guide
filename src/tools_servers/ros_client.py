import roslibpy
import time

class RosCommandClient:
    def __init__(self, host="0.0.0.0", port=9090, logger=None):
        self.host = host
        self.port = port
        self.logger = logger
        self.ros_bridge_connected = False
        self.ros_bridge_connected = self._connect()

    def _connect(self):
        try:
            # Initialize rosbridge connection
            self.ros = roslibpy.Ros(self.host, self.port)
            self.ros.run()

            # Door coordinator service
            self.door_coordinator_srv = roslibpy.Service(
                self.ros,
                "/agent/start_door_coordinator",
                "std_srvs/Trigger"
            )

            # Navigation service (CUSTOM)
            self.navigation_srv = roslibpy.Service(
                self.ros,
                "/agent/start_navigation",
                "door_navigation/StartNavigation"
            )
            if self.logger:
                self.logger.info(f"Connected to ROS Bridge at {self.host}:{self.port}")
            else:
                print(f"Connected to ROS Bridge at {self.host}:{self.port}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to connect to ROS Bridge: {e}")
            else:
                print(f"Failed to connect to ROS Bridge: {e}")
            return False

    def _ensure_connection(self):
        if not self.ros.is_connected: # if not connected, try to reconnect
            return self._connect()
        return True # if already connected, return True

    # -------------------------------------------------
    # Navigation (Service)
    # -------------------------------------------------

    def start_navigation(self, goal=None, timeout=900):
        """
        Triggers navigation on Jetson via Service.
        Blocks until navigation completes.
        """
        try:
            if goal is None:
                goal = {}
            print(f"Received navigation goal: {goal}")
            
            if not goal:
                return {"status": "Failed", "reason": "Goal for navigation cannot be empty.", "message": "Navigation failed."}

            if not self._ensure_connection(): # need to ensure the ROS Bridge is connected
                return {"status": "Failed", "reason": "Failed to connect to ROS Bridge.", "message": "Navigation failed."}

            print("ROS Bridge connection established. Starting door coordinator...")
            # need to check if door coordinator is already running else need to start
            if self.start_door_coordinator():
                if self.logger:
                    self.logger.info("Door coordinator is running. Proceeding with navigation.")
                request = roslibpy.ServiceRequest(goal) # {"person": "Alice", "room": "3.012"} 
                response = self.navigation_srv.call(request, timeout=timeout)

                # Response is explicit
                if not response.get("success", False):
                    if self.logger:
                        self.logger.error(f"Navigation failed: {response.get('reason', 'unknown')}")
                    else:
                        print(f"Navigation failed: {response.get('reason', 'unknown')}")

                if self.logger:
                    self.logger.info(f"Navigation success: {response}")
                return response  # {"success": True, "reason": "arrived"}

            else:
                return {"status": "Failed", "reason": "Door coordinator is not running.", "message": "Navigation failed."}
        
        except Exception as e:
                if self.logger:
                    self.logger.error(f"Exception occurred in start_navigation: {e}")
                else:
                    print(f"Exception occurred in start_navigation: {e}")
                return {"status": "Failed", "reason": str(e), "message": "Navigation failed."}


      
    # -------------------------------------------------
    # Door coordinator (Service)
    # -------------------------------------------------

    def start_door_coordinator(self):
        if not self._ensure_connection(): # ensure ROS Bridge is connected before calling the service
            return False

        try:
            request = roslibpy.ServiceRequest()
            return self.door_coordinator_srv.call(request)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Door coordinator failed: {e}")
            else:
                print(f"Door coordinator failed: {e}")
            return False

    def close(self):
        self.ros.terminate()


# -------------------------------------------------
# Standalone test
# -------------------------------------------------

if __name__ == "__main__":
    ros_client = RosCommandClient()
    try:
        s_time = time.time()
        print("Starting door coordinator...")
        result = ros_client.start_navigation(goal={"person": "Alice", "room": "3.012"}, timeout=120)
        print("Navigation Result:", result)
        e_time = time.time()
        print(f"Total time taken: {e_time - s_time:.2f} seconds")
    except RuntimeError as e:
        print("Navigation Error:", e)
    finally:
        ros_client.close()
