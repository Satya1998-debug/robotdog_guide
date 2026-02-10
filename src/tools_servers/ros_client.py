#import roslibpy
import time

class RosCommandClient:
    def __init__(self, host="localhost", port=9090):
        self.host = host
        self.port = port
        self._connect()

    def _connect(self):
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

    def _ensure_connection(self):
        if not self.ros.is_connected:
            self._connect()

    # -------------------------------------------------
    # Navigation (Service)
    # -------------------------------------------------

    def start_navigation(self, target="door", timeout=900):
        """
        Triggers navigation on Jetson via Service.
        Blocks until navigation completes.
        """

        self._ensure_connection()

        # OPTIONAL: start door coordinator first (if that is your policy)
        self.start_door_coordinator()

        request = roslibpy.ServiceRequest({
            "target": target
        })

        try:
            response = self.navigation_srv.call(request, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Navigation service call failed: {e}")

        # Response is explicit
        if not response.get("success", False):
            raise RuntimeError(
                f"Navigation failed: {response.get('reason', 'unknown')}"
            )

        return response  # {"success": True, "reason": "arrived"}

    # -------------------------------------------------
    # Door coordinator (Service)
    # -------------------------------------------------

    def start_door_coordinator(self):
        self._ensure_connection()

        try:
            request = roslibpy.ServiceRequest()
            return self.door_coordinator_srv.call(request)
        except Exception as e:
            raise RuntimeError(f"Door coordinator failed: {e}")

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
        result = ros_client.start_navigation(target="door", timeout=120)
        print("Navigation Result:", result)
        e_time = time.time()
        print(f"Total time taken: {e_time - s_time:.2f} seconds")
    except RuntimeError as e:
        print("Navigation Error:", e)
    finally:
        ros_client.close()
