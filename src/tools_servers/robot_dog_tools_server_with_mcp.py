"""
MCP Tools for RobotDog Indoor Navigation
Dummy methods that will be exposed as MCP tools for controlling and interacting with RobotDog
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("robot_dog_tools_server")

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_dog_tools_server")

"""Collection of tools for RobotDog indoor navigation via MCP"""

@mcp.tool()
def stand_up() -> Dict:
    """Method to make the robot dog stand up (Action). 
    
    Note: This tool is called if there is direct command from user to stand up or modified query from RAG system indicates stand up action.
    This also serves as a safety check to ensure the robot dog is in a standing position before executing other actions.

    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the robot dog's state, 
        'message': A message indicating the result of the action.
    """
    logger.info("Stand up executed.")
    
    return {"status": "success", "message": "Robot dog is now standing"}

@mcp.tool()
def sit_down() -> Dict:
    """Method to make the robot dog sit down (Action). 

    Note: This tool is called if there is direct command from user to sit down or modified query from RAG system indicates sit down action.
    This also serve as a safety check if the battery is low or if any collision is detected.

    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the robot dog's state, 
        'message': A message indicating the result of the action.
    """
    logger.info("Sit down executed.")
    return {"status": "success", "message": "Robot dog is now sitting"}

def detect_door() -> Dict:
    """Detect doors in the current field of view (Action). 
    This tool is called if there is direct command from user to detect doors or modified query from RAG system indicates door detection.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the detection result,
              'message': A message indicating the result of the action,
              'doors_detected' (int): Number of doors detected 
    """
    logger.info("Detect door executed.")
    return {
            "status": "success",
            "doors_detected": 1,
            "message": "Found 1 door"
    }

@mcp.tool()
def navigate_to(x: float, y: float) -> Dict:
    """Navigate to a specific coordinate (x, y) (Action). 

    Note: This tool is called if there is direct command from user to navigate or modified query from RAG system indicates navigation action or any other context requiring navigation.
    This tool will need further confirmation from the user before executing the navigation action.

    Args:
        x (float): X coordinate to navigate to, in the map.
        y (float): Y coordinate to navigate to, in the map.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the navigation result,
              'message': A message indicating the result of the action,
              'current_position': The robot dog's current position as a dict with 'x', 'y' keys.
    """
    current_position = {"x": 2.0, "y": 3.0, "z": 0.0}
    logger.info(f"Navigate to ({x}, {y}) executed.")
    
    detect_door_result = detect_door()
    if detect_door_result.get("doors_detected", 0) == 0:
        return {"status": "failure", "message": "No doors detected, cannot navigate"}

    detect_obstacles_result = detect_obstacles()
    if detect_obstacles_result.get("obstacles_detected", 0) > 0:
        return {"status": "failure", "message": "Obstacles detected, cannot navigate"}
        
    return {
            "status": "success",
            "message": f"Moving to {current_position['x']}, {current_position['y']})",
            "current_position": current_position,
        }

# helper functions for navigation tool
def detect_obstacles() -> Dict:
    """Detect obstacles in the surrounding area. (Action). 
    This tool is called if there is direct command from user to detect obstacles or modified query from RAG system indicates obstacle detection.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the detection result,
              'obstacles_detected' (int): Number of obstacles detected,
              'message': A message indicating the result of the action."""
    logger.info("Detect obstacles executed.")
    return {
            "status": "success",
            "obstacles_detected": 0,
            "message": "No obstacles found"
        }

@mcp.tool()
def emergency_stop() -> Dict:
    """Emergency stop - halt all movement (Action). 
    
    Note: This tool is called if there is direct command from user to stop or modified query from RAG system indicates stop action.
    This serves as a safety measure to immediately halt all robot dog movements if any unsafe condition (collision, abuse, sudden vibration) is detected.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the robot dog's state, 'message': a message indicating the result of the action."""
    logger.info("Emergency stop executed.")
    return {"status": "success", "message": "Emergency stop activated"}

@mcp.tool()
def get_sensor_data() -> Dict:
    """Get current sensor readings such as temperature, battery level, etc. (Action). 

    Note: This tool is called if there is direct command from user to get sensor data or modified query from RAG system indicates sensor data retrieval.
    This serves to provide real-time status of the robot dog to ensure it is operating within safe parameters.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the retrieval state, 
              'temperature' (float): Current temperature reading,
              'battery' (int): Current battery level percentage,
              'message': A message indicating the result of the action.
    """
    logger.info("Get sensor data executed.")
    return {
            "status": "success",
            "temperature": 25.0,
            "battery": 85,
            "message": "Sensor data retrieved"
        }

# Example usage
if __name__ == "__main__":
    mcp.run(transport="stdio")