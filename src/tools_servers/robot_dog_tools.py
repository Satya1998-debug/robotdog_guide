"""
MCP Tools for RobotDog Indoor Navigation
Dummy methods that will be exposed as MCP tools for controlling and interacting with RobotDog
"""

import json
import logging
import time
from typing import Dict
from langchain.tools import tool
from src.logger import logger
from src.tools_servers.ros_client import RosCommandClient

ros_client = RosCommandClient()

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

@tool
def navigate(person: str, location: str) -> Dict:
    """Navigate to a specific coordinate (x, y) (Action). 

    Note: This tool is called if there is direct command from user to navigate or modified query from RAG system indicates navigation action or any other context requiring navigation.
    This tool will need further confirmation from the user before executing the navigation action.

    Args:
        person (str): Name of the person to navigate to.
        location (str): Location to navigate to.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the navigation result,
              'message': A message indicating the result of the action,
    """    
    logger.info(f"Navigate to ({person}, {location}) requested.")

    # the person coordinates is mapped on Jetson side
    goal = {"person": person, "room": location}

    try:
        result = ros_client.start_navigation(goal, timeout=900) # empty result dict is returned if using movebase action
        return {"status": "success", "reason": result["reason"], "message": f"Robot arrived at {location}."}
    except RuntimeError as e:
        return {"status": "failure", "reason": str(e)}

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

def emergency_stop() -> Dict:
    """Emergency stop - halt all movement (Action). 
    
    Note: This tool is called if there is direct command from user to stop or modified query from RAG system indicates stop action.
    This serves as a safety measure to immediately halt all robot dog movements if any unsafe condition (collision, abuse, sudden vibration) is detected.
    
    Returns:
        Dict: 'Status': 'success' or 'failure' indicating the robot dog's state, 'message': a message indicating the result of the action."""
    logger.info("Emergency stop executed.")
    return {"status": "success", "message": "Emergency stop activated"}

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

tools = [
    navigate,
]