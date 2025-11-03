# MCP-based RobotDog Indoor Navigation

This repository contains code and resources for a RobotDog model based on the Model Context Protocol (MCP) for indoor navigation tasks.

## Overview
- **MCP (Model Context Protocol):** A protocol for integrating and managing robot models in a modular and extensible way.
- **Purpose:** To provide a framework for developing, simulating, and controlling a RobotDog using MCP principles for indoor navigation.

## Features
- Comprehensive RobotDog control tools exposed as MCP methods
- Indoor navigation capabilities (door detection, obstacle avoidance, room scanning)
- Modular architecture for easy extension
- Real-time sensor data integration
- Path planning and following
- Emergency stop functionality

## Available MCP Tools
- `stand_up` / `sit_down` - Basic posture control
- `detect_door` - Door detection and analysis
- `navigate_to` - Navigate to specific coordinates
- `detect_obstacles` - Obstacle detection and mapping
- `scan_room` - 360-degree room scanning
- `follow_path` - Multi-waypoint path following
- `open_door` - Door opening mechanism
- `get_battery_status` - Battery monitoring
- `emergency_stop` - Safety stop function
- `get_sensor_data` - Comprehensive sensor readings
- `set_movement_speed` - Speed control
- `get_current_position` - Position and orientation tracking

## Getting Started
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the example: `python example_usage.py`
4. Integrate with your MCP server setup

## Repository Structure
- `src/robot_dog_tools.py` - Core RobotDog control methods
- `src/mcp_server.py` - MCP server configuration and tool definitions
- `example_usage.py` - Example usage and demonstration
- `requirements.txt` - Python dependencies
- `tests/` - Test cases and validation scripts (to be added)

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.
