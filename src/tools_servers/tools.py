
from src.tools_servers.robot_dog_tools import tools

def get_all_tools():
    all_tools = tools
    if all_tools is None:
        raise RuntimeError("Tools not initialized.")
    else:
        return all_tools