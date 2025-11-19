from src.graph.state import RobotDogState
from langgraph.graph import StateGraph, START, END
from src.nodes.decision_nodes import context_processor, decision_node, conversation_node, \
    clarification_node, decide_query_intention, should_continue, decide_mcp_execution
from src.nodes.rag_nodes import rag_pipeline
from src.nodes.action_nodes import action_planner, action_classifier, call_mcp_model
from src.nodes.speech_process_nodes import speak_to_human, listen_to_human
from langgraph.checkpoint.memory import MemorySaver

import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize MCP client globally
_mcp_client = None
_mcp_tools = None

async def get_mcp_tools_async():
    """Get MCP tools asynchronously (cached after first call)"""
    global _mcp_client, _mcp_tools
    
    if _mcp_tools is None:
        _mcp_client = MultiServerMCPClient(
            {
                "robot_dog_tools_server": {
                    "command": "python",
                    "args": [os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "robot_dog_tools_server.py")],
                    "transport": "stdio"
                }
            }
        )
        # Get tools asynchronously
        _mcp_tools = await _mcp_client.get_tools()
        print(f"[Workflow] MCP tools initialized: {len(_mcp_tools)} tools available")
    
    return _mcp_tools

def get_mcp_tools():
    """
    Get cached MCP tools (synchronous access).
    Must be called AFTER async initialization via get_mcp_tools_async().
    Used by action nodes that run synchronously in the graph.
    """
    global _mcp_tools
    if _mcp_tools is None:
        raise RuntimeError("MCP tools not initialized. Call get_mcp_tools_async() first during graph build.")
    return _mcp_tools


async def build_robotdog_workflow_graph() -> StateGraph[RobotDogState]:
    """
    Build LangGraph workflow with MCP tool integration.
    """
    
    # Get MCP tools once at graph build time (async)
    mcp_tools = await get_mcp_tools_async()

    # Graph definition
    graph = StateGraph(RobotDogState)

    # Speech & Decision nodes
    graph.add_node("listen_to_human_node", listen_to_human)
    graph.add_node("context_processor_node", context_processor) # use LLM-1
    graph.add_node("decision_node", decision_node)

    # Conversation nodes
    graph.add_node("conversation_node", conversation_node) # use LLM-2
    graph.add_node("clarification_node", clarification_node)
    
    # RAG & Action nodes
    graph.add_node("rag_node", rag_pipeline) # use LLM-3
    graph.add_node("action_classifier_node", action_classifier)
    graph.add_node("action_planner_node", action_planner) # use LLM-4

    # MCP nodes (LangGraph pattern with ToolNode)
    graph.add_node("mcp_llm_node", call_mcp_model)  # Model decides which tools to call
    graph.add_node(ToolNode(mcp_tools))      # the node name has to be default "tools", as lagraph expects

    # Feedback nodes
    # graph.add_node("perception_feedback_node", perception_feedback)
    # graph.add_node("summarizer_node", summarizer_node)
    graph.add_node("speak_to_human_node", speak_to_human)

    # Edges - main flow
    graph.add_edge(START, "listen_to_human_node")
    graph.add_edge("listen_to_human_node", "context_processor_node")
    graph.add_edge("context_processor_node", "decision_node")  

    # Decision routing
    graph.add_conditional_edges("decision_node", decide_query_intention)

    # Conversation path
    graph.add_edge("conversation_node", "speak_to_human_node")
    graph.add_edge("clarification_node", "speak_to_human_node")
    
    # RAG path
    graph.add_edge("rag_node", "action_classifier_node")
    graph.add_conditional_edges("action_classifier_node", decide_mcp_execution)
    
    # Action planner path (direct to MCP)
    graph.add_edge("action_planner_node", "mcp_llm_node")
    
    # MCP execution loop (LangGraph pattern)
    # call_mcp_model -> tools_condition -> either "tools" or "__end__"
    # tools -> call_mcp_model (loop until no more tools needed)
    # When no more tools, exit loop and go to perception_feedback

    # graph.add_conditional_edges("mcp_llm_node", tools_condition)
    graph.add_conditional_edges(
            "mcp_llm_node",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "mcp_llm_node", "__end__": "speak_to_human_node"},
        )
    # graph.add_edge("tools", "mcp_llm_node")  # tools_condition will either go to "tools" or END by default
    
    graph.add_conditional_edges("speak_to_human_node", should_continue)  # loop back to listening

    # Feedback flow
    # graph.add_edge("perception_feedback_node", "summarizer_node")
    # graph.add_edge("summarizer_node", "speak_to_human_node")

    # for history tracking
    checkpointer = MemorySaver()


    return graph.compile(checkpointer=checkpointer)
