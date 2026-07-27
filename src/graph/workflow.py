from src.graph.state import RobotDogState
from langgraph.graph import StateGraph, START, END
from src.nodes.decision_nodes import context_processor, decision_node, conversation_node, \
    clarification_node, decide_query_intention, should_continue, decide_tool_call_execution
from src.nodes.rag_nodes import rag_pipeline
from src.tools_servers.tools import get_all_tools
from src.nodes.action_nodes import action_planner, action_classifier, call_llm_with_tools
from src.nodes.speech_process_nodes import speak_to_human, listen_to_human
from src.nodes.feedback_nodes import summarizer_node
from langgraph.checkpoint.memory import MemorySaver

import os

from langgraph.prebuilt import ToolNode, tools_condition

from src.logger import logger


def build_robotdog_workflow_graph() -> StateGraph[RobotDogState]:
    """
    Build LangGraph workflow with MCP tool integration.
    """
    all_tools = get_all_tools()

    # graph definition
    graph = StateGraph(RobotDogState)

    # speech & decision nodes
    graph.add_node("listen_to_human_node", listen_to_human)
    graph.add_node("context_processor_node", context_processor) # use LLM-1
    graph.add_node("decision_node", decision_node)

    # conversation nodes
    graph.add_node("conversation_node", conversation_node) # use LLM-2
    graph.add_node("clarification_node", clarification_node)
    
    # RAG & action nodes
    graph.add_node("rag_node", rag_pipeline) # use LLM-3
    graph.add_node("action_classifier_node", action_classifier)
    # graph.add_node("action_planner_node", action_planner) # use LLM-4

    # LLM with tools nodes (LangGraph pattern with ToolNode)
    graph.add_node("llm_tools_node", call_llm_with_tools)  # Model decides which tools to call
    graph.add_node(ToolNode(all_tools))      # the node name has to be default "tools", as lagraph expects

    # feedback nodes
    # graph.add_node("perception_feedback_node", perception_feedback)
    graph.add_node("summarizer_node", summarizer_node)
    graph.add_node("speak_to_human_node", speak_to_human)

    # edges - main flow
    graph.add_edge(START, "listen_to_human_node")
    graph.add_conditional_edges("listen_to_human_node", should_continue)
    graph.add_edge("context_processor_node", "decision_node")  

    # decision routing
    graph.add_conditional_edges("decision_node", decide_query_intention)

    # conversation path
    graph.add_edge("conversation_node", "summarizer_node")
    graph.add_edge("clarification_node", "summarizer_node")
    
    # RAG path
    graph.add_edge("rag_node", "action_classifier_node")
    graph.add_conditional_edges("action_classifier_node", decide_tool_call_execution)
    
    # action planner path (direct to LLM with tools)
    # graph.add_edge("action_planner_node", "llm_tools_node")
    
    # llm_tools_node -> tools_condition -> either "tools" or "__end__"
    # tools -> llm_tools_node (loop until no more tools needed)
    # When no more tools, exit loop and go to perception_feedback

    # graph.add_conditional_edges("mcp_llm_node", tools_condition)
    graph.add_conditional_edges(
            "llm_tools_node",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "tools", "__end__": "conversation_node"},
        )
    graph.add_edge("tools", "llm_tools_node")

    graph.add_edge("summarizer_node", "speak_to_human_node")
    # graph.add_edge("tools", "mcp_llm_node")  # tools_condition will either go to "tools" or END by default
    
    graph.add_edge("speak_to_human_node", "listen_to_human_node")  # loop back to listening

    # for history tracking
    checkpointer = MemorySaver() # save in ram
    
    # debug: print graph structure
    logger.info("\n=========== GRAPH Initialized ===========\n")

    return graph.compile(checkpointer=checkpointer)
