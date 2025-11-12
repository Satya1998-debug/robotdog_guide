from src.graph.state import RobotDogState
from langgraph.graph import StateGraph, START, END
from src.nodes.decision_nodes import context_processor, decision_node, conversation_node, clarification_node, decide_query_intention
from src.nodes.rag_nodes import rag_pipeline
from src.nodes.action_nodes import action_planner, mcp_client, action_classifier
from src.nodes.feedback_nodes import summarizer_node, perception_feedback
from src.nodes.speech_process_nodes import speech_to_text, text_to_speech

def build_robotdog_workflow_graph() -> StateGraph[RobotDogState]:
# Graph definition
    graph = StateGraph(RobotDogState)

    # nodes
    graph.add_node("input_node", speech_to_text)

    graph.add_node("context_processor_node", context_processor) # use LLM-1
    graph.add_node("decision_node", decision_node)

    graph.add_node("conversation_node", conversation_node) # use LLM-2
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("rag_node", rag_pipeline) # use LLM-3
    graph.add_node("action_classifier_node", action_classifier)
    graph.add_node("action_planner_node", action_planner)

    graph.add_node("mcp_node", mcp_client)
    graph.add_node("perception_feedback_node", perception_feedback)
    graph.add_node("summarizer_node", summarizer_node) # use LLM-4
    graph.add_node("output_node", text_to_speech)

    # edges
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "context_processor_node")
    graph.add_edge("context_processor_node", "decision_node")  

    graph.add_conditional_edges("decision_node", decide_query_intention) # can call 4 nodes (conversation_node, clarification_node, rag_node, action_planner_node)

    graph.add_edge("conversation_node", "output_node")
    graph.add_edge("clarification_node", "conversation_node")
    graph.add_edge("rag_node", "action_classifier_node")
    graph.add_edge("action_classifier_node", "mcp_node")
    graph.add_edge("action_planner_node", "mcp_node")
    graph.add_edge("mcp_node", "perception_feedback_node")
    graph.add_edge("perception_feedback_node", "summarizer_node")
    graph.add_edge("summarizer_node", "output_node")
    graph.add_edge("output_node", END)

    # graph compile
    graph.compile()
    return graph

