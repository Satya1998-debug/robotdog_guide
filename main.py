from src.graph.workflow import build_robotdog_workflow_graph
from langchain_core.messages import SystemMessage

def main():
    print("Hello from robotdog-guide!")
    robot_graph = build_robotdog_workflow_graph()
    
    # Save the graph as PNG
    # graph_png = robot_graph.get_graph(xray=True).draw_mermaid_png()
    # with open("robotdog_graph.png", "wb") as f:
    #     f.write(graph_png)
    # print("Graph saved as 'robotdog_graph.png'")

    final_state = {"exit": False}
    
    while True:
            
        if final_state.get("exit") == True:
            print("RobotDog conversation ended.")
            break
        
        initial_state = {"start_conversation": True}
        print("Starting robotdog conversation ...")
        final_state = robot_graph.invoke(initial_state)
        
    
    
if __name__ == "__main__":
    main()
