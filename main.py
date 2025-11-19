from random import random
from src.graph.workflow import build_robotdog_workflow_graph
from langchain_core.messages import SystemMessage
import asyncio

async def main():

    print("Hello from robotdog-guide!")
    robot_graph = await build_robotdog_workflow_graph()
    
    # # Save the graph as PNG
    # graph_png = robot_graph.get_graph(xray=True).draw_mermaid_png()
    # with open("robotdog_graph5.png", "wb") as f:
    #     f.write(graph_png)
    # print("Graph saved.")

    final_state = {"exit": False}
    initial_state = {"start_conversation": True,
                     "messages": [SystemMessage(content="You are RobotDog, a helpful assistant who can "
                     "listen to human speech, process it, and respond appropriately. Also perform physical actions " \
                     "as needed to assist the human.")]}
    
    while True:

        # threads for history saving
        thread_id = random.randint(1, 1_000_000)  # generate random thread ID every time for unique history
        config_thread = {"configurable": {"thread_id": thread_id}}

        # TODO: speak few sentences to start conversation
            
        if final_state.get("exit") == True:
            print("RobotDog conversation ended.")
            break
        
        print("Starting robotdog conversation ...")
        final_state = await robot_graph.invoke(initial_state, config=config_thread)
        
    
    
if __name__ == "__main__":
    asyncio.run(main())