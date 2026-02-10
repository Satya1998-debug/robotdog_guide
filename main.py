import random
from src.graph.workflow import build_robotdog_workflow_graph
from langchain_core.messages import SystemMessage
from src.logger import logger
from langgraph.types import Command

def get_user_permission():
    # user text from speech
    from src.nodes.speech_process_nodes import speech_to_text
    input_text = speech_to_text("")
    if "yes" in input_text.lower() or "approve" in input_text.lower() or "okay" in input_text.lower():
        return {"approved": True}
    else:        
        return {"approved": False}
    

def main():

    print("Hello from robotdog-guide!")
    robot_graph = build_robotdog_workflow_graph()
    
    # # # Save the graph as PNG
    # graph_png = robot_graph.get_graph(xray=True).draw_mermaid_png()
    # with open("robotdog_graph10.png", "wb") as f:
    #     f.write(graph_png)
    # print("Graph saved.")    
    
    logger.info("Starting RobotDog conversation loop...")
    while True:
        
        initial_state = { 
                         "start_conversation": True,
                         "chat_history": [SystemMessage(content="You are RobotDog, a helpful assistant who can "
                                                        "listen to human speech, process it, and respond appropriately. Also perform physical actions " \
                                                            "as needed to assist the human.")],
                         "llm_tool_call_once": False
                         }

        # threads for history saving
        thread_id = random.randint(1, 1_000_000)  # generate random thread ID every time for unique history
        config_thread = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}  # this is number of nodes it will execute before hitting a END condition
        logger.info(f"A RobotDog session with thread ID: {thread_id} started.")

        # TODO: speak few sentences to start conversation
        
        print("Starting robotdog conversation ...")
        result = robot_graph.invoke(initial_state, config=config_thread, )
        
        interrupt = result.get("__interrupt__", [])
        if interrupt:
            from src.nodes.speech_process_nodes import text_to_speech
            print(interrupt[0])
            text = interrupt[0].value.get("message")
            text_to_speech(text)
        
        # need to resume graph
        final_state = robot_graph.invoke(Command(resume=get_user_permission()), config=config_thread)
        
        # TODO: add audio saying goodbye
        logger.info("Your RobotDog session has ended. Thank you!")
        
if __name__ == "__main__":
    main()