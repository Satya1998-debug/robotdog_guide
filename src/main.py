from src.graph.workflow import build_robotdog_workflow_graph

def main():
    print("Hello from robotdog-guide!")
    robot_graph = build_robotdog_workflow_graph()
    
    session_state = {}
    while True:
        user_input = input("You: ")
        
        # Merge outputs back into the session state
        
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Exiting robotdog-guide. Goodbye!")
            break
        
        initial_state = {"input_text": user_input}
        final_state = robot_graph.invoke(initial_state)
        final_response = final_state.get("response", "No response generated.")
        print(f"RobotDog: {final_response}")
        
        session_state.update(final_state)

    
    
if __name__ == "__main__":
    main()
