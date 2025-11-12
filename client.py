import os
os.environ["OMP_NUM_THREADS"] = "1"  # Fix for sklearn OpenMP TLS issue
os.environ["LD_PRELOAD"] = "/home/ias/satya/robotdog_guide/.venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0"

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import asyncio
import re
from langchain_ollama import ChatOllama
from src.rag_server import config
from src.mcp_servers.rag_module import call_RAG_generate_context_query
from src.rag_server.src.logger import set_logger
from src.rag_server.src.voiceAssistant import VoiceAssistant

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = "qwen3:4B"

# MCP logger
mcp_logger = set_logger("MCP_Client")

OUTPUT_DIR = config.OUTPUT_DIR

def clean_response(text):
    """Extract content that comes after <think> tags"""
    # Find the last </think> tag and get everything after it
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match:
        # Return the content after the last </think> tag
        return match.group(1).strip()
    else:
        # If no think tags found, return the original text
        return text.strip()

def initialize_system():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHROMA_PATH, exist_ok=True)
    assistant = VoiceAssistant(config=config, logger=mcp_logger)
    mcp_logger.info("System initialized.")
    return assistant

async def main(query=None):
    assistant = initialize_system()
    client = MultiServerMCPClient(
        {
            "robot_dog_tools_server": {
                "command": "python",
                "args": [os.path.join(os.path.dirname(__file__), "src", "robot_dog_tools_server.py")],
                "transport": "stdio"
            }
        }
    )
    
    tools = await client.get_tools()
    llm = ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        validate_model_on_init=True,
        temperature=0.2,
    )

    agent = create_agent(
        llm, tools
    )
    mcp_logger.info("Agent created successfully.")

    messages = {"messages": [
        {"role": "system", "content": "You are a RobotDog assistant for indoor navigation. Answer always in 'first person'."},
    ]}

    # response = await agent.ainvoke(messages)
    # mcp_logger.info("Response obtained: " + clean_response(response['messages'][-1].content))

    is_first_interaction = [True]
        
    """Starts the interactive voice assistant loop."""
    if is_first_interaction[0]:
        # assistant.speak_gtts("dog-bark.wav")
        # text = "Hi! I am your RoboDog guide Scooby Doo. I am an ongoing project at the IAS institute at University of Stuttgart. You can ask me questions related to the institute or any other question as well. I will try to answer to the best of my knowledge. If you need directions in the institute building just say directions."
        text = "Hi! I am your RoboDog guide Scooby Doo. \n"
        assistant.logger.info("Starting interaction with user.\n ROBOT-DOG: %s", text)
        # assistant.speak_gtts(text)
        # assistant.speak_gtts("How can I help you today?")
        # assistant.speak_gtts("You can say 'exit', 'quit', or 'stop' to end the interaction.")
        assistant.logger.info("You can say 'exit', 'quit', or 'stop' to end the interaction.")

        while True:
            # query = self.get_voice_input()
            query = await assistant.get_text_input()
            # assistant.logger.info("You said: %s", query)
            # assistant.speak_gtts(f"You said: {query}")
            assistant.logger.info("USER SAID: %s", query)
            if query in ["exit", "quit", "stop"]:
                # assistant.speak_gtts("It was nice interacting with you. Have a nice day ahead!")
                assistant.logger.info("Exiting the interaction loop.")
                # assistant.close()
                break
    
            # get modified query from RAG system
            if is_first_interaction[0]: # only if its the first query then need to call RAG
                mcp_logger.info("First interaction.")
                is_first_interaction[0] = False  # Set to False after the first interaction
                # append user's initial query
                messages["messages"].append({"role": "user", "content": query})
                # assistant.speak_gtts("Let me think about it.")
                modified_query = call_RAG_generate_context_query(query) # this will use another LLM only for RAG
                assistant.logger.info("Modified query from RAG: %s", modified_query)
                # assistant.speak_gtts(modified_query)
                messages["messages"].append({"role": "system", "content": modified_query})
                
            else: # after first interaction, make new message thread and include previous context
                mcp_logger.info("Subsequent interaction.")
                messages = {"messages": []}
                messages["messages"].append({"role": "system", "content": "The user has asked a follow-up question. Answer based on previous context and tools available."})
                messages["messages"].append({"role": "user", "content": query})
                # assistant.speak_gtts("Let me think about it.")
                
                # TODO: need to implement memory management for follow-up questions

            messages["messages"].append({"role": "user", "content": "Based on the above context, answer the question and take necessary actions."})
            response = await agent.ainvoke(messages)
            final_response = clean_response(response['messages'][-1].content)
            
            # TODO: if we fail to get proper response, we can guid the agent to RAG again for better response
            
            mcp_logger.info("Final response obtained: " + final_response)
            # assistant.speak_gtts(final_response)
            assistant.logger.info("ROBOT-DOG: %s", final_response)
            
if __name__ == "__main__":
    query = "Who is Joachim Grimstad in University of Stuttgart? Give me his room number and take me there if possible."
    asyncio.run(main())