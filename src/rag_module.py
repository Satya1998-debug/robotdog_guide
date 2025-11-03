"""
MCP Tools for RobotDog Indoor Navigation
Dummy methods that will be exposed as MCP tools for controlling and interacting with RobotDog
"""
import os
from typing import Dict
# from mcp.server.fastmcp import FastMCP

os.sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_server.src.voiceAssistant import VoiceAssistant
from rag_server.src.databaseHandler import DatabaseHandler
from rag_server.src.answerGenerator import AnswerGenerator
import rag_server.config as config
from rag_server.src.logger import set_logger


# mcp = FastMCP("RAG_tools_server")
logger = set_logger("RAG_tools_server")

class RAGTools:
    """RAG Tools for RobotDog query validation via MCP"""
    def __init__(self):
        self.components = self.initialize_components()

    def initialize_components(self) -> Dict:
        vector_db_handler = DatabaseHandler(path=config.CHROMA_PATH, model_name=config.EMBEDDING_MODEL_NAME, logger=logger)
        ans_generator = AnswerGenerator(db_handler=vector_db_handler, logger=logger)
        logger.info("RAG Components initialized successfully.")
        return {
            "vector_db_handler": vector_db_handler,
            "ans_generator": ans_generator
        }
        
    def generate_answer(self, query: str) -> str:
        """Generate an answer for the given query using the answer generator.

        Args:
            query (str): The input query.

        Returns:
            str: The generated answer.
        """
        return self.components["ans_generator"].generate_ollama(query, use_mcp=False)

def call_RAG_generate_context_query(query: str) -> str:
    rag_server = RAGTools()
    """MCP tool wrapper for modifying the original query using RAG system. The method is called for all general purpose queries and university-related questions.

    General queries: like science, weather, sports, current affairs, non-university related questions.
    University-related questions: like person-specific info, names, room numbers, offices, course information, enrollment details, campus facilities, etc.

    Args:
        query (str): The input query containing user's original query that has general or university-related intention.

    Returns:
        str: The modified answer explaining the steps to be taken to answer the original query, which is used later for MCP agent's final response and tool calling with 'Action' intent.
    """
    modified_query = rag_server.generate_answer(query)
    logger.info(f"Modified query from RAG obtained.")
    return modified_query

# Example usage
if __name__ == "__main__":
    # mcp.run(transport="stdio")
    print(call_RAG_generate_context_query("Who is Joachim Grimstad in University of Stuttgart? Give me his room number and take me there."))