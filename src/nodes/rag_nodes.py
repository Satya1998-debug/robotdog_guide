from multiprocessing import context
from src.graph.state import RobotDogState
from src.graph.schemas import RAGNodeOutput
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.config import ollama_base_url
from src.logger import logger

# rag related imports
from src.rag_server.text_scraper import TextScraper
from src.rag_server.voiceAssistant import VoiceAssistant
from src.rag_server.databaseHandler import DatabaseHandler
from src.rag_server.documentProcessor import DocumentProcessor
from src.rag_server.answerGenerator import AnswerGenerator
import src.rag_server.config as rag_config
import os


def rag_pipeline(state: RobotDogState) -> RobotDogState:
    """
    Retrieve and generate response using RAG with structured output.
    Uses LLM-3 for RAG-based answer generation.
    """
    logger.info("[Node] -> rag_node")
    query = state.get("original_query", "")
    context_output = state.get("context_proc_node_output", {})
    context_tags = context_output.get("context_tags", {})
    intent_reasoning = state.get("decision_node_output", {}).get("intent_reasoning", "")
        
    # Retrieve relevant documents from vector database
    retrieved_docs = get_rag_output(query)
    retrieved_context = "\n\n".join([doc["content"] if isinstance(doc, dict) else str(doc) for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
    logger.info(f"[rag_node] Retrieved {len(retrieved_docs) if retrieved_docs else 0} docs | Query: {query[:50]}...")
    
    messages = []
    if state.get("summary", ""):  # insert the summary first
        summary_system_msg = f"Previous conversation summary: {state['summary']}"
        messages.append(SystemMessage(content=summary_system_msg))

    messages.extend(state.get("chat_history", [])) # include prior chat history after previous session's summary

    # Use LLM-3 to generate RAG-based response with structured output
    system_prompt = """You are a helpful robot assistant that answers questions based on retrieved institutional information.
        Your role is to:
        1. Analyze the user's query and the retrieved context
        2. Determine if the query requires physical robot action (navigation, manipulation) or just information
        3. Extract specific details like full names, room numbers, building locations
        4. Provide confidence score for whether robot action is needed
        5. Generate an informational response

        Robot action is needed when:
        - User wants to go somewhere ("take me to...", "navigate to...", "find...")
        - User wants robot to find/follow someone
        - User needs physical assistance
        
        Robot action is NOT needed when:
        - User asks for information only ("where is...", "what is...", "who is...")
        - User wants to know schedules, contact info, etc.
        
        Keep answers concise but informative."""

    user_prompt = f"""User query: "{query}"
        Context tags: {context_tags}
        Intent reasoning: {intent_reasoning}

        Retrieved context from knowledge base:
        {retrieved_context}

        Analyze and provide:
        1. Retrieved context (summarized)
        2. Modified query with full details (e.g., "take me to Dr. Smith" → "take me to Dr. John Smith, Room 305, Building A")
        3. Does this require robot action? (navigation, finding someone, physical movement)
        4. Confidence score (0.0-1.0) that robot action is needed
        5. Extract target location if found (e.g., "Room 305", "Building A", "Main Office")
        6. Extract full person name if applicable
        7. Informational response: 
           - If NO action needed: Direct answer to the question
           - If action needed: Summary of where/what the robot will do
        8. List probable robot actions based on the query: (keep the list empty if no action needed)
           - "navigation", "manipulation", "perception", etc. 
           - "other tools" like stand, sit, crawl, speak, etc.

        Be precise about action requirements."""

    messages.extend([ # include current node's msgs
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Use LLM-3 (RAG model) to generate response
    rag_llm = ChatOllama(
        model=state.get("rag_LLM_model", "qwen2.5-coder:1.5b"),  # LLM-3
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.2,  # low temperature for factual accuracy
    )

    structured_llm = rag_llm.with_structured_output(RAGNodeOutput)
    rag_output = structured_llm.invoke(messages)

    response_content = f"""RAG Retrieved Context: {rag_output.retrieved_context}\n\
        Modified Query: {rag_output.rag_modified_query}\n\
        Requires Robot Action: {rag_output.requires_robot_action}\n\
        Action Confidence: {rag_output.action_confidence}\n\
        Target Location: {rag_output.target_location}\n\
        Target Person: {rag_output.target_person}\n\
        Probable Actions: {', '.join(rag_output.probable_actions)}\n\
        Informational Response: {rag_output.informational_response}"""
    
    logger.info(f"[rag_node] Action required: {rag_output.requires_robot_action} | Confidence: {rag_output.action_confidence:.2f} | Location: {rag_output.target_location} | Person: {rag_output.target_person}")
    
    return {"rag_node_output": dict(rag_output), 
            "informational_response": rag_output.informational_response,
            "chat_history": [SystemMessage(content="RAG node: You are a helpful assistant that retrieves and analyzes context using RAG for robot actions."),
                             HumanMessage(content="Extract context and analyze the user query accordingly."), 
                             AIMessage(content=response_content)]}

def get_rag_output(query):
    """
    Helper to extract RAG context from vector database.
    """
    OUTPUT_DIR = rag_config.OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(rag_config.CHROMA_PATH, exist_ok=True)

    vector_db_handler = DatabaseHandler(path=rag_config.CHROMA_PATH, model_name=rag_config.EMBEDDING_MODEL_NAME, logger=logger)
    retrieved_docs = vector_db_handler.query(query) # get context from documents

    if rag_config.SCRAPE['need_scraping']: # if scrapping needed, as specified in config
        data_processor = DocumentProcessor(rag_config.CSV_FILE_PATH)  # needed for processing scraped data
        scraper = TextScraper(rag_config.SCRAPE["base_url"], rag_config.SCRAPE["data_dir"], rag_config.SCRAPE["max_pages"])
        logger.info("Starting web scraping...")
        scraper.scrape()
        scraper.save_to_csv()
        logger.info("Web scraping completed.")
        
        # normal docs chunks and room info chunks are combined
        text_chunks, metadatas = data_processor.get_combined_chunks_with_rooms(rag_config.ROOMS_CSV_PATH)
        vector_db_handler.store_documents(text_chunks, metadatas)
        logger.info(f"Stored {len(text_chunks)} chunks in ChromaDB")
        retrieved_docs = vector_db_handler.query(query) # get context from updated documents

    return retrieved_docs