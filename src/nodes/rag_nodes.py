from multiprocessing import context
from src.graph.state import RobotDogState
from src.graph.schemas import RAGNodeOutput
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.config import ollama_base_url, rag_LLM_model
from src.logger import logger

# rag related imports
from src.rag_server.text_scraper import TextScraper
from src.rag_server.databaseHandler import DatabaseHandler
from src.rag_server.documentProcessor import DocumentProcessor
import src.rag_server.config as rag_config
import os

# Lazy initialization - only create when first needed to avoid blocking on import
_vector_db_handler = None

def get_vector_db_handler():
    """Lazy initialization of DatabaseHandler to avoid blocking import."""
    global _vector_db_handler
    if _vector_db_handler is None:
        try:
            logger.info("[RAG] Initializing DatabaseHandler...")
            _vector_db_handler = DatabaseHandler(path=rag_config.CHROMA_PATH, 
                                                model_name=rag_config.EMBEDDING_MODEL_NAME, 
                                                logger=logger)
            logger.info("[RAG] DatabaseHandler initialized successfully")
        except Exception as e:
            logger.error(f"[RAG] Failed to initialize DatabaseHandler: {e}")
            raise
    return _vector_db_handler

# Use LLM-3 (RAG model) to generate response
rag_llm = ChatOllama(
        model=rag_LLM_model,  # LLM-3
        base_url=ollama_base_url,
        validate_model_on_init=True,
        temperature=0.2,  # low temperature for factual accuracy
    )


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
    
    messages = []
    summary = state.get("summary", "")
    
    # Retrieve relevant documents from vector database with error handling
    try:
        logger.info("[rag_node] Starting document retrieval...")
        retrieved_docs = get_rag_output(summary + "\n" + query + "\n" + str(context_tags))
        retrieved_context = "\n\n".join([doc["content"] if isinstance(doc, dict) else str(doc) for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
        logger.info(f"[rag_node] Retrieved {len(retrieved_docs) if retrieved_docs else 0} docs | Query: {query[:50]}")
    except Exception as e:
        logger.error(f"[rag_node] Error retrieving documents: {e}")
        retrieved_docs = []
        retrieved_context = "Error retrieving documents from knowledge base."

    if summary:  # insert the summary first
        summary_system_msg = f"Previous conversation summary: {summary}"
        messages.append(SystemMessage(content=summary_system_msg))
    
    messages.extend(state.get("chat_history", [])) # include prior chat history after previous session's summary

    # Use LLM-3 to generate RAG-based response with structured output
    system_prompt = system_prompt = """
        You are a highly reliable robot assistant. You answer questions ONLY using the retrieved institutional context and your reasoning rules.

        Your responsibilities:
        1. Understand the user query and align it with the retrieved context.
        2. Decide whether the user requires physical robot action (navigation, perception, manipulation).
        3. Extract structured details: full names, departments, room numbers, building names, office locations.
        4. Produce a confidence score (0.0–1.0) for whether robot action is needed.
        5. Generate a safe and factual informational response.

        --- ACTION DECISION RULES ---
        Robot action IS REQUIRED when the user:
        • Wants to go somewhere: “take me to…”, “bring me to…”, “navigate to…”
        • Wants the robot to guide/escort them
        • Wants the robot to locate a person/place physically
        • Wants physical help or physical interaction (“pick up”, “follow him”, “carry this”, etc.)

        Robot action is NOT REQUIRED when the user:
        • Requests information only → “where is…”, “what is…”, “who is…”
        • Asks about schedules, phone numbers, office hours, policies
        • Asks about general institutional knowledge

        --- OUTPUT RULES ---
        • Use ONLY the retrieved context to answer. If the answer is missing, say so clearly.
        • Never invent room numbers, titles, or building names.
        """

    user_prompt = user_prompt = f"""
        User Query:
        \"\"\"{query}\"\"\"

        Context Tags (from classifier):
        {context_tags}

        Intent Classification Reasoning:
        {intent_reasoning}

        Retrieved Context (RAG results):
        \"\"\"{retrieved_context}\"\"\"

        Your tasks:
        1. Summarize the retrieved context relevant to the query (short, factual).
        2. Rewrite the user query with full explicit details if possible 
        (e.g., “take me to Dr. Smith” → “take me to Dr. John Smith, Office 305, Building A”).
        3. Decide if robot action is required:
        - "yes" or "no"
        4. Provide a confidence score (0.0 - 1.0) for requiring robot action.
        5. Extract target location (only room number) if the context provides it.
        6. Extract person name (only full name of a person) if available.
        7. Generate the final informational response:
        - If NO action needed → answer using retrieved context.
        - If action IS needed → describe clearly what the robot should do.
        8. Output a list of probable robot actions:
        Example allowed actions: "navigation", "stand", "sit", "speak", "sit". 
        Leave empty if no action is needed.

        Be precise, grounded, and avoid adding any information that is not explicitly present in the retrieved context.
        """

    messages.extend([ # include current node's msgs
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Invoke LLM with error handling
    try:
        logger.info("[rag_node] Invoking RAG LLM for structured output...")
        structured_llm = rag_llm.with_structured_output(RAGNodeOutput)
        rag_output = structured_llm.invoke(messages)
        logger.info("[rag_node] RAG LLM invocation completed successfully")
    except Exception as e:
        logger.error(f"[rag_node] Error invoking RAG LLM: {e}")
        # Fallback output if LLM fails
        rag_output = RAGNodeOutput(
            retrieved_context=retrieved_context[:500] if retrieved_context else "No context available",
            rag_modified_query=query,
            requires_robot_action=False,
            action_confidence=0.0,
            target_location=None,
            target_person=None,
            probable_actions=[],
            informational_response=f"I encountered an error processing your request. Please try rephrasing your question."
        )

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
    try:
        OUTPUT_DIR = rag_config.OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(rag_config.CHROMA_PATH, exist_ok=True)

        # Get the handler (lazy initialization)
        vector_db_handler = get_vector_db_handler()
        
        # Query with logging
        logger.info(f"[RAG] Querying vector database...")
        retrieved_docs = vector_db_handler.query(query) # get context from documents
        logger.info(f"[RAG] Query completed, retrieved {len(retrieved_docs) if retrieved_docs else 0} documents")

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
    
    except Exception as e:
        logger.error(f"[RAG Pipeline] Error in retrieving RAG output: {e}")
        return []