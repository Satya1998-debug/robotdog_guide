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

# initialize vector DB handler globally and initialize once
vector_db_handler = DatabaseHandler(path=rag_config.CHROMA_PATH, 
                                    model_name=rag_config.EMBEDDING_MODEL_NAME, 
                                    logger=logger)


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
    
    # Retrieve relevant documents from vector database
    retrieved_docs = get_rag_output(summary + "\n" + query + "\n" + str(context_tags))
    retrieved_context = "\n\n".join([doc["content"] if isinstance(doc, dict) else str(doc) for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
    logger.info(f"[rag_node] Retrieved {len(retrieved_docs) if retrieved_docs else 0} docs | Query: {query[:50]}")

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
        5. Extract target location (room number, office, building) if the context provides it.
        6. Extract person name (full name and role) if applicable.
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

    # Use LLM-3 (RAG model) to generate response
    rag_llm = ChatOllama(
        model=rag_LLM_model,  # LLM-3
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
    try:
        OUTPUT_DIR = rag_config.OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(rag_config.CHROMA_PATH, exist_ok=True)

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
    
    except Exception as e:
        logger.error(f"[RAG Pipeline] Error in retrieving RAG output: {e}")
        return []