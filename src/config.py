context_LLM_model = "qwen2.5-coder:1.5b"              # LLM-1: Context processing & clarification
conversation_LLM_model = "qwen2.5-coder:1.5b"        # LLM-2: Conversation responses
rag_LLM_model = "qwen2.5-coder:1.5b"                 # LLM-3: RAG with action classification
action_planner_LLM_model = "qwen2.5-coder:1.5b"     # LLM-4: Action planning
ollama_base_url = "http://localhost:11434"
mcp_service_url = "http://localhost:5000/mcp"

# ACTION THRESHOLD CHECK
ACTION_CONFIDENCE_THRESHOLD = 0.7