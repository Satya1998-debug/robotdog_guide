
# qwen2.5-coder 
context_LLM_model = "phi3:mini"              # LLM-1: Context processing & clarification
conversation_LLM_model = "phi3:mini"        # LLM-2: Conversation responses
rag_LLM_model = "phi3:mini"                 # LLM-3: RAG with action classification
action_planner_LLM_model = "phi3:mini"     # LLM-4: Action planning
tool_LLM_model = "phi3:mini"            # LLM-5: MCP tool usage
summarizer_LLM_model = "phi3:mini"        # LLM-6: Summarization for feedback
ollama_base_url = "http://localhost:11434"
mcp_service_url = "http://localhost:5000/mcp"

# ACTION THRESHOLD CHECK
ACTION_CONFIDENCE_THRESHOLD = 0.7