

# qwen7b for orin
# phi4 phi4:14b-q4_K_M for orin

context_LLM_model = "qwen2.5:14b-instruct"              # LLM-1: Context processing & clarification
conversation_LLM_model = "phi4:14b"        # LLM-2: Conversation responses
clarrification_LLM_model = "phi4:14b"        # LLM for clarification questions
rag_LLM_model = "qwen2.5:14b-instruct"                 # LLM-3: RAG with action classification
action_planner_LLM_model = "qwen2.5:14b-instruct"     # LLM-4: Action planning
tool_LLM_model = "qwen2.5:7b-instruct"            # LLM-5: MCP tool usage
summarizer_LLM_model = "phi4:14b"        # LLM-6: Summarization for feedback

ENABLE_SUMMARY = True  # this will enable summarizer_node in workflow all the time

ollama_base_url = "http://localhost:11434"

## for orin
# context_LLM_model        = "qwen2.5:7b-instruct"
# conversation_LLM_model   = "phi4:14b-q4_K_M"  # or phi3.5-mini
# rag_LLM_model            = "qwen2.5:7b-instruct"
# action_planner_LLM_model = "qwen2.5:7b-instruct"
# tool_LLM_model           = "qwen2.5:7b-instruct"
# summarizer_LLM_model     = "phi3.5:mini"

# ACTION THRESHOLD CHECK
ACTION_CONFIDENCE_THRESHOLD = 0.7