

# LLM MODEL CONFIGURATION
context_LLM_model = "qwen2.5:7b-instruct"              # LLM-1: Context processing & clarification
conversation_LLM_model = "qwen2.5:7b-instruct"        # LLM-2: Conversation responses
clarrification_LLM_model = "qwen2.5:7b-instruct"        # LLM for clarification questions
rag_LLM_model = "qwen2.5:7b-instruct" # LLM-3: RAG with action classification
# heavy LLM
action_planner_LLM_model = "deepseek-r1:7b-qwen-distill-q4_K_M"     # LLM-4: Action planning
# lighter LLMs
tool_LLM_model = "qwen2.5:3b-instruct"            # LLM-5: MCP tool usage
summarizer_LLM_model = "qwen2.5:3b-instruct"        # LLM-6: Summarization for feedback

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