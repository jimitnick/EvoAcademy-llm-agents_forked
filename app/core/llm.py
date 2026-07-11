import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
# Architect model for task planning and tutoring
architect_llm = ChatNVIDIA(
  model="nvidia/nemotron-3-ultra-550b-a55b",
  api_key="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", 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  reasoning_budget=16384,
  chat_template_kwargs={"enable_thinking":True},
)

# Coder model for parallel code block writing
coder_llm = ChatNVIDIA(
  model="nvidia/nemotron-3-ultra-550b-a55b",
  api_key="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", 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  reasoning_budget=16384,
  chat_template_kwargs={"enable_thinking":True},
)