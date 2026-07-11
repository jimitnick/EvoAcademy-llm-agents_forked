import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
# Architect model for task planning and tutoring
architect_llm = ChatNVIDIA(
  model="nvidia/nemotron-3-ultra-550b-a55b",
  api_key=os.getenv("NVIDIA_API_KEY"), 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  reasoning_budget=16384,
  chat_template_kwargs={"enable_thinking":True},
)

# Coder model for parallel code block writing
coder_llm = ChatNVIDIA(
  model="nvidia/nemotron-3-ultra-550b-a55b",
  api_key=os.getenv("NVIDIA_API_KEY"), 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  reasoning_budget=16384,
  chat_template_kwargs={"enable_thinking":True},
)