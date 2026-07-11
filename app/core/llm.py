# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()
# # Architect model for task planning and tutoring
# architect_llm = ChatOpenAI(
#   base_url="https://router.huggingface.co/v1",
#   model="zai-org/GLM-5.2:novita",
#   api_key=os.getenv("HF_TOKEN"), 
#   temperature=1,
#   top_p=0.95,
#   max_tokens=16384,
#   timeout=300,  # Increased timeout for long thinking times
# )

# # Coder model for parallel code block writing
# coder_llm = ChatOpenAI(
#   base_url="https://router.huggingface.co/v1",
#   model="zai-org/GLM-5.2:novita",
#   api_key=os.getenv("HF_TOKEN"), 
#   temperature=1,
#   top_p=0.95,
#   max_tokens=16384,
#   timeout=300,  # Increased timeout for long thinking times
# )

# import os
# from dotenv import load_dotenv
# from langchain_nvidia_ai_endpoints import ChatNVIDIA

# load_dotenv()
# # Architect model for task planning and tutoring
# architect_llm = ChatNVIDIA(
#   model="nvidia/nemotron-3-ultra-550b-a55b",
#   api_key=os.getenv("NVIDIA_API_KEY"), 
#   temperature=1,
#   top_p=0.95,
#   max_tokens=16384,
#   reasoning_budget=16384,
#   chat_template_kwargs={"enable_thinking":True},
# )

# # Coder model for parallel code block writing
# coder_llm = ChatNVIDIA(
#   model="nvidia/nemotron-3-ultra-550b-a55b",
#   api_key=os.getenv("NVIDIA_API_KEY"), 
#   temperature=1,
#   top_p=0.95,
#   max_tokens=16384,
#   reasoning_budget=16384,
#   chat_template_kwargs={"enable_thinking":True},
# )

# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()
# architect_llm = ChatOpenAI(
#   base_url="https://api.endpoints.deepinfra.com/v1",
#   model="meta-llama/llama-4-scout-17b-16e",
#   api_key=os.getenv("DEEPINFRA_API_KEY"), 
#   temperature=1,
#   max_tokens=16384,
#   timeout=300,
# )

# coder_llm = ChatOpenAI(
#   base_url="https://api.endpoints.deepinfra.com/v1",
#   model="meta-llama/llama-4-scout-17b-16e",
#   api_key=os.getenv("DEEPINFRA_API_KEY"), 
#   temperature=1,
#   max_tokens=16384,
#   timeout=300,
# )

import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

# Architect model for task planning and tutoring
architect_llm = ChatNVIDIA(
  model="z-ai/glm-5.2",
  api_key=os.getenv("NVIDIA_API_KEY"), 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  timeout=300,  # Increased timeout
)

# Coder model for parallel code block writing
coder_llm = ChatNVIDIA(
  model="z-ai/glm-5.2",
  api_key=os.getenv("NVIDIA_API_KEY"), 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  timeout=300,  # Increased timeout
)
