import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Architect model for task planning and tutoring
architect_llm = ChatOpenAI(
    model="meta-llama/llama-3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Coder model for parallel code block writing
coder_llm = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
