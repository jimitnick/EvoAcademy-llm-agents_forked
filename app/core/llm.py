import os
from dotenv import load_dotenv
from langchain_openapi import ChatOpenAi

load_dotenv()

architect_llm = ChatOpenAi(
    model="meta-llama/llama-3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url = "https://openrouter.ai/api/v1",
)

coder_llm = ChatOpenAi(
    model = "meta-llama/llama-3-8b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


