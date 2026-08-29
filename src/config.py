import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env")

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Chat model
CHAT_MODEL = "gpt-4o-mini"