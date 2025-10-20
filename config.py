"""
Configuration file for Hybrid AI Travel Assistant.
Uses environment variables from .env file for security.
Copy .env.example to .env and fill in your actual credentials.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# OpenAI Configuration (optional if using Groq)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Groq Configuration (free, fast alternative)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate: Need at least one LLM API key
if not OPENAI_API_KEY and not GROQ_API_KEY:
    raise ValueError("Either OPENAI_API_KEY or GROQ_API_KEY must be set in environment variables")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vietnam-travel")
# Embedding model dimensions:
# - all-mpnet-base-v2 (local): 768 dimensions (RECOMMENDED)
# - text-embedding-3-small (OpenAI): 1536 dimensions
# - text-embedding-3-large (OpenAI): 3072 dimensions
PINECONE_VECTOR_DIM = int(os.getenv("PINECONE_VECTOR_DIM", "768"))  # Default: 768 for all-mpnet-base-v2

# Pinecone Serverless Configuration
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")  # Options: aws, gcp, azure
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # Must match cloud provider format

