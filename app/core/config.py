"""
Application Configuration with Google Gemini Integration
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Google Gemini API Configuration
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str]
    PINECONE_ENVIRONMENT: Optional[str]
    PINECONE_INDEX_NAME: str
    
    # Database Configuration
    DATABASE_URL: str
    
    # API Configuration
    API_VERSION: str
    API_TOKEN: str
    
    # Application Settings
    DEBUG: bool
    HOST: str
    PORT: int
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    
    # Document Processing Settings
    MAX_FILE_SIZE_MB: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    
    # Cache Settings
    ENABLE_CACHE: bool
    CACHE_TTL_SECONDS: int
    
    # Performance Optimization Settings
    OPTIMIZE_FOR_SPEED: bool = True
    MAX_CONCURRENT_DOCS: int = 5
    MAX_CONCURRENT_QUESTIONS: int = 10
    EMBEDDING_BATCH_SIZE: int = 32
    FAISS_NPROBE: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()