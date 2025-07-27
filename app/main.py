"""
Context-IQ Application Entry Point
Main FastAPI application with Google Gemini integration
"""

# SSL certificate fix for macOS
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Context-IQ Application...")
    yield
    logger.info("Shutting down Context-IQ Application...")


# Create FastAPI app
app = FastAPI(
    title="Context-IQ: Intelligent Query-Retrieval System",
    description="LLM-Powered document processing and query retrieval system with Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials


# Include API router
app.include_router(
    api_router,
    prefix=f"/api/{settings.API_VERSION}",
    dependencies=[Depends(verify_token)]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Context-IQ: Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "context-iq"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
