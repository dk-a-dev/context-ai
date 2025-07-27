"""
API v1 Router
"""

from fastapi import APIRouter
from app.api.v1.endpoints import router as endpoints_router

api_router = APIRouter()
api_router.include_router(endpoints_router, tags=["query-processing"])
