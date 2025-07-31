"""
Main Query Processing Service
Orchestrates document processing, vector search, and LLM analysis
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import time

from app.services.hybrid_llm_service import HybridLLMService
from app.services.document_processor import DocumentProcessor
from app.services.vector_search import VectorSearchService
from app.core.config import settings

logger = logging.getLogger(__name__)


class QueryProcessingService:
    """Main service that orchestrates the entire query processing pipeline"""
    
    def __init__(self, use_pinecone: bool = False):
        """
        Initialize the query processing service
        
        Args:
            use_pinecone: Whether to use Pinecone for vector search
        """
        self.llm_service = HybridLLMService(primary_provider="gemini")
        self.document_processor = DocumentProcessor()
        self.vector_search = VectorSearchService(use_pinecone=use_pinecone)
        
        # Cache for processed documents
        self.document_cache = {}
        
        logger.info("QueryProcessingService initialized")
    
    async def process_documents_and_queries(
        self,
        document_urls: List[str],
        questions: List[str]
    ) -> Dict[str, Any]:
        """
        Process documents and answer questions - main API endpoint logic
        
        Args:
            document_urls: List of document URLs to process
            questions: List of questions to answer
            
        Returns:
            Dictionary with answers and processing metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Process documents in parallel
            logger.info(f"Processing {len(document_urls)} documents")
            processed_docs = await self._process_documents_parallel(document_urls)
            
            if not processed_docs:
                return {
                    "error": "Failed to process any documents",
                    "answers": [],
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: Add documents to vector index
            logger.info("Adding documents to vector index")
            index_success = self.vector_search.add_documents(processed_docs)
            
            if not index_success:
                logger.warning("Failed to add documents to vector index")
            
            # Step 3: Process questions in parallel
            logger.info(f"Processing {len(questions)} questions")
            answers = await self._process_questions_parallel(questions, processed_docs)
            
            # Step 4: Prepare response
            processing_time = time.time() - start_time
            
            response = {
                "answers": answers,
                "processing_time": processing_time,
                "documents_processed": len(processed_docs),
                "questions_processed": len(questions),
                "vector_index_stats": self.vector_search.get_stats()
            }
            
            logger.info(f"Query processing completed in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            import traceback
            logger.error(f"Error in query processing: {str(e)}\n{traceback.format_exc()}")
            return {
                "error": f"{str(e)}\n{traceback.format_exc()}",
                "answers": [],
                "processing_time": time.time() - start_time
            }
    
    async def _process_documents_parallel(self, document_urls: List[str]) -> List[Dict[str, Any]]:
        """Process documents in parallel with semaphore"""
        processed_docs = []
        
        # Check cache first
        tasks = []
        for url in document_urls:
            if url in self.document_cache:
                logger.info(f"Using cached document: {url}")
                processed_docs.append(self.document_cache[url])
            else:
                tasks.append(url)
        
        if tasks:
            semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

            async def process_with_semaphore(url):
                async with semaphore:
                    try:
                        return await self._process_single_document(url)
                    except Exception as e:
                        logger.error(f"Error processing document {url}: {str(e)}")
                        return None

            task_coroutines = [process_with_semaphore(url) for url in tasks]
            results = await asyncio.gather(*task_coroutines)

            for i, result in enumerate(results):
                if result is None:
                    logger.error(f"Skipping document due to processing error: {tasks[i]}")
                else:
                    processed_docs.append(result)
                    # Cache the processed document
                    self.document_cache[result["url"]] = result
        
        return processed_docs
    
    async def _process_questions_parallel(self, questions: List[str], processed_docs: List[Dict[str, Any]]):
        """Process questions in parallel with semaphore"""
        semaphore = asyncio.Semaphore(10)  # Higher concurrency for questions
        
        async def process_with_semaphore(question):
            async with semaphore:
                try:
                    return await self._process_single_question(question, processed_docs)
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    return f"Error processing question: {str(e)}"
        
        tasks = [process_with_semaphore(q) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_document(self, document_url: str) -> Dict[str, Any]:
        """Process a single document"""
        try:
            logger.info(f"Processing document: {document_url}")
            processed_doc = await self.document_processor.process_document(document_url)
            logger.info(f"Successfully processed document: {document_url}")
            return processed_doc
        except Exception as e:
            import traceback
            logger.error(f"Error processing document {document_url}: {str(e)}\n{traceback.format_exc()}")
            raise
    
    async def _process_single_question(
        self,
        question: str,
        processed_docs: List[Dict[str, Any]]
    ) -> str:
        """Process a single question using the full pipeline"""
        
        try:
            # Step 1: Analyze query intent
            query_analysis = await self.llm_service.analyze_query_intent(question)
            logger.debug(f"Query analysis: {query_analysis}")
            
            # Step 2: Perform vector search to find relevant chunks
            search_results = self.vector_search.search(
                query=question,
                top_k=10,
                score_threshold=0.3
            )
            
            if not search_results:
                # Fallback: use simple text matching if vector search fails
                search_results = self._fallback_text_search(question, processed_docs)
            
            # Step 3: Extract relevant clauses from search results (limit to top 3 for optimization)
            relevant_clauses = [result["text"] for result in search_results[:3]]
            
            if not relevant_clauses:
                return "I couldn't find relevant information in the provided documents to answer this question."
            
            # Step 4: Generate answer using document-aware caching
            # Use the first processed document URL for caching (or combine multiple URLs)
            document_url = processed_docs[0]["url"] if processed_docs else "unknown"
            if len(processed_docs) > 1:
                # For multiple documents, create a combined cache key
                document_urls = [doc["url"] for doc in processed_docs]
                document_url = "|".join(sorted(document_urls))
            
            answer = await self._generate_answer_with_context(
                question=question,
                document_url=document_url,
                relevant_clauses=relevant_clauses,
                query_analysis=query_analysis
            )
            
            return answer
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing question '{question}': {str(e)}\n{traceback.format_exc()}")
            return f"I encountered an error while processing this question: {str(e)}\n{traceback.format_exc()}"
    
    def _fallback_text_search(
        self,
        question: str,
        processed_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback text search when vector search fails"""
        
        # Simple keyword matching
        question_lower = question.lower()
        keywords = question_lower.split()
        
        results = []
        
        for doc in processed_docs:
            for chunk in doc.get("chunks", []):
                text_lower = chunk["text"].lower()
                score = 0
                
                # Count keyword matches
                for keyword in keywords:
                    if len(keyword) > 2 and keyword in text_lower:
                        score += 1
                
                if score > 0:
                    results.append({
                        "text": chunk["text"],
                        "score": score / len(keywords),
                        "document_url": doc["url"],
                        "chunk_id": chunk["id"]
                    })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Fallback search found {len(results)} relevant chunks")
        return results[:10]
    
    async def _generate_answer_with_context(
        self,
        question: str,
        document_url: str,
        relevant_clauses: List[str],
        query_analysis: Dict[str, Any]
    ) -> str:
        """Generate answer using document-aware caching with context"""
        
        try:
            # Prepare optimized context from relevant clauses (limit to most relevant parts)
            context = "\n".join([f"- {clause}" for clause in relevant_clauses])
            
            # Determine answer type based on query analysis
            intent = query_analysis.get("intent", "search")
            question_type = query_analysis.get("question_type", "general")
            
            # Create a tailored prompt based on the question type
            if question_type in ["coverage", "eligibility"]:
                system_instruction = """You are an expert insurance policy analyst. Your job is to provide accurate, detailed answers about policy coverage and eligibility based on the provided policy documents. Always cite specific clauses and conditions."""
            elif question_type in ["conditions", "requirements"]:
                system_instruction = """You are a legal document expert. Provide precise information about conditions, requirements, and procedures based on the provided clauses. Be specific about any limitations or exceptions."""
            elif question_type == "limits":
                system_instruction = """You are a policy limits specialist. Focus on monetary limits, time limits, and coverage restrictions. Provide exact figures and conditions when available."""
            else:
                system_instruction = """You are a document analysis expert. Provide accurate, helpful answers based on the provided text. Be precise and cite relevant sections."""
            
            full_context = f"""
Based on the following relevant clauses from the policy documents, please answer this question:

Question: {question}

Relevant clauses:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. References specific clauses or sections
3. Mentions any conditions, limitations, or exceptions
4. Is clear and easy to understand

If the question cannot be fully answered from the provided clauses, please indicate what information is missing.
"""
            
            # Use document-aware caching
            answer = await self.llm_service.generate_response_with_document(
                document_url=document_url,
                question=question,
                context=full_context,
                system_instruction=system_instruction,
                temperature=0.1,
                max_tokens=1500
            )
            
            return answer
            
        except Exception as e:
            import traceback
            logger.error(f"Error generating answer: {str(e)}\n{traceback.format_exc()}")
            return f"I encountered an error while generating the answer: {str(e)}\n{traceback.format_exc()}"
    
    async def get_explanation(
        self,
        question: str,
        answer: str,
        relevant_clauses: List[str]
    ) -> str:
        """Generate detailed explanation for a given answer"""
        
        try:
            explanation = await self.llm_service.generate_explanation(
                query=question,
                retrieved_clauses=relevant_clauses,
                decision=answer
            )
            return explanation
            
        except Exception as e:
            import traceback
            logger.error(f"Error generating explanation: {str(e)}\n{traceback.format_exc()}")
            return f"Unable to generate explanation: {str(e)}\n{traceback.format_exc()}"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            "vector_search": self.vector_search.get_stats(),
            "document_cache_size": len(self.document_cache),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.GEMINI_MODEL,
            "config": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "embedding_dimension": settings.EMBEDDING_DIMENSION
            }
        }