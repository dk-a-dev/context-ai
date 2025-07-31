"""
Hybrid LLM Service Integration with Multiple Providers and Question Caching
Supports Google Gemini with intelligent fallback and caching
"""

import asyncio
import logging
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Literal
from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)


class QuestionCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_cache_key(self, document_url: str, question: str) -> str:
        # Normalize the question for better cache hits
        normalized_question = question.lower().strip()
        # Create cache key with document URL and question separated by comma
        cache_key = f"{document_url},{normalized_question}"
        return cache_key
    
    def _get_question_hash(self, question: str, context: str = "") -> str:
        # Normalize the question for better cache hits
        normalized_question = question.lower().strip()
        # Limit context to avoid very long keys while maintaining uniqueness
        limited_context = context[:500] if context else ""
        combined = f"{normalized_question}|||{limited_context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, document_url: str, question: str, context: str = "") -> Optional[str]:
        key = self._get_cache_key(document_url, question)
        if key in self.cache:
            cached_item = self.cache[key]
            if time.time() - cached_item['timestamp'] < self.ttl_seconds:
                self.hit_count += 1
                logger.info(f"✅ Cache HIT for document: {document_url}, question: {question[:50]}...")
                return cached_item['answer']
            else:
                # Remove expired item
                del self.cache[key]
                logger.debug(f"🗑️ Removed expired cache entry for: {document_url}, {question[:50]}...")
        
        self.miss_count += 1
        logger.info(f"❌ Cache MISS for document: {document_url}, question: {question[:50]}...")
        return None
    
    def set(self, document_url: str, question: str, answer: str, context: str = ""):
        key = self._get_cache_key(document_url, question)
        self.cache[key] = {
            'answer': answer,
            'timestamp': time.time(),
            'document_url': document_url,
            'question': question[:100],  # Store truncated question for debugging
            'cache_key': key,  # Store the actual cache key for debugging
            'context_preview': context[:100] if context else ""
        }
        logger.info(f"💾 Cached answer for document: {document_url}, question: {question[:50]}...")
    
    def get_by_hash(self, question: str, context: str = "") -> Optional[str]:
        key = self._get_question_hash(question, context)
        if key in self.cache:
            cached_item = self.cache[key]
            if time.time() - cached_item['timestamp'] < self.ttl_seconds:
                self.hit_count += 1
                logger.info(f"✅ Cache HIT (hash) for question: {question[:50]}...")
                return cached_item['answer']
            else:
                # Remove expired item
                del self.cache[key]
                logger.debug(f"🗑️ Removed expired cache entry (hash) for: {question[:50]}...")
        
        self.miss_count += 1
        logger.info(f"❌ Cache MISS (hash) for question: {question[:50]}...")
        return None
    
    def set_by_hash(self, question: str, answer: str, context: str = ""):
        key = self._get_question_hash(question, context)
        self.cache[key] = {
            'answer': answer,
            'timestamp': time.time(),
            'question': question[:100],  # Store truncated question for debugging
            'context_preview': context[:100] if context else ""
        }
        logger.info(f"💾 Cached answer (hash) for question: {question[:50]}...")
    
    def clear_expired(self):
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['timestamp'] > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            logger.info(f"🧹 Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        self.clear_expired()  # Clean up expired entries first
        
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        # Get memory usage estimate
        import sys
        cache_size_bytes = sys.getsizeof(self.cache)
        for key, value in self.cache.items():
            cache_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
            cache_size_bytes += sum(sys.getsizeof(v) for v in value.values())
        
        # Sample cache entries (for debugging)
        sample_entries = []
        for key, value in list(self.cache.items())[:3]:  # Show first 3 entries
            entry_info = {
                "cache_key": key,
                "is_document_aware": "," in key and not key.startswith("hash:"),
                "question_preview": value.get("question", "")[:50] + "..." if value.get("question", "") else "",
                "document_url": value.get("document_url", ""),
                "timestamp": value.get("timestamp", 0),
                "age_seconds": time.time() - value.get("timestamp", 0)
            }
            sample_entries.append(entry_info)
        
        return {
            "cache_enabled": True,
            "cache_ttl_seconds": self.ttl_seconds,
            "total_entries": len(self.cache),
            "cache_size_mb": cache_size_bytes / (1024 * 1024),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percentage": round(hit_rate, 2),
            "total_requests": total_requests,
            "sample_entries": sample_entries
        }


class HybridLLMService:
    
    # Streamlined prompt templates
    SYSTEM_INSTRUCTION_TEMPLATES = {
        "default": "You are a helpful assistant that answers questions based on provided context.",
        "insurance": "You are an expert insurance policy analyst. Answer questions precisely using only the provided context.",
        "legal": "You are a legal document expert. Provide accurate information citing specific clauses.",
        "hr": "You are an HR policy specialist. Answer questions about company policies and benefits.",
        "compliance": "You are a compliance officer. Provide answers based on regulatory documents."
    }
    
    def __init__(self, primary_provider: Literal["gemini"] = "gemini"):
        self.primary_provider = primary_provider
        self.cache = QuestionCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
        
        # Initialize Gemini client
        try:
            self.gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            self.gemini_model = settings.GEMINI_MODEL
            logger.info(f"✅ Gemini client initialized with model: {self.gemini_model}")
        except Exception as e:
            logger.warning(f"⚠️ Gemini initialization failed: {e}")
            self.gemini_client = None
        
        logger.info(f"🚀 Hybrid LLM Service initialized with primary provider: {primary_provider}")
    
    def detect_domain(self, question: str) -> str:
        question_lower = question.lower()
        if any(kw in question_lower for kw in ["policy", "cover", "claim", "premium"]):
            return "insurance"
        if any(kw in question_lower for kw in ["clause", "agreement", "contract", "legal"]):
            return "legal"
        if any(kw in question_lower for kw in ["employee", "benefit", "hr", "leave"]):
            return "hr"
        if any(kw in question_lower for kw in ["compliance", "regulation", "standard"]):
            return "compliance"
        return "default"
    
    async def generate_response_with_document(
        self,
        document_url: str,
        question: str,
        context: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        system_instruction: Optional[str] = None,
        use_cache: bool = True,
        provider: Optional[Literal["gemini"]] = None
    ) -> str:
        # Check cache first if enabled
        if use_cache and settings.ENABLE_CACHE:
            cached_answer = self.cache.get(document_url, question)
            if cached_answer:
                return cached_answer
        
        # Domain-specific system instructions
        if not system_instruction:
            domain = self.detect_domain(question)
            system_instruction = self.SYSTEM_INSTRUCTION_TEMPLATES.get(domain, self.SYSTEM_INSTRUCTION_TEMPLATES["default"])
        
        # Streamlined prompt
        full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            if self.gemini_client:
                response = await self._generate_with_gemini(full_prompt, temperature, max_tokens, system_instruction)
            else:
                raise Exception("Gemini client not available")
            if use_cache and settings.ENABLE_CACHE:
                self.cache.set(document_url, question, response)
            return response
        except Exception as e:
            import traceback
            logger.error(f"❌ Error generating response for document {document_url}: {e}\n{traceback.format_exc()}")
            return "error getting response!!"

    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        system_instruction: Optional[str] = None,
        use_cache: bool = True,
        provider: Optional[Literal["gemini"]] = None
    ) -> str:
        # Check cache first if enabled (using legacy hash-based method)
        if use_cache and settings.ENABLE_CACHE:
            cache_key_context = f"gemini:{system_instruction or ''}"
            cached_answer = self.cache.get_by_hash(prompt, cache_key_context)
            if cached_answer:
                return cached_answer
        
        try:
            if self.gemini_client:
                response = await self._generate_with_gemini(prompt, temperature, max_tokens, system_instruction)
            else:
                raise Exception("Gemini client not available")
            if use_cache and settings.ENABLE_CACHE:
                cache_key_context = f"gemini:{system_instruction or ''}"
                self.cache.set_by_hash(prompt, response, cache_key_context)
            return response
        except Exception as e:
            import traceback
            logger.error(f"❌ Error generating response: {e}\n{traceback.format_exc()}")
            return "error getting response!!"

    async def _generate_with_gemini(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: int, 
        system_instruction: Optional[str]
    ) -> str:
        contents = []
        final_prompt = prompt
        # Add precision and max 2-line reply instruction
        extra_instruction = (
            "You must answer every question as precisely as possible. "
            "For each question or sub-question, reply in no more than 2 lines. "
            "If you cannot answer, say so directly and briefly."
        )
        if system_instruction:
            final_prompt = f"{system_instruction}\n\n{extra_instruction}\n\n{prompt}"
        else:
            final_prompt = f"{extra_instruction}\n\n{prompt}"
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=final_prompt)]
        ))
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1
            )
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            raise Exception("No response generated from Gemini")

    async def extract_structured_data(
        self,
        text: str,
        schema: Dict[str, Any],
        context: str = "Extract structured information from the following text:",
        provider: Optional[Literal["gemini"]] = None
    ) -> Dict[str, Any]:
        try:
            system_instruction = """You are a helpful assistant that extracts structured information from text. 
            CRITICAL: Always respond with valid JSON only, no additional text, markdown formatting, or explanations.
            The response must be parseable JSON that matches the provided schema."""
            
            prompt = f"""
            {context}
            
            Text to analyze:
            {text}
            
            Please extract information according to this JSON schema:
            {json.dumps(schema, indent=2)}
            
            IMPORTANT: Respond with valid JSON only, no additional text or formatting.
            """
            
            response = await self.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.0,
                max_tokens=2000,
                use_cache=False  # Don't cache structured data extraction
            )
            
            # Clean and parse JSON response
            cleaned_response = self._clean_json_response(response)
            
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                # Try to extract JSON from response using regex
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                # Return error response in expected format
                logger.error("Could not extract valid JSON from response")
                return {
                    "error": "Failed to parse structured response",
                    "raw_response": response[:500],
                    "schema": schema
                }
        except Exception as e:
            import traceback
            logger.error(f"Error extracting structured data: {str(e)}\n{traceback.format_exc()}")
            return {
                "error": "error getting response!!",
                "schema": schema
            }
    
    def _clean_json_response(self, response: str) -> str:
        """Clean response text to extract valid JSON"""
        # Remove common markdown formatting
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return response.strip()
    
    async def analyze_query_intent(self, query: str, provider: Optional[Literal["gemini"]] = None) -> Dict[str, Any]:
        schema = {
            "intent": "Primary intent category (search, comparison, eligibility, coverage, claims, etc.)",
            "entities": ["List of specific entities mentioned (procedures, conditions, benefits, etc.)"],
            "question_type": "Type of question (what, how, when, where, why, yes/no)",
            "key_terms": ["Important search terms for document retrieval"],
            "context_needed": ["Additional context that might be needed for accurate response"],
            "urgency": "Response urgency level (low, medium, high)",
            "complexity": "Question complexity level (simple, moderate, complex)"
        }
        
        return await self.extract_structured_data(
            text=query,
            schema=schema,
            context="Analyze this insurance/legal/HR query and extract the intent and key information:",
            provider=provider
        )
    
    async def generate_explanation(
        self,
        query: str,
        retrieved_clauses: List[str],
        decision: str,
        provider: Optional[Literal["gemini"]] = None
    ) -> str:
        clauses_text = "\n".join([f"- {clause}" for clause in retrieved_clauses])
        
        system_instruction = """You are an expert insurance/legal document analyst. 
        Provide clear, detailed explanations that reference specific clauses and sections. 
        Use structured formatting with headers and bullet points.
        Be precise and thorough in your analysis."""
        
        prompt = f"""
        User Query: {query}
        
        Decision/Answer: {decision}
        
        Relevant Document Clauses:
        {clauses_text}        
        Be precise.
        """
        
        return await self.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2,
            max_tokens=1500,
            use_cache=True, 
            provider=provider
        )
    
    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.cache.clear()
        self.cache.hit_count = 0
        self.cache.miss_count = 0
        logger.info("🧹 Cleared all cached responses and reset counters")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        self.cache.clear_expired() 
        stats = self.cache.get_stats()
        
        # Add provider information
        stats.update({
            "primary_provider": self.primary_provider,
            "gemini_available": self.gemini_client is not None,
            "cache_ttl_seconds": self.cache.ttl_seconds
        })
        
        return stats
    
    def switch_primary_provider(self, provider: Literal["gemini"]):
        old_provider = self.primary_provider
        self.primary_provider = provider
        logger.info(f"🔄 Switched primary provider from {old_provider} to {provider}")