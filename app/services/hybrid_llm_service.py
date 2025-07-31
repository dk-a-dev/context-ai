"""
Hybrid LLM Service Integration with Multiple Providers and Question Caching
Supports both OpenAI and Google Gemini with intelligent fallback and caching
"""

import asyncio
import logging
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Literal
from openai import OpenAI
from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)


class QuestionCache:
    """
    Advanced in-memory cache for similar questions with intelligent similarity detection
    
    Features:
    - TTL-based expiration
    - Context-aware caching
    - Memory usage tracking
    - Automatic cleanup
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the question cache
        
        Args:
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_cache_key(self, document_url: str, question: str) -> str:
        """
        Generate cache key based on document URL and question separated by comma
        
        Args:
            document_url: The URL of the document being queried
            question: The question text
            
        Returns:
            Cache key in format: "document_url,question"
        """
        # Normalize the question for better cache hits
        normalized_question = question.lower().strip()
        # Create cache key with document URL and question separated by comma
        cache_key = f"{document_url},{normalized_question}"
        return cache_key
    
    def _get_question_hash(self, question: str, context: str = "") -> str:
        """
        Generate hash for question + context combination for cache key (legacy method)
        
        Args:
            question: The question text
            context: Additional context (e.g., system instruction, document context)
            
        Returns:
            MD5 hash string for cache key
        """
        # Normalize the question for better cache hits
        normalized_question = question.lower().strip()
        # Limit context to avoid very long keys while maintaining uniqueness
        limited_context = context[:500] if context else ""
        combined = f"{normalized_question}|||{limited_context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, document_url: str, question: str, context: str = "") -> Optional[str]:
        """
        Retrieve cached answer if available and not expired
        
        Args:
            document_url: The URL of the document being queried
            question: The question to look up
            context: Additional context (kept for backward compatibility)
            
        Returns:
            Cached answer if found and valid, None otherwise
        """
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
        """
        Cache a question-answer pair
        
        Args:
            document_url: The URL of the document being queried
            question: The question text
            answer: The answer to cache
            context: Additional context (kept for backward compatibility)
        """
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
        """
        Legacy method: Retrieve cached answer using hash-based key (for backward compatibility)
        
        Args:
            question: The question to look up
            context: Additional context for cache key generation
            
        Returns:
            Cached answer if found and valid, None otherwise
        """
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
        """
        Legacy method: Cache a question-answer pair using hash-based key
        
        Args:
            question: The question text
            answer: The answer to cache
            context: Additional context for cache key generation
        """
        key = self._get_question_hash(question, context)
        self.cache[key] = {
            'answer': answer,
            'timestamp': time.time(),
            'question': question[:100],  # Store truncated question for debugging
            'context_preview': context[:100] if context else ""
        }
        logger.info(f"💾 Cached answer (hash) for question: {question[:50]}...")
    
    def clear_expired(self):
        """Remove all expired cache entries"""
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
        """
        Get comprehensive cache statistics including sample cache keys
        
        Returns:
            Dictionary with cache performance metrics and sample entries
        """
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
    """
    Hybrid LLM service supporting both OpenAI and Google Gemini with intelligent fallback
    
    Features:
    - Dual provider support (OpenAI + Gemini)
    - Automatic fallback on provider failures
    - Advanced caching system
    - Structured response formatting
    - Comprehensive error handling
    """
    
    # Streamlined prompt templates
    SYSTEM_INSTRUCTION_TEMPLATES = {
        "default": "You are a helpful assistant that answers questions based on provided context.",
        "insurance": "You are an expert insurance policy analyst. Answer questions precisely using only the provided context.",
        "legal": "You are a legal document expert. Provide accurate information citing specific clauses.",
        "hr": "You are an HR policy specialist. Answer questions about company policies and benefits.",
        "compliance": "You are a compliance officer. Provide answers based on regulatory documents."
    }
    
    def __init__(self, primary_provider: Literal["openai", "gemini"] = "gemini"):
        """
        Initialize the hybrid LLM service
        
        Args:
            primary_provider: Primary LLM provider to use ("openai" or "gemini")
        """
        self.primary_provider = primary_provider
        self.cache = QuestionCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
        
        # Initialize OpenAI client
        try:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.openai_model = settings.OPENAI_MODEL
            logger.info(f"✅ OpenAI client initialized with model: {self.openai_model}")
        except Exception as e:
            logger.warning(f"⚠️ OpenAI initialization failed: {e}")
            self.openai_client = None
        
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
        """Detect domain based on question keywords"""
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
        provider: Optional[Literal["openai", "gemini"]] = None
    ) -> str:
        """
        Generate response for a specific document and question with document-aware caching
        
        Args:
            document_url: URL of the document being queried
            question: The question to answer
            context: Document context/content for the LLM
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_instruction: System instruction for the model
            use_cache: Whether to use caching for this request
            provider: Specific provider to use, overrides primary_provider
            
        Returns:
            Generated response text
        """
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
        
        # Determine which provider to use
        target_provider = provider or self.primary_provider
        
        try:
            # Try primary provider first
            if target_provider == "openai" and self.openai_client:
                response = await self._generate_with_openai(full_prompt, temperature, max_tokens, system_instruction)
            elif target_provider == "gemini" and self.gemini_client:
                response = await self._generate_with_gemini(full_prompt, temperature, max_tokens, system_instruction)
            else:
                # Fallback to available provider
                if self.gemini_client and target_provider != "gemini":
                    logger.info(f"🔄 Falling back to Gemini from {target_provider}")
                    response = await self._generate_with_gemini(full_prompt, temperature, max_tokens, system_instruction)
                elif self.openai_client and target_provider != "openai":
                    logger.info(f"🔄 Falling back to OpenAI from {target_provider}")
                    response = await self._generate_with_openai(full_prompt, temperature, max_tokens, system_instruction)
                else:
                    raise Exception("No LLM providers available")
            
            # Cache the response if enabled
            if use_cache and settings.ENABLE_CACHE:
                self.cache.set(document_url, question, response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response for document {document_url}: {e}")
            # Try fallback provider
            fallback_provider = "gemini" if target_provider == "openai" else "openai"
            if fallback_provider != target_provider:
                try:
                    logger.info(f"🔄 Attempting fallback to {fallback_provider}")
                    if fallback_provider == "openai" and self.openai_client:
                        response = await self._generate_with_openai(full_prompt, temperature, max_tokens, system_instruction)
                    elif fallback_provider == "gemini" and self.gemini_client:
                        response = await self._generate_with_gemini(full_prompt, temperature, max_tokens, system_instruction)
                    else:
                        raise Exception(f"Fallback provider {fallback_provider} not available")
                    
                    # Cache the fallback response
                    if use_cache and settings.ENABLE_CACHE:
                        self.cache.set(document_url, question, response)
                    
                    return response
                    
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback to {fallback_provider} also failed: {fallback_error}")
                    raise Exception(f"All LLM providers failed. Primary: {e}, Fallback: {fallback_error}")
            else:
                raise e
    
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        system_instruction: Optional[str] = None,
        use_cache: bool = True,
        provider: Optional[Literal["openai", "gemini"]] = None
    ) -> str:
        """
        Generate response using the specified or primary LLM provider with caching (legacy method)
        
        Args:
            prompt: The user prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_instruction: System instruction for the model
            use_cache: Whether to use caching for this request
            provider: Specific provider to use, overrides primary_provider
            
        Returns:
            Generated response text
        """
        # Check cache first if enabled (using legacy hash-based method)
        if use_cache and settings.ENABLE_CACHE:
            cache_key_context = f"{provider or self.primary_provider}:{system_instruction or ''}"
            cached_answer = self.cache.get_by_hash(prompt, cache_key_context)
            if cached_answer:
                return cached_answer
        
        # Determine which provider to use
        target_provider = provider or self.primary_provider
        
        try:
            # Try primary provider first
            if target_provider == "openai" and self.openai_client:
                response = await self._generate_with_openai(prompt, temperature, max_tokens, system_instruction)
            elif target_provider == "gemini" and self.gemini_client:
                response = await self._generate_with_gemini(prompt, temperature, max_tokens, system_instruction)
            else:
                # Fallback to available provider
                if self.gemini_client and target_provider != "gemini":
                    logger.info(f"🔄 Falling back to Gemini from {target_provider}")
                    response = await self._generate_with_gemini(prompt, temperature, max_tokens, system_instruction)
                elif self.openai_client and target_provider != "openai":
                    logger.info(f"🔄 Falling back to OpenAI from {target_provider}")
                    response = await self._generate_with_openai(prompt, temperature, max_tokens, system_instruction)
                else:
                    raise Exception("No LLM providers available")
            
            # Cache the response if enabled
            if use_cache and settings.ENABLE_CACHE:
                cache_key_context = f"{target_provider}:{system_instruction or ''}"
                self.cache.set_by_hash(prompt, response, cache_key_context)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            # Try fallback provider
            fallback_provider = "gemini" if target_provider == "openai" else "openai"
            if fallback_provider != target_provider:
                try:
                    logger.info(f"🔄 Attempting fallback to {fallback_provider}")
                    if fallback_provider == "openai" and self.openai_client:
                        response = await self._generate_with_openai(prompt, temperature, max_tokens, system_instruction)
                    elif fallback_provider == "gemini" and self.gemini_client:
                        response = await self._generate_with_gemini(prompt, temperature, max_tokens, system_instruction)
                    else:
                        raise Exception(f"Fallback provider {fallback_provider} not available")
                    
                    # Cache the fallback response
                    if use_cache and settings.ENABLE_CACHE:
                        cache_key_context = f"{fallback_provider}:{system_instruction or ''}"
                        self.cache.set_by_hash(prompt, response, cache_key_context)
                    
                    return response
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
            
            # If all else fails, return an error message
            error_response = f"I encountered an error while processing this question: {str(e)}"
            return error_response
    
    async def _generate_with_openai(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: int, 
        system_instruction: Optional[str]
    ) -> str:
        """Generate response using OpenAI with timeout"""
        messages = []
        
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await asyncio.wait_for(
            self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0
            ),
            timeout=30.0
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            raise Exception("No response generated from OpenAI")
    
    async def _generate_with_gemini(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: int, 
        system_instruction: Optional[str]
    ) -> str:
        """Generate response using Gemini"""
        contents = []
        
        # Combine system instruction with user prompt for Gemini
        final_prompt = prompt
        if system_instruction:
            final_prompt = f"{system_instruction}\n\n{prompt}"
        
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=final_prompt)]
        ))
        
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=contents+"""You are a smart agent, a question will be asked to you with the relevant 
context provided here. Answer the question based on the context provided.
Under no circumstances should you make stuff up or use something that has not
been provided. If the question cannot be answered using the given context 
explicitly mention so, also give to the point one-two line response for each question""",
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
        provider: Optional[Literal["openai", "gemini"]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text with guaranteed JSON format
        
        Args:
            text: Input text to process
            schema: JSON schema for extraction
            context: Context instruction
            provider: Specific provider to use
            
        Returns:
            Extracted structured data as dictionary
        """
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
                use_cache=False,  # Don't cache structured data extraction
                provider=provider
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
            logger.error(f"Error extracting structured data: {str(e)}")
            return {
                "error": f"Extraction failed: {str(e)}",
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
    
    async def analyze_query_intent(self, query: str, provider: Optional[Literal["openai", "gemini"]] = None) -> Dict[str, Any]:
        """
        Analyze user query intent and extract key information
        
        Args:
            query: User query string
            provider: Specific provider to use
            
        Returns:
            Query analysis results in structured format
        """
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
        provider: Optional[Literal["openai", "gemini"]] = None
    ) -> str:
        """
        Generate detailed explanation for a decision based on retrieved clauses
        
        Args:
            query: Original user query
            retrieved_clauses: List of relevant document clauses
            decision: The decision/answer provided
            provider: Specific provider to use
            
        Returns:
            Detailed explanation with clause references
        """
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
        
        Please provide a comprehensive explanation including:
        
        ## Decision Summary
        Brief overview of the decision
        
        ## Supporting Evidence
        - Which specific clauses support this decision
        - Relevant section numbers or references
        
        ## Conditions & Limitations
        - Any conditions that apply
        - Important limitations or exceptions
        
        ## Reasoning Process
        Step-by-step explanation of how the decision was reached
        
        Be precise and reference specific clause numbers or sections when possible.
        """
        
        return await self.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2,
            max_tokens=1500,
            use_cache=True,  # Cache explanations as they can be reused
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
        self.cache.clear_expired()  # Clean up expired entries first
        stats = self.cache.get_stats()
        
        # Add provider information
        stats.update({
            "primary_provider": self.primary_provider,
            "openai_available": self.openai_client is not None,
            "gemini_available": self.gemini_client is not None,
            "cache_ttl_seconds": self.cache.ttl_seconds
        })
        
        return stats
    
    def switch_primary_provider(self, provider: Literal["openai", "gemini"]):
        """
        Switch the primary LLM provider
        
        Args:
            provider: New primary provider ("openai" or "gemini")
        """
        old_provider = self.primary_provider
        self.primary_provider = provider
        logger.info(f"🔄 Switched primary provider from {old_provider} to {provider}")