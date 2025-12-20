"""
LLM client abstraction for OpenAI, Anthropic, and Google Gemini.
Provides unified interface with retry logic, token tracking, and cost estimation.
"""

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import google.generativeai as genai
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMMessage(BaseModel):
    """Message format for LLM conversations."""
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion from messages."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream completion from messages."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with retry logic."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.default_model = settings.openai_model_primary
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on token usage."""
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o"])
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    
    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        request_params = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            request_params["response_format"] = {"type": "json_object"}
        
        logger.debug("OpenAI request", model=model, message_count=len(messages))
        
        response = await asyncio.wait_for(
            self.client.chat.completions.create(**request_params),
            timeout=settings.llm_timeout,
        )
        
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=LLMProvider.OPENAI,
            tokens_used=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=self._estimate_cost(model, prompt_tokens, completion_tokens),
        )
    
    async def stream_generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream completion using OpenAI API."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    """Anthropic API client with retry logic."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }
    
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.default_model = settings.anthropic_model_primary
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on token usage."""
        pricing = self.PRICING.get(model, self.PRICING["claude-3-5-sonnet-20241022"])
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    
    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        # Separate system message from conversation
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        logger.debug("Anthropic request", model=model, message_count=len(messages))
        
        request_params = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_message:
            request_params["system"] = system_message
        
        response = await asyncio.wait_for(
            self.client.messages.create(**request_params),
            timeout=settings.llm_timeout,
        )
        
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            provider=LLMProvider.ANTHROPIC,
            tokens_used=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=self._estimate_cost(model, prompt_tokens, completion_tokens),
        )
    
    async def stream_generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream completion using Anthropic API."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        request_params = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_message:
            request_params["system"] = system_message
        
        async with self.client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text


class GeminiClient(BaseLLMClient):
    """Google Gemini API client with retry logic."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gemini-1.5-pro-latest": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash-latest": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.default_model = settings.google_model_primary
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on token usage."""
        pricing = self.PRICING.get(model, self.PRICING["gemini-1.5-pro"])
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    
    def _convert_messages(self, messages: list[LLMMessage]) -> tuple[str, list[dict]]:
        """Convert messages to Gemini format, extracting system instruction."""
        system_instruction = ""
        gemini_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})
        
        return system_instruction, gemini_messages
    
    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using Gemini API."""
        model_name = model or self.default_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        system_instruction, gemini_messages = self._convert_messages(messages)
        
        logger.debug("Gemini request", model=model_name, message_count=len(messages))
        
        # Configure generation settings
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if json_mode:
            generation_config.response_mime_type = "application/json"
        
        # Create model with system instruction
        model_instance = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction if system_instruction else None,
            generation_config=generation_config,
        )
        
        # Run in executor since genai is sync
        loop = asyncio.get_event_loop()
        
        def _generate():
            chat = model_instance.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
            last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            return chat.send_message(last_message)
        
        response = await asyncio.wait_for(
            loop.run_in_executor(None, _generate),
            timeout=settings.llm_timeout,
        )
        
        # Extract token counts (Gemini provides these in usage_metadata)
        prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        return LLMResponse(
            content=response.text,
            model=model_name,
            provider=LLMProvider.GOOGLE,
            tokens_used=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=self._estimate_cost(model_name, prompt_tokens, completion_tokens),
        )
    
    async def stream_generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream completion using Gemini API."""
        model_name = model or self.default_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        system_instruction, gemini_messages = self._convert_messages(messages)
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        model_instance = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction if system_instruction else None,
            generation_config=generation_config,
        )
        
        chat = model_instance.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        
        # Stream response
        response = chat.send_message(last_message, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


class LLMClient:
    """
    Unified LLM client that routes to appropriate provider.
    Provides simple interface for the rest of the application.
    """
    
    def __init__(self, default_provider: Optional[LLMProvider] = None):
        # Lazy initialization - only initialize clients with valid API keys
        self._openai: Optional[OpenAIClient] = None
        self._anthropic: Optional[AnthropicClient] = None
        self._gemini: Optional[GeminiClient] = None
        
        # Set default provider from config or fallback
        if default_provider:
            self.default_provider = default_provider
        else:
            provider_map = {
                "google": LLMProvider.GOOGLE,
                "openai": LLMProvider.OPENAI,
                "anthropic": LLMProvider.ANTHROPIC,
            }
            self.default_provider = provider_map.get(
                settings.default_llm_provider.lower(), 
                LLMProvider.GOOGLE
            )
    
    @property
    def openai(self) -> OpenAIClient:
        """Lazy load OpenAI client."""
        if self._openai is None:
            self._openai = OpenAIClient()
        return self._openai
    
    @property
    def anthropic(self) -> AnthropicClient:
        """Lazy load Anthropic client."""
        if self._anthropic is None:
            self._anthropic = AnthropicClient()
        return self._anthropic
    
    @property
    def gemini(self) -> GeminiClient:
        """Lazy load Gemini client."""
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini
    
    def _get_client(self, provider: Optional[LLMProvider] = None) -> BaseLLMClient:
        """Get client for specified provider."""
        provider = provider or self.default_provider
        if provider == LLMProvider.OPENAI:
            return self.openai
        elif provider == LLMProvider.ANTHROPIC:
            return self.anthropic
        else:  # GOOGLE / Gemini is default
            return self.gemini
    
    async def generate(
        self,
        messages: list[LLMMessage],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate completion using specified provider."""
        client = self._get_client(provider)
        return await client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    
    async def stream_generate(
        self,
        messages: list[LLMMessage],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream completion using specified provider."""
        client = self._get_client(provider)
        async for chunk in client.stream_generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk
    
    async def generate_fast(
        self,
        messages: list[LLMMessage],
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate using fast model for simple tasks."""
        # Use the default provider's fast model
        if self.default_provider == LLMProvider.GOOGLE:
            return await self.gemini.generate(
                messages=messages,
                model=settings.google_model_fast,
                json_mode=json_mode,
            )
        elif self.default_provider == LLMProvider.ANTHROPIC:
            return await self.anthropic.generate(
                messages=messages,
                model="claude-3-haiku-20240307",
                json_mode=json_mode,
            )
        else:
            return await self.openai.generate(
                messages=messages,
                model=settings.openai_model_fast,
                json_mode=json_mode,
            )
    
    async def generate_smart(
        self,
        messages: list[LLMMessage],
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate using smart model (primary) for complex tasks."""
        return await self.generate(
            messages=messages,
            json_mode=json_mode,
        )


# Global LLM client instance
llm_client = LLMClient()

