"""
Unified LLM client using pydantic_ai for robust handling across providers.
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import random

# Load environment variables FIRST before any pydantic_ai imports
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
try:
    from pydantic_ai import Agent as PydanticAgent
except ImportError:
    PydanticAgent = None

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Logfire telemetry is opt-in: it is enabled only when BEAGLE_ENABLE_LOGFIRE=1
# and LOGFIRE_TOKEN are both set. It sends per-call latency / prompts / errors
# to your Logfire dashboard. Idempotent: only configures once per process.
_LOGFIRE_INSTRUMENTED = False
if (LOGFIRE_AVAILABLE and os.environ.get("BEAGLE_ENABLE_LOGFIRE") == "1"
        and os.environ.get("LOGFIRE_TOKEN") and not _LOGFIRE_INSTRUMENTED):
    try:
        logfire.configure(send_to_logfire=True, console=False)
        logfire.instrument_pydantic_ai()
        _LOGFIRE_INSTRUMENTED = True
        logger.info("Logfire instrumentation enabled (BEAGLE_ENABLE_LOGFIRE=1)")
    except Exception as e:
        logger.warning(f"Logfire setup failed; continuing without telemetry: {e}")

try:
    from mlx_lm import load as mlx_load
    from pydantic_ai.models.outlines import OutlinesModel
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_load = None
    OutlinesModel = None


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    MLX = "mlx"
    GATEWAY = "gateway"
    GROQ = "groq"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class RateLimiter:
    """Simple rate limiter to prevent excessive API calls."""

    def __init__(self, requests_per_minute: int = 15):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self.total_requests = 0

    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiter: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.total_requests += 1

        if self.total_requests % 10 == 0:
            logger.info(
                f"Rate limiter: {self.total_requests} total requests made"
            )


class LLMClient:
    """
    Unified LLM client using pydantic_ai for robust cross-provider support.

    Supports all providers via pydantic_ai format:
    - Gemini: "google-gla:gemini-2.5-flash"
    - OpenAI: "openai:gpt-4"
    - Anthropic: "anthropic:claude-sonnet-4-0"
    - Ollama: "ollama:llama3.1"
    - MLX: "mlx-community:model-name" or "mlx:mlx-community/model-name"

    Example:
        >>> client = LLMClient(model="google-gla:gemini-2.5-flash")
        >>> response = client.generate("What is the capital of France?")
        >>> print(response.content)
        
        >>> # Using MLX (Apple Silicon) - both formats work
        >>> client = LLMClient(model="mlx-community:TinyLlama-1.1B-Chat-v1.0-4bit")
        >>> # or
        >>> client = LLMClient(model="mlx:mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")
        >>> response = client.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        rate_limit_rpm: Optional[int] = None,
        session_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize LLM client with pydantic_ai.

        Args:
            model: Model in pydantic_ai format (e.g., "google-gla:gemini-2.5-flash")
                   For MLX: "mlx-community:model-name" or "mlx:mlx-community/model-name"
                   Examples:
                   - "mlx-community:gpt-oss-120b-MXFP4-Q8"
                   - "mlx:mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
            api_key: API key (falls back to environment variables)
            temperature: Temperature for sampling (0.0-1.0)
            max_tokens: Maximum tokens to generate
            rate_limit_rpm: Rate limit in requests per minute
            session_id: Session ID for trace grouping
            max_retries: Maximum number of retry attempts on failure (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0, uses exponential backoff)
            **kwargs: Additional provider-specific parameters
        """
        if PydanticAgent is None:
            raise ImportError(
                "pydantic_ai not installed. Install with: pip install pydantic-ai"
            )

        self.model_string = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.mlx_model_obj = None  # For MLX models
        self.session_id = session_id  # Store session ID for trace grouping
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Parse provider from model string
        if ':' in model:
            provider_str, model_name = model.split(':', 1)
            # When the caller explicitly opts into Vertex AI, remove any
            # ambient GOOGLE_API_KEY so the underlying google-genai SDK falls
            # through to ADC instead of failing with 401 on the Vertex endpoint
            # (Vertex doesn't accept API keys). load_dotenv() at module import
            # time would otherwise put the AI Studio key back.
            if provider_str == 'google-vertex' and os.environ.get('GOOGLE_API_KEY'):
                os.environ.pop('GOOGLE_API_KEY', None)
                os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', 'true')
            # Handle gateway format: gateway/groq:model -> provider=gateway
            if provider_str.startswith('gateway/'):
                self.provider = LLMProvider.GATEWAY
                self.model = model  # Keep full model string for gateway
            else:
                provider_map = {
                    'google-gla': 'gemini',
                    'google-vertex': 'gemini',
                    'gemini': 'gemini',
                    'anthropic': 'anthropic',
                    'openai': 'openai',
                    'ollama': 'ollama',
                    'mlx-community': 'mlx',
                    'mlx': 'mlx',
                    'groq': 'groq'
                }
                provider_str = provider_map.get(provider_str, provider_str)
                self.provider = LLMProvider(provider_str)

            # For mlx-community, preserve full model path
            if provider_str == 'mlx':
                # If format is "mlx-community:model", keep as "mlx-community/model"
                if model_name and not model_name.startswith('mlx-community/'):
                    self.model = f"mlx-community/{model_name}"
                else:
                    self.model = model_name
            elif self.provider != LLMProvider.GATEWAY:
                # Gateway already has self.model set to full model string
                self.model = model_name
        else:
            raise ValueError(
                f"Model must be in 'provider:model' format, got: {model}"
            )

        # Set up API keys in environment for pydantic_ai
        if self.provider == LLMProvider.GEMINI:
            api_key = api_key or os.getenv("GEMINI_API_KEY"
                                           ) or os.getenv("GOOGLE_API_KEY")
            if api_key:
                # pydantic_ai only needs GOOGLE_API_KEY for Gemini
                os.environ["GOOGLE_API_KEY"] = api_key
        elif self.provider == LLMProvider.ANTHROPIC:
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif self.provider == LLMProvider.OPENAI:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif self.provider == LLMProvider.MLX:
            # MLX runs locally - load the model
            if not MLX_AVAILABLE:
                raise ImportError(
                    "MLX support requires: pip install mlx-lm pydantic-ai[outlines]"
                )
            logger.info(f"Loading MLX model: {self.model}")
            # Load model and tokenizer using mlx_lm
            model_obj, tokenizer_obj = mlx_load(self.model)  # type: ignore
            # Wrap with OutlinesModel for pydantic_ai compatibility
            self.mlx_model_obj = OutlinesModel.from_mlxlm(  # type: ignore
                model_obj, tokenizer_obj
            )
            logger.info(f"MLX model loaded successfully")

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rate_limit_rpm
        ) if rate_limit_rpm else None

        # Create a persistent agent for this client (reused across calls)
        self._agent = None  # Will be created on first use with actual system prompt

        logger.info(f"Initialized LLM client: {self.model_string}")
        if self.rate_limiter:
            logger.info(
                f"Rate limiting enabled: {rate_limit_rpm} requests/minute"
            )

    @property
    def pydantic_model(self):
        """
        Get the model object to use with pydantic_ai Agent.
        
        For MLX models, returns the pre-loaded OutlinesModel object.
        For other providers, returns the model string.
        
        Returns:
            OutlinesModel object for MLX, or model string for other providers
        """
        if self.provider == LLMProvider.MLX:
            if self.mlx_model_obj is None:
                raise ValueError("MLX model not loaded properly")
            return self.mlx_model_obj
        else:
            return self.model_string

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **override_kwargs
    ) -> LLMResponse:
        """
        Generate text from the LLM with automatic retry on failure.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            messages: Full message history (if provided, prompt is ignored)
            **override_kwargs: Override default parameters

        Returns:
            LLMResponse object with generated content
            
        Raises:
            Exception: If all retries fail
        """
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Merge parameters
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs,
            **override_kwargs
        }

        # Use standard retry logic - Logfire auto-instruments pydantic_ai
        return self._generate_with_retry(messages, params)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **override_kwargs
    ) -> LLMResponse:
        """
        Async version of generate.
        """
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        # Merge parameters
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs,
            **override_kwargs
        }

        # Use standard retry logic
        return await self._generate_async_with_retry(messages, params)

    def _generate_with_retry(
        self, messages: List[Dict[str, str]], params: Dict[str, Any]
    ) -> LLMResponse:
        """Generate with retry logic and exponential backoff."""
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._generate_with_pydantic_ai(messages, params)
            except Exception as e:
                last_exception = e

                # Check if this is a retryable error
                # Covers: Gemini, OpenAI, Anthropic, and general HTTP errors
                error_str = str(e).lower()
                is_retryable = any(
                    keyword in error_str for keyword in [
                        # General HTTP errors
                        'rate limit', 'timeout', 'connection', 'network',
                        'service unavailable', '408', '429', '500', '502', '503',
                        '504', 'overloaded', 'retry', 'temporary', 'bad gateway',
                        # Gemini-specific errors
                        'malformed_function_call', 'content field missing',
                        'finish_reason', 'recitation', 'safety',
                        'resource_exhausted', 'deadline_exceeded', 'cancelled',
                        # OpenAI-specific errors
                        'insufficient_quota', 'server_error', 'engine_overloaded',
                        'openai', 'chatcompletion',
                        # Anthropic-specific errors (529 = overloaded)
                        '529', 'overloaded_error', 'anthropic',
                        # Generic
                        'internal', 'unavailable', 'unknown', 'temporarily',
                    ]
                )

                if not is_retryable:
                    logger.error(f"Non-retryable error: {e}")
                    raise

                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = self.retry_delay * (2**attempt
                                                ) + random.uniform(0, 1)
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} retry attempts failed"
                    )

        # If we get here, all retries failed
        raise last_exception if last_exception else Exception(
            "All retries failed"
        )

    async def _generate_async_with_retry(
        self, messages: List[Dict[str, str]], params: Dict[str, Any]
    ) -> LLMResponse:
        """Async version of _generate_with_retry."""
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await self._generate_async_with_pydantic_ai(messages, params)
            except Exception as e:
                last_exception = e

                # Check if this is a retryable error
                error_str = str(e).lower()
                is_retryable = any(
                    keyword in error_str for keyword in [
                        'rate limit', 'timeout', 'connection', 'network',
                        'service unavailable', '408', '429', '500', '502', '503',
                        '504', 'overloaded', 'retry', 'temporary', 'bad gateway',
                        'malformed_function_call', 'content field missing',
                        'finish_reason', 'recitation', 'safety',
                        'resource_exhausted', 'deadline_exceeded', 'cancelled',
                        'insufficient_quota', 'server_error', 'engine_overloaded',
                        'openai', 'chatcompletion',
                        '529', 'overloaded_error', 'anthropic',
                        'internal', 'unavailable', 'unknown', 'temporarily',
                    ]
                )

                if not is_retryable:
                    logger.error(f"Non-retryable error: {e}")
                    raise

                if attempt < self.max_retries - 1:
                    import asyncio
                    delay = self.retry_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} retry attempts failed")

        raise last_exception if last_exception else Exception("All retries failed")

    def _generate_with_pydantic_ai(
        self, messages: List[Dict[str, str]], params: Dict[str, Any]
    ) -> LLMResponse:
        """Generate using pydantic_ai.Agent for robust handling."""
        try:
            # This should never happen due to __init__ check, but satisfies type checker
            if PydanticAgent is None:
                raise ImportError("pydantic_ai not available")

            # Extract system instruction and user messages
            system_instruction = None
            user_prompt = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    if user_prompt:
                        user_prompt += "\n\n" + msg["content"]
                    else:
                        user_prompt = msg["content"]

            # Create agent once and reuse it (for better tracing grouping)
            if self._agent is None:
                # Use session_id as agent name for grouping traces
                agent_name = self.session_id if self.session_id else "llm_agent"

                # For MLX models, use the pre-loaded OutlinesModel object
                if self.provider == LLMProvider.MLX:
                    if self.mlx_model_obj is None:
                        raise ValueError("MLX model not loaded properly")

                    # Create agent with the MLX model object
                    self._agent = PydanticAgent(
                        self.mlx_model_obj,
                        instructions="You are a helpful assistant.",
                        name=agent_name
                    )
                else:
                    # Create agent with model string for other providers
                    self._agent = PydanticAgent(
                        self.model_string,
                        instructions="You are a helpful assistant.",
                        name=agent_name
                    )

            # Build model settings
            # For Gemini 2.5 Flash: DO NOT set max_tokens to avoid thinking token issues
            model_settings = {}
            if "temperature" in params and params["temperature"] is not None:
                model_settings["temperature"] = params["temperature"]

            # Only set max_tokens if explicitly provided (not None)
            if "max_tokens" in params and params["max_tokens"] is not None:
                model_settings["max_tokens"] = params["max_tokens"]

            # If system instruction differs from agent's default, pass it as context
            # Run synchronously with optional system prompt override
            if system_instruction and system_instruction != "You are a helpful assistant.":
                # Prepend system instruction to user prompt
                full_prompt = f"System: {system_instruction}\n\nUser: {user_prompt}"
                result = self._agent.run_sync(
                    full_prompt,
                    model_settings=model_settings
                    if model_settings else None  # type: ignore[arg-type]
                )
            else:
                result = self._agent.run_sync(
                    user_prompt,
                    model_settings=model_settings
                    if model_settings else None  # type: ignore[arg-type]
                )

            # Extract content
            content = None
            if hasattr(result, 'output'):
                content = result.output
            else:
                content = str(result)

            if not content:
                logger.error("Empty content from LLM response")

            # Get usage if available
            usage = None
            if hasattr(result, 'usage'):
                usage_obj = result.usage()
                usage = {
                    "prompt_tokens": getattr(usage_obj, 'request_tokens', 0),
                    "completion_tokens":
                    getattr(usage_obj, 'response_tokens', 0),
                    "total_tokens": getattr(usage_obj, 'total_tokens', 0)
                }

            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                usage=usage,
                raw_response=result
            )

        except Exception as e:
            logger.error(f"pydantic_ai error: {e}")
            raise

    async def _generate_async_with_pydantic_ai(
        self, messages: List[Dict[str, str]], params: Dict[str, Any]
    ) -> LLMResponse:
        """Async version of _generate_with_pydantic_ai."""
        try:
            if PydanticAgent is None:
                raise ImportError("pydantic_ai not available")

            # Extract system instruction and user messages
            system_instruction = None
            user_prompt = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    if user_prompt:
                        user_prompt += "\n\n" + msg["content"]
                    else:
                        user_prompt = msg["content"]

            # Create agent once and reuse it
            if self._agent is None:
                agent_name = self.session_id if self.session_id else "llm_agent"
                if self.provider == LLMProvider.MLX:
                    if self.mlx_model_obj is None:
                        raise ValueError("MLX model not loaded properly")
                    self._agent = PydanticAgent(
                        self.mlx_model_obj,
                        instructions="You are a helpful assistant.",
                        name=agent_name
                    )
                else:
                    self._agent = PydanticAgent(
                        self.model_string,
                        instructions="You are a helpful assistant.",
                        name=agent_name
                    )

            # Build model settings
            model_settings = {}
            if "temperature" in params and params["temperature"] is not None:
                model_settings["temperature"] = params["temperature"]
            if "max_tokens" in params and params["max_tokens"] is not None:
                model_settings["max_tokens"] = params["max_tokens"]

            # Use run instead of run_sync
            if system_instruction and system_instruction != "You are a helpful assistant.":
                full_prompt = f"System: {system_instruction}\n\nUser: {user_prompt}"
                result = await self._agent.run(
                    full_prompt,
                    model_settings=model_settings if model_settings else None
                )
            else:
                result = await self._agent.run(
                    user_prompt,
                    model_settings=model_settings if model_settings else None
                )

            # Extract content
            content = None
            if hasattr(result, 'output'):
                content = result.output
            else:
                content = str(result)

            if not content:
                logger.error("Empty content from LLM response")

            # Get usage if available
            usage = None
            if hasattr(result, 'usage'):
                usage_obj = result.usage()
                usage = {
                    "prompt_tokens": getattr(usage_obj, 'request_tokens', 0),
                    "completion_tokens": getattr(usage_obj, 'response_tokens', 0),
                    "total_tokens": getattr(usage_obj, 'total_tokens', 0)
                }

            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                usage=usage,
                raw_response=result
            )

        except Exception as e:
            logger.error(f"pydantic_ai error: {e}")
            logger.exception("Full traceback:")
            raise
