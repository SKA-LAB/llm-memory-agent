from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama import chat
from pydantic import BaseModel
from typing import Union, Optional
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

def get_llm(
    provider: str = "ollama", 
    model: str = None, 
    temperature: float = 0.7,
    **kwargs
) -> Union[ChatOllama, ChatOpenAI]:
    """
    Get a language model instance based on the provider.
    
    Args:
        provider: Either "ollama" or "openai"
        model: Model name (defaults based on provider)
        temperature: Temperature for generation
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Language model instance
    """
    if provider.lower() == "ollama":
        default_model = model or "llama3.2"
        return ChatOllama(
            model=default_model,
            temperature=temperature,
            num_ctx=kwargs.get("num_ctx", 100000),
            **{k: v for k, v in kwargs.items() if k != "num_ctx"}
        )
    elif provider.lower() == "openai":
        default_model = model or "Qwen/Qwen2.5-7B-Instruct-Turbo"
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        return ChatOpenAI(
            model=default_model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            **{k: v for k, v in kwargs.items() if k not in ["api_key", "num_ctx"]}
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'ollama' or 'openai'")

def parse_to_json(
    text: str, 
    response_schema: BaseModel, 
    llm_instance: Optional[Union[ChatOllama, ChatOpenAI]] = None,
    model: str = None
) -> dict:
    """
    Parse text to JSON using either a provided LLM instance or Ollama chat.
    
    Args:
        text: Text to parse
        response_schema: Pydantic model schema for the response
        llm_instance: Optional LLM instance to use for parsing
        model: Model name for Ollama chat (used if llm_instance is None)
    
    Returns:
        Parsed response as a dictionary
    """
    if llm_instance is not None:
        # Use the provided LLM instance (works with both OpenAI and Ollama)
        system_message = f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {response_schema.model_json_schema()}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        
        # For structured output with OpenAI
        if isinstance(llm_instance, ChatOpenAI):
            try:
                # Try using structured output if available
                structured_llm = llm_instance.with_structured_output(response_schema)
                response = structured_llm.invoke(messages)
                return response
            except Exception:
                # Fallback to regular chat completion
                response = llm_instance.invoke(messages)
                return response_schema.model_validate_json(response.content)
        else:
            # For Ollama and other providers
            response = llm_instance.invoke(messages)
            return response_schema.model_validate_json(response.content)
    else:
        # Fallback to direct Ollama chat (original behavior)
        default_model = model or 'Osmosis/Osmosis-Structure-0.6B'
        response = chat(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {response_schema.model_json_schema()}"
                },
                {
                    'role': 'user',
                    'content': text,
                }
            ],
            model=default_model,
            format=response_schema.model_json_schema(),
        )
        
        answer = response_schema.model_validate_json(response.message.content)
        return answer

# Convenience functions for backward compatibility
def get_ollama_llm(model: str = "llama3.2", temperature: float = 0.7, **kwargs) -> ChatOllama:
    """Get an Ollama LLM instance."""
    return get_llm(provider="ollama", model=model, temperature=temperature, **kwargs)

def get_openai_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.7, **kwargs) -> ChatOpenAI:
    """Get an OpenAI LLM instance."""
    return get_llm(provider="openai", model=model, temperature=temperature, **kwargs)