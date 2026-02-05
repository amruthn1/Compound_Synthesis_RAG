"""OpenRouter API client for LLM calls."""

import os
import requests
from typing import Optional


class OpenRouterAgent:
    """Agent that uses OpenRouter API instead of local models."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen-2.5-7b-instruct",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter agent.
        
        Args:
            api_key: OpenRouter API key (reads from env if None)
            model: Model to use (default: qwen/qwen-2.5-7b-instruct)
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/materials-rag",
            "X-Title": "Materials Science RAG",
            "Content-Type": "application/json"
        }
        
        print(f"OpenRouter Agent initialized with model: {model}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text using OpenRouter API.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (ignored, always uses sampling)
            
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"OpenRouter API error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nDetails: {error_detail}"
                except:
                    error_msg += f"\nResponse: {e.response.text}"
            
            print(error_msg)
            raise Exception(error_msg)
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Alias for generate() method for compatibility.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        return self.generate(prompt, max_new_tokens, temperature, top_p, do_sample)
    
    def format_chat_prompt(
        self,
        system_message: str,
        user_message: str
    ) -> str:
        """
        Format a chat prompt (for compatibility with LlamaAgent).
        OpenRouter handles this internally, so just return combined text.
        
        Args:
            system_message: System prompt
            user_message: User message
            
        Returns:
            Combined prompt
        """
        return f"{system_message}\n\n{user_message}"
