"""Llama-3.1 based agent for materials reasoning."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class LlamaAgent:
    """Llama-3.1 agent for materials science reasoning."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = True
    ):
        """
        Initialize Llama agent.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use (cuda/cpu, auto-detected if None)
            load_in_4bit: Use 4-bit quantization (recommended for Phi-3)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Quantization: {'4-bit' if load_in_4bit and self.device == 'cuda' else 'FP16/FP32'}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if load_in_4bit and self.device == "cuda":
            # Stable 4-bit config that works with Phi-3, Mistral, Qwen
            model_kwargs.update({
                "load_in_4bit": True,
                "device_map": "auto",
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            })
        else:
            # FP16/FP32 fallback
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if not load_in_4bit and self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        # Move inputs to the model's current device.
        # This works for both quantized and non-quantized models.
        input_device = next(self.model.parameters()).device
        inputs = inputs.to(input_device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Alias for generate() method.
        
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
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Format messages into Llama chat format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Fallback format
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == 'user':
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"<|assistant|>\n{content}\n")
        
        prompt_parts.append("<|assistant|>\n")
        return "".join(prompt_parts)
    
    def query_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024
    ) -> str:
        """
        Query the model with retrieved context (RAG).
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: System prompt (optional)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert materials scientist specializing in solid-state chemistry, "
                "synthesis methods, and materials characterization. Provide detailed, accurate, "
                "and scientifically rigorous answers based on the provided context."
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        prompt = self.format_chat_prompt(messages)
        return self.generate(prompt, max_new_tokens=max_new_tokens)
    
    def extract_information(
        self,
        text: str,
        instruction: str,
        max_new_tokens: int = 512
    ) -> str:
        """
        Extract specific information from text.
        
        Args:
            text: Input text
            instruction: What to extract
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Extracted information
        """
        messages = [
            {
                "role": "system",
                "content": "You are a precise information extraction assistant. Extract only the requested information."
            },
            {
                "role": "user",
                "content": f"Text:\n{text}\n\nInstruction: {instruction}"
            }
        ]
        
        prompt = self.format_chat_prompt(messages)
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.3)


if __name__ == "__main__":
    # Test Llama agent
    print("Testing Llama agent...")
    
    agent = LlamaAgent(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=torch.cuda.is_available()
    )
    
    # Test simple query
    response = agent.query_with_context(
        query="What is the typical synthesis temperature for BaTiO3?",
        context="BaTiO3 is typically synthesized at 1000-1400°C using solid-state reaction.",
        max_new_tokens=256
    )
    
    print(f"\nResponse:\n{response}")
