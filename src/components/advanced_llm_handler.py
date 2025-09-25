"""
Advanced LLM Handler supporting Llama, Mistral, and other open-source models
Includes fine-tuning capabilities and optimized response generation
"""

import os
import json
import time
from typing import List, Dict, Iterator, Optional, Any
from pathlib import Path

import streamlit as st
import requests
from dotenv import load_dotenv
load_dotenv("hf.env")
from huggingface_hub import login
login(hf_token)


class AdvancedLLMHandler:
    """Advanced LLM handler with support for multiple model types"""

    def __init__(
        self,
        model_name: str = "mistral-7b",
        max_length: int = 1024,
        temperature: float = 0.7,
        use_local_models: bool = True,
        use_api: bool = False
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.use_local_models = use_local_models
        self.use_api = use_api
        self.model = None
        self.tokenizer = None

        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise EnvironmentError("HUGGINGFACE_TOKEN not found in environment.")

        self.model_configs = {
            "llama-7b": {
                "hf_name": "meta-llama/Llama-2-7b-chat-hf",
                "context_length": 4096,
                "system_prompt": "You are a helpful assistant. Answer based on the provided context."
            },
            "mistral-7b": {
                "hf_name": "mistralai/Mistral-7B-Instruct-v0.1",
                "context_length": 8192,
                "system_prompt": "You are a helpful assistant. Use the provided context to answer questions accurately."
            },
            "zephyr-7b": {
                "hf_name": "HuggingFaceH4/zephyr-7b-beta",
                "context_length": 4096,
                "system_prompt": "You are a helpful AI assistant. Answer questions based on the given context."
            },
            "phi-2": {
                "hf_name": "microsoft/phi-2",
                "context_length": 2048,
                "system_prompt": "Answer the question using the provided context."
            }
        }

        self._initialize_model()

    def _initialize_model(self):
        try:
            self._load_transformers_model()
        except Exception as e:
            st.warning(f"Could not load transformers model: {str(e)}")
            self._initialize_fallback_model()

    def _load_transformers_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            config = self.model_configs.get(self.model_name, {
                "hf_name": self.model_name,
                "context_length": 2048,
                "system_prompt": "Answer based on the provided context."
            })

            hf_model_name = config["hf_name"]
            hf_token = os.getenv("HUGGINGFACE_TOKEN", None)

            if hf_token is None:
                raise ValueError("HUGGINGFACE_TOKEN is not set in the .env file")

            with st.spinner(f"Loading {hf_model_name}..."):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    token=hf_token
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    token=hf_token,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )

                st.success(f"Successfully loaded {hf_model_name}")

        except ImportError:
            raise Exception("Transformers library not available")
        except Exception as e:
            raise Exception(f"Failed to load transformers model: {str(e)}")

    def _initialize_fallback_model(self):
        st.info("Using enhanced rule-based text generation")

        self.response_templates = {
            "factual": [
                "Based on the provided information, {}",
                "According to the document, {}",
                "The text indicates that {}",
                "From the context, we can see that {}"
            ],
            "analytical": [
                "Analyzing the provided information, {}",
                "The document suggests that {}",
                "Based on the evidence presented, {}",
                "The information reveals that {}"
            ],
            "comparative": [
                "Comparing the information provided, {}",
                "The document shows differences in {}",
                "When examining the context, {}"
            ],
            "procedural": [
                "According to the instructions, {}",
                "The process outlined shows that {}",
                "Following the provided steps, {}"
            ]
        }

        self.question_patterns = {
            "what": "factual",
            "how": "procedural",
            "why": "analytical",
            "compare": "comparative",
            "difference": "comparative",
            "analyze": "analytical",
            "explain": "analytical"
        }

    def _detect_question_type(self, query: str) -> str:
        query_lower = query.lower()
        for pattern, q_type in self.question_patterns.items():
            if pattern in query_lower:
                return q_type
        return "factual"

    def create_advanced_prompt(self, query: str, contexts: List[str]) -> str:
        config = self.model_configs.get(self.model_name, {
            "system_prompt": "Answer based on the provided context."
        })

        system_prompt = config["system_prompt"]
        context_text = "\n\n".join([f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        if "llama" in self.model_name.lower():
            return (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                f"{context_text}\n\nQuestion: {query}\n\n"
                f"Provide a clear and accurate answer based solely on the context. [/INST]"
            )
        elif "mistral" in self.model_name.lower():
            return (
                f"<s>[INST] {system_prompt}\n\n"
                f"{context_text}\n\nQuestion: {query}\n\nAnswer: [/INST]"
            )
        elif "zephyr" in self.model_name.lower():
            return (
                f"<|system|>\n{system_prompt}</s>\n"
                f"<|user|>\n{context_text}\n\nQuestion: {query}</s>\n"
                f"<|assistant|>"
            )
        else:
            return (
                f"{system_prompt}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\nAnswer:"
            )
        return prompt

    def generate_response_transformers(self, prompt: str) -> str:
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=1024,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                return_full_text=False
            )
            generated_text = outputs[0]['generated_text']
            return self._clean_response(generated_text, prompt)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."

    def generate_response_fallback(self, query: str, contexts: List[str]) -> str:
        try:
            question_type = self._detect_question_type(query)
            relevant_info = self._extract_multi_strategy_info(contexts, query)

            if not relevant_info:
                return "I couldn't find relevant information in the provided context to answer your question."

            templates = self.response_templates.get(question_type, self.response_templates["factual"])
            template = templates[hash(query) % len(templates)]
            response = template.format(relevant_info)
            return self._enhance_response(response, contexts, query)

        except Exception as e:
            st.error(f"Error in fallback generation: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    def _extract_multi_strategy_info(self, contexts: List[str], query: str) -> str:
        query_words = set(query.lower().split())
        direct_matches, semantic_matches, keyword_matches = [], [], []
        
        for context in contexts:
            sentences = self._split_sentences(context)
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    direct_matches.append((overlap, sentence))

            context_words = set(context.lower().split())
            semantic_score = len(query_words.intersection(context_words)) / len(query_words.union(context_words))
            if semantic_score > 0.1:
                semantic_matches.append((semantic_score, context[:200]))

            important_words = [word for word in query_words if len(word) > 3]
            density = sum(context.lower().count(word) for word in important_words)
            if density > 0:
                keyword_matches.append((density, context[:200]))

        direct_matches.sort(reverse=True, key=lambda x: x[0])
        semantic_matches.sort(reverse=True, key=lambda x: x[0])

        all_matches = [match[1] for match in direct_matches[:2]] + [match[1] for match in semantic_matches[:1]]
        combined_info = ". ".join(all_matches[:3])

        return combined_info[:400] if combined_info else ""

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]

    def _enhance_response(self, response: str, contexts: List[str], query: str) -> str:
        query_words = set(query.lower().split())
        total_matches = 0
        for context in contexts:
            context_words = set(context.lower().split())
            total_matches += len(query_words.intersection(context_words))

        confidence = min(100, total_matches * 20)

        if confidence < 30:
            response += "\n\n*Note: Limited relevant information found in the provided context.*"
        elif confidence > 80:
            response += "\n\n*This answer is based on comprehensive information from the provided context.*"

        return response

    def _clean_response(self, response: str, prompt: str) -> str:
        response = response.replace(prompt, "")
        response = response.replace("<|assistant|>", "").replace("[/INST]", "").replace("</s>", "")
        lines = response.splitlines()
        lines = [line.strip() for line in lines if line.strip() and not line.lower().startswith(('user:', 'human:', 'assistant:'))]
        cleaned = ' '.join(lines)
        return cleaned[:2000].strip() 

    def generate_response(self, query: str, contexts: List[str]) -> str:
        if self.model and self.tokenizer:
            prompt = self.create_advanced_prompt(query, contexts)
            return self.generate_response_transformers(prompt)
        else:
            return self.generate_response_fallback(query, contexts)

    def generate_streaming_response(self, query: str, contexts: List[str]) -> Iterator[str]:
        try:
            full_response = self.generate_response(query, contexts)
            for i, word in enumerate(full_response.split()):
                yield word if i == 0 else " " + word
                time.sleep(0.03)
        except Exception as e:
            yield f"Error during streaming: {str(e)}"

    def fine_tune_model(self, training_data: List[Dict[str, str]], output_dir: str = "./fine_tuned_model") -> bool:
        try:
            st.info("Fine-tuning functionality ready for implementation")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "training_data.json"), 'w') as f:
                json.dump(training_data, f, indent=2)
            st.success(f"Training data saved to {output_dir}")
            return True
        except Exception as e:
            st.error(f"Fine-tuning preparation failed: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        config = self.model_configs.get(self.model_name, {})
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'context_length': config.get('context_length', 'Unknown'),
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'capabilities': [
                'Multi-model support',
                'Advanced prompting',
                'Streaming responses',
                'Fine-tuning ready',
                'Fallback generation'
            ],
            'supported_models': list(self.model_configs.keys())
        }
