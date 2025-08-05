import logging
from typing import List, Optional, Tuple

import numpy as np
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    INSTRUCTION_PROMPT, MAX_TOKENS, TEMPERATURE, LOGPROBS, OPENAI_API_KEY
)

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, backend: str, model_name: str, device: int = 0):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.generator = None
        
        if backend == "openai":
            self._setup_openai()
        elif backend == "hf":
            self._setup_huggingface()
        else:
            raise ValueError(f"Backend '{backend}' not supported")
    
    def _setup_openai(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not defined for OpenAI backend")
        openai.api_key = OPENAI_API_KEY
        logger.info(f"OpenAI client configured (model: {self.model_name})")
    
    def _setup_huggingface(self):
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading HF model: {e}")
            raise
    
    def generate(self, context: str) -> Tuple[str, float, float]:
        if self.backend == "openai":
            return self._call_openai(context)
        else:
            return self._call_huggingface(context)
    
    def _call_openai(self, context: str) -> Tuple[str, float, float]:
        try:
            full_prompt = INSTRUCTION_PROMPT + context
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                logprobs=LOGPROBS
            )
            choice = response.choices[0]
            completion = self._extract_single_line(choice.message.content.strip())
            pavg, ptot = self._compute_confidence(choice.logprobs.token_logprobs or [])
            return completion, pavg, ptot
        except Exception as e:
            logger.error(f"Error in OpenAI call: {e}")
            return "", 0.0, 0.0
    
    @torch.no_grad()
    def _call_huggingface(self, context: str) -> Tuple[str, float, float]:
        try:
            full_prompt = INSTRUCTION_PROMPT + context
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            # Geração sem sampling
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

            generated_ids = output_ids.sequences[0][inputs['input_ids'].shape[-1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion = self._extract_single_line(generated_text.strip())

            # Extrai scores dos tokens gerados
            scores = output_ids.scores  # lista de tensores (1, vocab_size), um por token
            probs = [torch.softmax(score, dim=-1) for score in scores]

            token_logprobs = []
            for prob, token_id in zip(probs, generated_ids):
                token_logprobs.append(torch.log(prob[0, token_id]).item())

            pavg, ptot = self._compute_confidence(token_logprobs)

            return completion, pavg, ptot
        except Exception as e:
            logger.error(f"Error in HuggingFace call: {e}")
            return "", 0.0, 0.0
    
    def _compute_confidence(self, logprobs: List[Optional[float]]) -> Tuple[float, float]:
        if not logprobs:
            return 0.0, 0.0

        valid_probs = [np.exp(lp) for lp in logprobs if lp is not None]
        if not valid_probs:
            return 0.0, 0.0

        mean_conf = float(np.mean(valid_probs))
        product_conf = float(np.prod(valid_probs))
        return mean_conf, product_conf
    
    def _extract_single_line(self, generated_text: str) -> str:
        if not generated_text:
            return ""
        
        if "Complete the Python function" in generated_text:
            lines = generated_text.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not any(keyword in line for keyword in 
                    ["Complete", "Return", "function", "single", "code", "explanations"]):
                    generated_text = '\n'.join(lines[i:])
                    break
        
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                return line
        
        return lines[0].strip() if lines else ""