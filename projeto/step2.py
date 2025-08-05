import os
import re
import json
import faiss
import torch
import random
import logging
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ModelManager:
    def __init__(self):
        self.hf_tokenizer = None
        self.hf_model = None
        self.rag_model = None
        self.rag_index = None
        self.rag_code_tokenizer = None
        self.rag_code_model = None
        self.rag_code_index = None
        self.bm25_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_hf_model(self, model_path: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        if self.hf_tokenizer is None or self.hf_model is None:
            logger.info(f"Loading HF model: {model_path}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
            logger.info("HF model loaded successfully")
        return self.hf_tokenizer, self.hf_model
    
    def load_rag_model(self, model_name: str = "all-MiniLM-L6-v2"):
        if self.rag_model is None:
            logger.info(f"Loading RAG model: {model_name}")
            self.rag_model = SentenceTransformer(model_name)
            logger.info("RAG model loaded successfully")
        return self.rag_model
    
    def load_rag_code_model(self, model_name: str = "microsoft/codebert-base"):
        if self.rag_code_tokenizer is None or self.rag_code_model is None:
            logger.info(f"Loading RAG code model: {model_name}")
            self.rag_code_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.rag_code_model = AutoModel.from_pretrained(model_name)
            self.rag_code_model.eval()
            logger.info("RAG code model loaded successfully")
        return self.rag_code_tokenizer, self.rag_code_model

model_manager = ModelManager()

def preparar_index_rag(dataset_slice):
    logger.info(f"Preparing optimized RAG index for {len(dataset_slice)} samples")
    model = model_manager.load_rag_model()
    contexts = [ex["context"] + "\n" + ex["predicted_line"] for ex in dataset_slice]
    
    logger.info(f"Encoding {len(contexts)} contexts for RAG")
    embeddings = model.encode(contexts, convert_to_numpy=True, show_progress_bar=True)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("Optimized RAG index prepared successfully")
    return index

def preparar_index_rag_code(dataset_slice):
    logger.info(f"Preparing optimized RAG code index for {len(dataset_slice)} samples")
    tokenizer, model = model_manager.load_rag_code_model()

    contexts = [ex["context"] + "\n" + ex["predicted_line"] for ex in dataset_slice]
    embeddings = []
    logger.info(f"Encoding {len(contexts)} contexts for RAG code")

    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            if i % (batch_size * 4) == 0:
                logger.debug(f"Encoded {i}/{len(contexts)} contexts")
            
            for text in batch:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[0][0].numpy()
                embeddings.append(cls_embedding)

    embeddings = np.vstack(embeddings)
    logger.info(f"Generated code embeddings shape: {embeddings.shape}")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("Optimized RAG code index prepared successfully")
    return index

def preparar_bm25(dataset_slice):
    logger.info(f"Preparing optimized BM25 for {len(dataset_slice)} samples")
    corpus = [re.findall(r"\w+", (ex["context"] + "\n" + ex["predicted_line"]).lower()) for ex in dataset_slice]
    bm25 = BM25Okapi(corpus)
    logger.info("Optimized BM25 prepared successfully")
    return bm25

def montar_prompt_reflexivo(context: str, predicted_line: str, selector=None) -> str:
    logger.debug("Building reflective prompt")
    few_shot_exemplos = selector(context, predicted_line) if selector else []
    logger.debug(f"Selected {len(few_shot_exemplos)} few-shot examples")

    prompt_parts = []

    for i, exemplo in enumerate(few_shot_exemplos):
        prompt_parts.append(
            f"{exemplo['context']}\n{exemplo['predicted_line']}\nReflexão: {exemplo['reflexao']}"
        )

    prompt_parts.append("---")
    if not few_shot_exemplos:
        prompt_parts.append("Is the code below correct? Answer only with True or False.\n")

    prompt_parts.append(f"{context}\n{predicted_line}\nReflexão:")
    
    final_prompt = "\n\n".join(prompt_parts)
    logger.debug(f"Final reflective prompt length: {len(final_prompt)} characters")
    
    return final_prompt

def montar_prompt_verbalized(context: str, predicted_line: str, selector=None) -> str:
    """Prompt para Verbalized Self-Ask (pv)"""
    logger.debug("Building verbalized self-ask prompt")
    few_shot_exemplos = selector(context, predicted_line) if selector else []
    logger.debug(f"Selected {len(few_shot_exemplos)} few-shot examples for verbalized prompt")

    prompt_parts = []

    for i, exemplo in enumerate(few_shot_exemplos):
        confidence_score = 0.8 if exemplo['reflexao'] == "True" else 0.2
        prompt_parts.append(
            f"{exemplo['context']}\n{exemplo['predicted_line']}\nConfiança: {confidence_score}"
        )

    prompt_parts.append("---")
    if not few_shot_exemplos:
        prompt_parts.append("Rate your confidence in the correctness of the code below on a scale from 0.0 to 1.0:\n")

    prompt_parts.append(f"{context}\n{predicted_line}\nConfiança:")
    
    final_prompt = "\n\n".join(prompt_parts)
    logger.debug(f"Final verbalized prompt length: {len(final_prompt)} characters")

    return final_prompt

class SelectorManager:
    def __init__(self, dataset_slice, k_shots):
        self.dataset_slice = dataset_slice
        self.k_shots = k_shots
        self.rag_index = None
        self.rag_code_index = None
        self.bm25_model = None
        
    def setup_rag(self):
        if self.rag_index is None:
            self.rag_index = preparar_index_rag(self.dataset_slice)
    
    def setup_rag_code(self):
        if self.rag_code_index is None:
            self.rag_code_index = preparar_index_rag_code(self.dataset_slice)
    
    def setup_bm25(self):
        if self.bm25_model is None:
            self.bm25_model = preparar_bm25(self.dataset_slice)

    def selector_random(self, context, predicted_line):
        logger.debug(f"Using random selector with k_shots={self.k_shots}")
        exemplos = random.sample(self.dataset_slice, min(self.k_shots, len(self.dataset_slice)))
        return [
            {
                "context": ex["context"],
                "predicted_line": ex["predicted_line"],
                "reflexao": "True" if ex["is_correct"] else "False"
            }
            for ex in exemplos
        ]

    def selector_bm25(self, context, predicted_line):
        logger.debug(f"Using BM25 selector with k_shots={self.k_shots}")
        self.setup_bm25()
        
        query = re.findall(r"\w+", (context + "\n" + predicted_line).lower())
        scores = self.bm25_model.get_scores(query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k_shots]
        
        return [
            {
                "context": self.dataset_slice[i]["context"],
                "predicted_line": self.dataset_slice[i]["predicted_line"],
                "reflexao": "True" if self.dataset_slice[i]["is_correct"] else "False"
            }
            for i in top_k_indices
        ]

    def selector_rag_code(self, context, predicted_line):
        logger.debug(f"Using RAG code selector with k_shots={self.k_shots}")
        self.setup_rag_code()
        
        tokenizer, model = model_manager.load_rag_code_model()
        query_text = context + "\n" + predicted_line
        inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            query_embedding = outputs.last_hidden_state[0][0].numpy().reshape(1, -1)

        _, I = self.rag_code_index.search(query_embedding, self.k_shots)
        
        return [
            {
                "context": self.dataset_slice[i]["context"],
                "predicted_line": self.dataset_slice[i]["predicted_line"],
                "reflexao": "True" if self.dataset_slice[i]["is_correct"] else "False"
            }
            for i in I[0]
        ]

def avaliar_verbalized_hf(
    prompt: str,
    predicted_line: str,
    context: str,
    is_correct: bool
):
    logger.debug("Evaluating verbalized self-ask (optimized)")
    
    tokenizer, model = model_manager.load_hf_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model_manager.device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.debug(f"Generated: '{generated_text}'")
    
    match = re.search(r'(\d+\.?\d*)', generated_text.strip())
    if match:
        try:
            pv = float(match.group(1))
            if pv > 1.0:
                pv = pv / 10.0 if pv <= 10.0 else 1.0
            logger.debug(f"Extracted pv: {pv}")
        except:
            pv = 0.5
    else:
        pv = 0.5

    return {
        "pv": pv,
        "generated_text": generated_text,
        "is_correct": is_correct
    }

def avaliar_reflexao_hf(
    prompt: str,
    predicted_line: str,
    context: str,
    is_correct: bool
):
    logger.debug("Evaluating question answering logits (optimized)")
    
    tokenizer, model = model_manager.load_hf_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model_manager.device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_token_id = output.sequences[0][-1].item()
    generated_token = tokenizer.decode(generated_token_id).strip()
    scores = output.scores[0][0]

    true_variations = ["True", " True", "true", " true"]
    false_variations = ["False", " False", "false", " false"]
    
    true_logprobs = []
    false_logprobs = []
    
    for var in true_variations:
        token_id = tokenizer.convert_tokens_to_ids(var)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            true_logprobs.append(scores[token_id].item())
    
    for var in false_variations:
        token_id = tokenizer.convert_tokens_to_ids(var)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            false_logprobs.append(scores[token_id].item())

    true_id = tokenizer.convert_tokens_to_ids("True")
    all_probs = torch.softmax(scores, dim=0)
    pB = all_probs[true_id].item() if true_id is not None else 0.0

    if true_logprobs and false_logprobs:
        max_true_logprob = max(true_logprobs)
        max_false_logprob = max(false_logprobs)
        prob_true = np.exp(max_true_logprob)
        prob_false = np.exp(max_false_logprob)
        pNB = prob_true / (prob_true + prob_false)
    else:
        pNB = 0.5

    resposta_binaria = "True" if "true" in generated_token.lower() else "False" if "false" in generated_token.lower() else "Unknown"

    return {
        "resposta": resposta_binaria,
        "pB": pB,
        "pNB": pNB,
        "is_correct": is_correct,
        "predicted_line": predicted_line,
        "context": context,
        "token": generated_token
    }

def avaliar_todas_metricas(predicted_line, context, is_correct, pavg, ptot, selector=None):
    logger.debug("Starting optimized comprehensive metric evaluation")
    
    prompt_pv = montar_prompt_verbalized(context, predicted_line, selector)
    result_pv = avaliar_verbalized_hf(prompt_pv, predicted_line, context, is_correct)
    
    prompt_tf = montar_prompt_reflexivo(context, predicted_line, selector)
    result_tf = avaliar_reflexao_hf(prompt_tf, predicted_line, context, is_correct)
    
    return {
        "context": context,
        "predicted_line": predicted_line,
        "pavg": pavg,
        "ptot": ptot,
        "pv": result_pv["pv"],
        "ask_tf": result_tf["pB"],
        "ask_tf_n": result_tf["pNB"],
        "is_correct": is_correct
    }

def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    return np.mean((np.array(y_prob) - np.array(y_true)) ** 2)

def skill_score(y_true: List[int], y_prob: List[float]) -> float:
    b_model = brier_score(y_true, y_prob)
    p_base = np.mean(y_true)
    b_base = p_base * (1 - p_base)
    return (b_base - b_model) / b_base if b_base > 0 else 0.0

def expected_calibration_error(y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        low, high = bin_bounds[i], bin_bounds[i+1]
        in_bin = (y_prob >= low) & (y_prob < high)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total) * abs(acc - conf)
    return ece

if __name__ == "__main__":
    logger.info("Starting optimized main execution")

    file_path = "results/deepseek_predictions.json"
    logger.info(f"Loading dataset from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Dataset loaded: {len(dataset)}")

    k_shots = 5
    logger.info(f"k_shots set to: {k_shots}")

    selectors_config = [
        ("0_shot", None),
        ("fs_random", "random"),
        ("fs_bm25", "bm25"),
        ("rag", "rag_code")
    ]

    logger.info(f"Testing {len(selectors_config)} selector configurations")

    for selector_idx, (selector_name, selector_type) in enumerate(selectors_config):
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing configuration {selector_idx+1}/{len(selectors_config)}: {selector_name}")
        logger.info(f"{'='*50}")
        
        current_k_shots = 0 if selector_name == "0_shot" else k_shots
        
        if selector_type:
            selector_manager = SelectorManager(dataset, current_k_shots)
            if selector_type == "random":
                selector_func = selector_manager.selector_random
            elif selector_type == "bm25":
                selector_func = selector_manager.selector_bm25
            elif selector_type == "rag_code":
                selector_func = selector_manager.selector_rag_code
        else:
            selector_func = None
        
        all_results = []
        
        logger.info(f"Processing {len(dataset)} samples...")
        for i, ex in enumerate(dataset):
            if i % 2 == 0:
                logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
            result = avaliar_todas_metricas(
                predicted_line=ex["predicted_line"],
                context=ex["context"],
                is_correct=ex["is_correct"],
                pavg=ex["pavg"],
                ptot=ex["ptot"],
                selector=selector_func
            )
            result["sample_id"] = ex.get("sample_id", i)
            all_results.append(result)
        
        output_filename = f"plots_mini/confidence_results_{selector_name}.json"
        logger.info(f"Saving results to: {output_filename}")
        if os.path.exists(output_filename):
            with open(output_filename, 'r', encoding='utf-8') as f:
                try:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        logger.warning("Existing results are not a list. Overwriting.")
                        existing_results = []
                except json.JSONDecodeError:
                    logger.warning("Could not decode existing JSON. Overwriting.")
                    existing_results = []
        else:
            existing_results = []

        combined_results = existing_results + all_results

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Total results now stored: {len(combined_results)}")
        
        y_true = [r["is_correct"] for r in all_results]
        y_prob_pnb = [r["ask_tf_n"] for r in all_results]
        
        brier = brier_score(y_true, y_prob_pnb)
        skill = skill_score(y_true, y_prob_pnb)
        ece = expected_calibration_error(y_true, y_prob_pnb)
        
        logger.info(f"\n=== Calibration Metrics for {selector_name} (using pNB) ===")
        logger.info(f"- Brier Score: {brier:.4f}")
        logger.info(f"- Skill Score: {skill:.4f}")
        logger.info(f"- Expected Calibration Error (ECE): {ece:.4f}")
        
        metrics_filename = f"metricas_calibracao_{selector_name}.txt"
        with open(metrics_filename, "w") as f:
            f.write(f"Brier Score: {brier:.4f}\n")
            f.write(f"Skill Score: {skill:.4f}\n")
            f.write(f"Expected Calibration Error (ECE): {ece:.4f}\n")
        
        logger.info(f"Configuration {selector_name} completed successfully")

    logger.info(f"\n{'='*50}")
    logger.info("Processing completed successfully!")
    logger.info(f"{'='*50}")
