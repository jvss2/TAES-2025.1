#!/usr/bin/env python3
"""
Script para avaliar modelos de linguagem na tarefa de completar código Python.
Usa o dataset dypybench_functions para testar a capacidade do modelo de gerar
a linha correta removida de uma função.
"""

import os
import json
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import openai
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Dataset
DATASET_NAME = "claudios/dypybench_functions"
DATASET_SPLIT = "train"
MIN_EXECUTED_LINES = 2
SAMPLE_SIZE = None  # Use None para processar todas as amostras

# Modelo
LLM_BACKEND = "hf"  # "openai" ou "hf"
# LLM_MODEL = "Salesforce/codegen-350M-mono"
LLM_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
HF_DEVICE = 0

# Geração
MAX_TOKENS = 32  # Reduzido para evitar gerações longas
TEMPERATURE = 0
LOGPROBS = 1
PLACEHOLDER = "# [YOUR_LINE_HERE]"

# Prompt para instrução do modelo
INSTRUCTION_PROMPT = """Complete the Python function below by generating the next line of code.
Return ONLY the single next line that should come after the existing code.
Do not add any extra code, comments, or explanations beyond that single line.

"""

# Output
OUTPUT_FILE = "results2/dypybench_predictions"

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================

def setup_logging():
    """Configura o sistema de logging com formatação clara."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# CLASSES DE DADOS
# =============================================================================

@dataclass
class PredictionResult:
    """Resultado de uma predição do modelo."""
    sample_id: int
    context: str
    original_line: str
    predicted_line: str
    confidence: float
    is_correct: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para serialização JSON."""
        return asdict(self)

# =============================================================================
# INICIALIZAÇÃO DO MODELO
# =============================================================================

class ModelManager:
    """Gerencia a inicialização e chamadas do modelo."""
    
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
            raise ValueError(f"Backend '{backend}' não suportado")
    
    def _setup_openai(self):
        """Configura o cliente OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não definida para backend OpenAI")
        openai.api_key = api_key
        logger.info(f"✓ Cliente OpenAI configurado (modelo: {self.model_name})")
    
    def _setup_huggingface(self):
        """Configura o pipeline do Hugging Face."""
        try:
            logger.info(f"⏳ Carregando modelo HuggingFace: {self.model_name}")
            self.generator = pipeline(
                task="text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            logger.info(f"✓ Modelo HuggingFace carregado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo HF: {e}")
            raise
    
    def generate(self, context: str) -> Tuple[str, float, float]:
        """Gera uma completação para o contexto dado."""
        if self.backend == "openai":
            return self._call_openai(context)
        else:
            return self._call_huggingface(context)
    
    def _call_openai(self, context: str) -> Tuple[str, float]:
        """Faz chamada para o modelo OpenAI."""
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
            logger.error(f"❌ Erro na chamada OpenAI: {e}")
            return "", 0.0, 0.0
    
    def _call_huggingface(self, context: str) -> Tuple[str, float]:
        """Faz chamada para o modelo HuggingFace."""
        try:
            full_prompt = INSTRUCTION_PROMPT + context
            output = self.generator(
                full_prompt,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                temperature=TEMPERATURE,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            completion = self._extract_single_line(output[0]["generated_text"].strip())
            # HF não retorna logprobs por padrão, então confiança = 0.0
            return completion, 0.0, 0.0
        except Exception as e:
            logger.error(f"❌ Erro na chamada HuggingFace: {e}")
            return "", 0.0, 0.0
    
    def _compute_confidence(self, logprobs: List[Optional[float]]) -> float:
        """Calcula confiança baseada nos log-probabilities."""
        if not logprobs:
            return 0.0, 0.0

        valid_probs = [np.exp(lp) for lp in logprobs if lp is not None]
        if not valid_probs:
            return 0.0, 0.0

        mean_conf = float(np.mean(valid_probs))
        product_conf = float(np.prod(valid_probs))
        return mean_conf, product_conf
    
    def _extract_single_line(self, generated_text: str) -> str:
        """Extrai apenas a primeira linha não-vazia da geração."""
        if not generated_text:
            return ""
        
        # Remove instruções do prompt se ainda estiverem presentes
        if "Complete the Python function" in generated_text:
            # Encontra onde termina a instrução e começa a resposta
            lines = generated_text.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not any(keyword in line for keyword in 
                    ["Complete", "Return", "function", "single", "code", "explanations"]):
                    generated_text = '\n'.join(lines[i:])
                    break
        
        # Pega apenas a primeira linha não-vazia
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                return line
        
        # Se não encontrou nenhuma linha válida, retorna a primeira linha
        return lines[0].strip() if lines else ""

# =============================================================================
# PROCESSAMENTO DO DATASET
# =============================================================================

class DatasetProcessor:
    """Processa o dataset e prepara os prompts."""
    
    @staticmethod
    def load_dataset(name: str, split: str, min_executed: int) -> List[Dict[str, Any]]:
        """Carrega e filtra o dataset."""
        logger.info(f"⏳ Carregando dataset '{name}' (split={split})")
        
        try:
            ds = load_dataset(name, split=split)
            total_samples = len(ds)
            logger.info(f"📊 Total de amostras no dataset: {total_samples}")
            
            # Filtra amostras com execução suficiente
            filtered = [
                sample for sample in ds 
                if len(sample.get("executed_lines", [])) >= min_executed
            ]
            
            filtered_count = len(filtered)
            logger.info(f"📊 Amostras com >= {min_executed} linhas executadas: {filtered_count}")
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dataset: {e}")
            raise
    
    @staticmethod
    def prepare_prompt(source: str, executed_lines: List[int], placeholder: str, docstring: Optional[str] = None) -> Tuple[str, str]:
        """Prepara o prompt removendo uma linha e tudo que vem depois."""
        lines = source.split("\n")
        
        if not executed_lines:
            raise ValueError("Lista executed_lines está vazia")
        
        # Encontra linhas de código válidas (não comentários, não definição da função)
        valid_code_lines = DatasetProcessor._find_valid_code_lines(lines)
        
        if not valid_code_lines:
            raise ValueError("Nenhuma linha de código válida encontrada na função")
        
        # Seleciona uma linha aleatória entre as válidas
        target_idx = random.choice(valid_code_lines)
        
        # A linha que queremos prever é a linha no target_idx
        target_line = lines[target_idx].strip()
        
        # O contexto é tudo ANTES da linha target (sem incluí-la)
        context_lines = lines[:target_idx]
        context = "\n".join(context_lines)

        if docstring:
            docstring_cleaned = docstring.strip()
            if docstring_cleaned:
                context = f'"""{docstring_cleaned}"""\n\n' + context
        
        return context, target_line
    
    @staticmethod
    def _find_valid_code_lines(lines: List[str]) -> List[int]:
        """
        Encontra índices de linhas que são código válido para remoção.
        Exclui: definições de função, comentários, linhas vazias, docstrings.
        """
        valid_indices = []
        in_docstring = False
        docstring_delim = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Pula linhas vazias
            if not stripped:
                continue
            
            # Detecta início/fim de docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_delim = stripped[:3]
                    in_docstring = True
                    # Se a docstring termina na mesma linha
                    if stripped.count(docstring_delim) >= 2 and len(stripped) > 3:
                        in_docstring = False
                    continue
            else:
                # Estamos dentro de uma docstring
                if docstring_delim in stripped:
                    in_docstring = False
                continue
            
            # Pula se ainda estamos em docstring
            if in_docstring:
                continue
            
            # Pula comentários de linha única
            if stripped.startswith('#'):
                continue
            
            # Pula definições de função/classe
            if stripped.startswith('def ') or stripped.startswith('class '):
                continue
            
            # Pula decorators
            if stripped.startswith('@'):
                continue
            
            # Esta é uma linha de código válida
            valid_indices.append(i)
        
        return valid_indices

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def print_sample_header(sample_num: int, total: int):
    """Imprime cabeçalho para uma amostra."""
    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info(f"🔍 PROCESSANDO AMOSTRA {sample_num}/{total}")
    logger.info(f"{separator}")

def print_sample_details(context: str, target_line: str, predicted_line: str, is_correct: bool, line_number: int):
    """Imprime detalhes de uma amostra processada."""
    logger.info(f"🎯 LINHA ALVO PARA PREDIÇÃO (linha {line_number}):")
    logger.info(f"   '{target_line}'")
    logger.info(f"")
    logger.info(f"🤖 PREDIÇÃO DO MODELO:")
    logger.info(f"   '{predicted_line}'")
    logger.info(f"")
    logger.info(f"✅ RESULTADO: {'CORRETO' if is_correct else 'INCORRETO'}")
    logger.info(f"")
    logger.info(f"📋 CONTEXTO FORNECIDO AO MODELO:")
    for i, line in enumerate(context.split('\n'), 1):
        logger.info(f"   {i:2d}: {line}")
    logger.info(f"   {line_number:2d}: ??? ← PRÓXIMA LINHA A SER PREDITA")

def process_samples(samples: List[Dict[str, Any]], model: ModelManager) -> List[PredictionResult]:
    """Processa todas as amostras do dataset."""
    results = []
    processor = DatasetProcessor()
    
    logger.info(f"🚀 Iniciando processamento de {len(samples)} amostras")
    logger.info(f"⚙️  Configurações: backend={model.backend}, modelo={model.model_name}")
    logger.info(f"🎯 Modo: Predição da próxima linha (ao invés de preenchimento)")
    
    for idx, sample in enumerate(tqdm(samples, desc="Processando amostras")):
        try:
            print_sample_header(idx + 1, len(samples))
            
            # Prepara o prompt (contexto até antes da linha alvo)
            context, target_line = processor.prepare_prompt(
                sample.get("function", ""),
                sample.get("executed_lines", []),
                PLACEHOLDER,
                sample.get("docstring", "")
            )
            
            # Encontra o número da linha alvo
            all_lines = sample.get("function", "").split("\n")
            line_number = len(context.split('\n')) + 1  # Próxima linha após o contexto
            
            # Gera predição
            predicted_line, pavg, ptot = model.generate(context)
            
            # Avalia resultado
            is_correct = predicted_line.strip() == target_line.strip()
            
            # Imprime detalhes
            print_sample_details(context, target_line, predicted_line, is_correct, line_number)
            
            # Armazena resultado
            result = PredictionResult(
                sample_id=idx + 1,
                context=context,
                original_line=target_line,
                predicted_line=predicted_line,
                pavg=pavg,
                ptot=ptot,
                is_correct=is_correct
            )
            results.append(result)

            if (len(results) % 1000 == 0):
                output_path = f"{OUTPUT_FILE}_{len(results)}.json" 
                save_results(results, output_path)

        except Exception as e:
            print(f"[ERRO] Falha ao processar amostra {idx + 1}: {e}")

    return results


def save_results(results: List[PredictionResult], filepath: str):
    """Salva os resultados em arquivo JSON."""
    logger.info(f"💾 Salvando {len(results)} resultados em '{filepath}'")
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([result.to_dict() for result in results], f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Resultados salvos com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar resultados: {e}")
        raise

def print_summary(results: List[PredictionResult]):
    """Imprime resumo dos resultados."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info(f"📊 RESUMO DOS RESULTADOS")
    logger.info(f"{separator}")
    logger.info(f"Total de amostras processadas: {total}")
    logger.info(f"Predições corretas: {correct}")
    logger.info(f"Predições incorretas: {total - correct}")
    logger.info(f"Acurácia: {accuracy:.1f}%")
    logger.info(f"{separator}")

def main():
    """Função principal."""
    try:
        logger.info("🎯 Iniciando avaliação do modelo de código")
        
        # Inicializa modelo
        model = ModelManager(LLM_BACKEND, LLM_MODEL, HF_DEVICE)
        
        # Carrega dataset
        processor = DatasetProcessor()
        samples = processor.load_dataset(DATASET_NAME, DATASET_SPLIT, MIN_EXECUTED_LINES)
        
        # Amostragem aleatória
        random.shuffle(samples)
        if SAMPLE_SIZE:
            samples = samples[:SAMPLE_SIZE]
            logger.info(f"🎲 Usando amostra aleatória de {len(samples)} exemplos")
        
        # Processa amostras
        results = process_samples(samples, model)
        
        # Salva resultados
        # save_results(results, OUTPUT_FILE)
        
        # Imprime resumo
        print_summary(results)
        
        logger.info("🏁 Processamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        raise

if __name__ == "__main__":
    main()