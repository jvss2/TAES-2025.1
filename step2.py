import re
import faiss
import torch
import openai
import random
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def avaliar_reflexao(completion, context, passou_testes, selector=None, modelo="openai", modelo_args=None):
    """
    Avalia reflexão sobre uma completion usando o modelo especificado.
    
    modelo: "openai" ou "hf"
    modelo_args: dicionário com parâmetros extras (ex: tokenizer, caminho do modelo, etc)
    """
    prompt = montar_prompt_reflexivo(context, completion, selector)

    if modelo == "openai":
        return avaliar_reflexao_openai(prompt, completion, context, passou_testes, **(modelo_args or {}))
    elif modelo == "hf":
        return avaliar_reflexao_hf(prompt, completion, context, passou_testes, **(modelo_args or {}))
    else:
        raise ValueError(f"Modelo não suportado: {modelo}")

def montar_prompt_reflexivo(context: str, completion: str, selector=None) -> str:
    few_shot_exemplos = selector(context, completion) if selector else []

    prompt_parts = []

    for exemplo in few_shot_exemplos:
        prompt_parts.append(
            f"{exemplo['context']}\n{exemplo['completion']}\nReflexão: {exemplo['reflexao']}"
        )

    prompt_parts.append("---")
    if not few_shot_exemplos:
        prompt_parts.append("Is the code below correct? Answer only with True or False.\n")

    prompt_parts.append(f"{context}\n{completion}\nReflexão:")

    return "\n\n".join(prompt_parts)

def selector_random(context, completion):
    exemplos = random.sample(dataset, k_shots)
    return [
        {
            "context": ex["context"],
            "completion": ex["completion"],
            "reflexao": "True" if ex["passou_testes"] else "False"
        }
        for ex in exemplos
    ]

def selector_bm25(context, completion):
    # Preprocessa e indexa
    corpus = [re.findall(r"\w+",  (ex["context"] + "\n" + ex["completion"]).lower()) for ex in dataset]
    bm25 = BM25Okapi(corpus)
    query = re.findall(r"\w+", (context + "\n" + completion).lower())
    scores = bm25.get_scores(query)

    # Top-k índices mais relevantes
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_shots]
    return [
        {
            "context": dataset[i]["context"],
            "completion": dataset[i]["completion"],
            "reflexao": "True" if dataset[i]["passou_testes"] else "False"
        }
        for i in top_k_indices
    ]

def preparar_index_rag():
    model = SentenceTransformer(RAG_model_name)
    contexts = [ex["context"] + "\n" + ex["completion"] for ex in dataset]
    embeddings = model.encode(contexts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

def selector_rag(context, completion):
    query_text = context + "\n" + completion
    query_vec = RAG_model.encode([query_text])[0].reshape(1, -1)
    _, I = RAG_index.search(query_vec, k_shots)
    return [
        {
            "context": dataset[i]["context"],
            "completion": dataset[i]["completion"],
            "reflexao": "True" if dataset[i]["passou_testes"] else "False"
        }
        for i in I[0]
    ]

def preparar_index_rag_code():
    tokenizer = AutoTokenizer.from_pretrained(RAG_code_model_name)
    model = AutoModel.from_pretrained(RAG_code_model_name)
    model.eval()

    contexts = [ex["context"] + "\n" + ex["completion"] for ex in dataset]
    embeddings = []

    with torch.no_grad():
        for text in contexts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            # CLS token como embedding
            cls_embedding = outputs.last_hidden_state[0][0].numpy()
            embeddings.append(cls_embedding)

    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, tokenizer, model

def selector_rag_code(context, completion):
    query_text = context + "\n" + completion
    inputs = RAG_code_tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = RAG_code_model(**inputs)
        query_embedding = outputs.last_hidden_state[0][0].numpy().reshape(1, -1)

    _, I = RAG_code_index.search(query_embedding, k_shots)

    return [
        {
            "context": dataset[i]["context"],
            "completion": dataset[i]["completion"],
            "reflexao": "True" if dataset[i]["passou_testes"] else "False"
        }
        for i in I[0]
    ]

def avaliar_reflexao_openai(prompt, completion, context, passou_testes, model="text-davinci-003", api_key=None):
    if api_key:
        openai.api_key = api_key

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=1,
        logprobs=5,
        echo=False,
    )

    choice = response["choices"][0]
    token = choice["text"].strip()
    logprobs = choice["logprobs"]["top_logprobs"][0]

    logprob_true = logprobs.get("True", -float("inf"))
    logprob_false = logprobs.get("False", -float("inf"))

    prob_true = np.exp(logprob_true)
    prob_false = np.exp(logprob_false)
    pnb = prob_true / (prob_true + prob_false) if (prob_true + prob_false) > 0 else 0.5

    reflexao_binaria = "True" if "true" in token.lower() else "False" if "false" in token.lower() else "Unknown"

    return {
        "prompt": prompt,
        "resposta": reflexao_binaria,
        "p_true": pnb,
        "passou_testes": passou_testes,
        "completion": completion,
        "context": context,
        "token": token,
        "logprob_true": logprob_true,
        "logprob_false": logprob_false
    }

def avaliar_reflexao_hf(
    prompt: str,
    completion: str,
    context: str,
    passou_testes: bool,
    model_path: str = "tiiuae/falcon-7b-instruct",
    tokenizer=None,
    model=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Avalia a reflexão usando um modelo HuggingFace.
    
    Retorna a resposta binária, pNB, e informações úteis.
    """
    print(prompt)

    # Carrega tokenizer e modelo, se não fornecidos
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    # Tokenização
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Geração com logprobs
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=0.0
        )

    # Token gerado
    generated_token_id = output.sequences[0][-1].item()
    generated_token = tokenizer.decode(generated_token_id).strip()

    # Score do próximo token
    scores = output.scores[0][0]  # logits da primeira posição gerada

    # Verifica se o vocabulário tem os tokens "True" e "False"
    true_id = tokenizer.convert_tokens_to_ids("True")
    false_id = tokenizer.convert_tokens_to_ids("False")

    logprob_true = scores[true_id].item() if true_id is not None else -float("inf")
    logprob_false = scores[false_id].item() if false_id is not None else -float("inf")

    # Normalização
    prob_true = np.exp(logprob_true)
    prob_false = np.exp(logprob_false)
    pnb = prob_true / (prob_true + prob_false) if (prob_true + prob_false) > 0 else 0.5

    # Interpretação do token
    resposta_binaria = "True" if "true" in generated_token.lower() else "False" if "false" in generated_token.lower() else "Unknown"
    print(pnb, resposta_binaria)

    return {
        "prompt": prompt,
        "resposta": resposta_binaria,
        "p_true": pnb,
        "passou_testes": passou_testes,
        "completion": completion,
        "context": context,
        "token": generated_token,
        "logprob_true": logprob_true,
        "logprob_false": logprob_false
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


dataset = [
    {
        "context": "def add(a, b):\n    # Soma dois números\n    ",
        "completion": "return a + b",
        "passou_testes": True
    },
    {
        "context": "def divide(a, b):\n    # Retorna a divisão\n    if b == 0:\n        ",
        "completion": "raise ValueError('Divisão por zero')",
        "passou_testes": True
    },
    {
        "context": "def get_first(lst):\n    # Retorna o primeiro elemento de uma lista\n    ",
        "completion": "return lst[0]",
        "passou_testes": True
    },
    {
        "context": "def invert_case(s):\n    # Inverte maiúsculas e minúsculas\n    ",
        "completion": "return s.swapcase()",
        "passou_testes": True
    },
    {
        "context": "def add(a, b):\n    # Soma dois números\n    ",
        "completion": "return a - b",  # Errado propositalmente
        "passou_testes": False
    },
    {
        "context": "def square_elements(lst):\n    # Retorna uma nova lista com os elementos ao quadrado\n    ",
        "completion": "return [x ** 2 for x in lst]",
        "passou_testes": True
    },
    {
        "context": "def is_palindrome(s):\n    # Verifica se a string é um palíndromo\n    ",
        "completion": "return s == s[::-1]",
        "passou_testes": True
    },
    {
        "context": "def get_even_numbers(nums):\n    # Filtra números pares\n    ",
        "completion": "return [n for n in nums if n % 2 == 0]",
        "passou_testes": True
    },
    {
        "context": "def multiply(a, b):\n    # Multiplica dois números\n    ",
        "completion": "return a + b",  # incorreto propositalmente
        "passou_testes": False
    },
    {
        "context": "def get_last(lst):\n    # Retorna o último elemento da lista\n    ",
        "completion": "return lst[-1]",
        "passou_testes": True
    }
]

k_shots = 3
selector = selector_rag_code
RAG_model_code = False
if selector == selector_rag:
    RAG_model_name = "all-MiniLM-L6-v2"
    RAG_index, RAG_model = preparar_index_rag()
if selector == selector_rag_code:
    RAG_code_model_name = "microsoft/codebert-base"
    RAG_code_index, RAG_code_tokenizer, RAG_code_model = preparar_index_rag_code()

# Rodar reflexão sobre todos os exemplos
avaliacoes = [
    avaliar_reflexao(
        completion=ex["completion"],
        context=ex["context"],
        passou_testes=ex["passou_testes"],
        selector=selector,  
        modelo="hf",
        modelo_args={"model_path": "Salesforce/codegen-350M-mono"}
        # modelo_args={"api_key": "sua_chave"}
    )
    for ex in dataset
]
#"tiiuae/falcon-7b-instruct", "Salesforce/codegen-350M-mono", "TheBloke/WizardCoder‑Python‑7B‑V1.0‑AWQ", 

df_resultados = pd.DataFrame([
    {
        "y_prob": av["p_true"].astype(int),
        "y_true": av["passou_testes"],
    }
    for av in avaliacoes
])

y_true = df_resultados["y_true"].tolist()
y_prob = df_resultados["y_prob"].tolist()

brier = brier_score(y_true, y_prob)
skill = skill_score(y_true, y_prob)
ece = expected_calibration_error(y_true, y_prob)

print("\nMétricas de Calibração:")
print(f"- Brier Score: {brier:.4f}")
print(f"- Skill Score: {skill:.4f}")
print(f"- Expected Calibration Error (ECE): {ece:.4f}")

with open(f"metricas_calibracao_{selector}.txt", "w") as f:
    f.write(f"Brier Score: {brier:.4f}\n")
    f.write(f"Skill Score: {skill:.4f}\n")
    f.write(f"Expected Calibration Error (ECE): {ece:.4f}\n")

def plot_calibration_with_hist(y_true, y_prob, n_bins=10, filename="calibration_plot_hist.png"):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_bounds = np.linspace(0, 1, n_bins + 1)

    accs, confs, counts = [], [], []

    for i in range(n_bins):
        low, high = bin_bounds[i], bin_bounds[i+1]
        in_bin = (y_prob >= low) & (y_prob < high)
        if in_bin.any():
            accs.append(np.mean(y_true[in_bin]))
            confs.append(np.mean(y_prob[in_bin]))
            counts.append(np.sum(in_bin))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Calibração
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    ax1.plot(confs, accs, marker='o', label='Modelo')
    ax1.set_ylabel("Frequência real de acerto")
    ax1.set_xticks(np.linspace(0, 1, n_bins + 1))
    ax1.set_title("Gráfico de Calibração com Histograma")
    ax1.grid(True)
    ax1.legend()

    # Histograma
    bin_centers = (bin_bounds[:-1] + bin_bounds[1:]) / 2
    ax2.bar(bin_centers, counts, width=1/n_bins, align='center', edgecolor='black')
    ax2.set_xlabel("Confiança prevista (p_true)")
    ax2.set_ylabel("Contagem")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_calibration_with_hist(y_true, y_prob, filename=f"calibration_plot_{selector}.png")