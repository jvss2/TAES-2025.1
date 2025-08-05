## Objetivo Geral do Artigo
Melhorar a **calibração da confiança** de LLMs em tarefas de **code generation**, especialmente **code completion**.

## RQ4 - **Few-shot prompting melhora a calibração reflexiva da confiança do modelo?**

### Tarefa Avaliada — Line-Level Code Completion

### Dataset: DYPYBENCH

- DYPYBENCH é um benchmark criado a partir de **50 projetos reais de código aberto em Python**.
- Cada exemplo é uma **função** de projeto real, com:
  - Um **corpo completo**
  - Uma linha removida (para ser prevista pelo modelo)
  - Uma **docstring** (comentário de alto nível)
  - Um conjunto de **testes automatizados** já prontos (usados como ground truth)

### Estrutura dos dados

Cada item inclui:
- `context`: a função com uma linha removida
- `target_line`: a linha que deve ser prevista (ground truth, usada só para avaliação ou como referência)
- `tests`: testes automatizados para a função (unitários, via `pytest` ou `unittest`)

### Importante: como medir corretude?
A **corretude** de uma *completion* (linha gerada pelo modelo) é **avaliada executando os testes** da função com a linha completada.

```python
# True se passou todos os testes. False se algum falhar.
is_correct = run_tests_with_generated_line(generated_line, function_context, test_suite)
```

Esse valor binário é usado como **ground truth** para:

* Comparar com a **reflexão do modelo** (se ele disse que estava certo ou não)
* Treinar ou selecionar exemplos few-shot (`prompt, completion, True/False`)

---

### Modelo: GPT-3.5 (gpt-3.5-turbo-instruct)

Usamos **GPT-3.5-turbo-instruct** porque:

* É um modelo **instruction-tuned**: responde bem a perguntas reflexivas do tipo “Esse código está certo?”
* Tem bom desempenho em **code completion**
* É **suficientemente grande** para se beneficiar de few-shot prompting

---

## Como Executar o Experimento (Etapas Detalhadas)

### 1. Preparar a Base de Dados

Para cada exemplo no `DYPYBENCH`:

* Remova a linha de destino do código (`target_line`)
* Salve o `context` da função como entrada do modelo

### 2. Gerar uma Completion (linha de código)

Com o `context`, peça ao modelo:

> Complete a linha que falta nesta função Python:

Salve:

* `completion_gerada`
* `contexto_original`

### 3. Avaliar Corretude com os Testes

* Reinsira a linha gerada no lugar correto
* Execute os testes automatizados associados à função
* Se **todos os testes passarem** → `True`
* Senão → `False`

Exemplo em Python:

```python
def is_correct(completion: str, context: str, tests: List[str]) -> bool:
    full_code = insert_line(context, completion)
    return run_tests(full_code, tests)
```

---

### 4. Perguntar ao Modelo (Reflexão T/F)

Agora, usando a mesma `completion_gerada`, pergunte ao modelo:

```text
Dado o seguinte código gerado, ele está correto? Responda apenas com True ou False.

<completion_gerada>
```

Essa é a confiança **estimada** do modelo.

#### Código 

```python
import openai
import numpy as np

openai.api_key = "YOUR_API_KEY"  # Substitua pela sua chave da OpenAI

def perguntar_reflexao_pnb(completion_gerada: str):
    prompt = (
        "Dado o seguinte código gerado, ele está correto?\n"
        "Responda apenas com True ou False.\n\n"
        f"{completion_gerada}\n"
    )

    response = openai.Completion.create(
        model="text-davinci-003",  # ou outro modelo compatível com logprobs
        prompt=prompt,
        temperature=0,
        max_tokens=1,
        logprobs=5,
        echo=False,  # não repetir o prompt na resposta
    )

    choice = response["choices"][0]
    token = choice["text"].strip()
    logprobs = choice["logprobs"]["top_logprobs"][0]

    # Extrair logprobs das respostas possíveis
    logprob_true = logprobs.get("True", -float("inf"))
    logprob_false = logprobs.get("False", -float("inf"))

    # Converter para probabilidade normalizada
    prob_true = np.exp(logprob_true)
    prob_false = np.exp(logprob_false)
    pnb = prob_true / (prob_true + prob_false) if (prob_true + prob_false) > 0 else 0.5

    # Binário estimado
    reflexao_binaria = "True" if "true" in token.lower() else "False" if "false" in token.lower() else "Unknown"

    return {
        "resposta": reflexao_binaria,
        "p_true": pnb,
        "token": token,
        "logprobs": logprobs
    }

# Exemplo de uso
completion = "self.jobs[:] = [job for job in self.jobs if job.tag != tag]"
resultado = perguntar_reflexao_pnb(completion)

print("Reflexão do modelo:", resultado["resposta"])
print("Probabilidade (pNB):", round(resultado["p_true"], 3))
print("Token retornado:", resultado["token"])
print("Logprobs:", resultado["logprobs"])
```

#### Notas

* `temperature=0` para respostas determinísticas
* A resposta deve ser só `"True"` ou `"False"`, mas o código lida com variações tipo `" true"` ou `"False."`
* Em produção, recomenda-se fazer **log de todas as respostas brutas** para auditar possíveis erros

---

### 5. Criar Dataset de Exemplos Few-Shot

Cada exemplo será um triplet:

```json
{
  "prompt": "<contexto>",
  "completion": "<linha gerada>",
  "reflection": "True" | "False"  // ground truth baseado nos testes
}
```

Você precisa de uma coleção disso para montar os **few-shots**.

---

### 6. Executar Few-Shot Reflective Prompting

Para avaliar um novo exemplo:

* Pegue 5 exemplos do banco (Random, BM25, RAG)
* Monte o prompt concatenando os 5 exemplos no formato:

```
Prompt 1
Completion 1
Reflexão: True

...

Prompt N
Completion N
Reflexão: False

---

Novo Prompt:
Completion:
Reflexão:
```

* Peça ao modelo completar a **reflexão** da nova *completion*

---

### 7. Avaliar Calibração

Para cada novo exemplo:

* Você tem a **resposta reflexiva do modelo**
* Você tem o **ground truth via testes**
* Compare os dois e calcule:

| Métrica     | Como calcular                                 |
| ----------- | --------------------------------------------- |
| Brier Score | Erro quadrático entre `p(model)` e `correct`  |
| Skill Score | Quanto melhora sobre o baseline aleatório     |
| ECE         | Diferença média entre confiança e acerto real |

#### 1. **Brier Score**

**O que mede:**
Diferença quadrática entre a **confiança** do modelo e o valor real (`True` = 1, `False` = 0). Mede quão boas são as previsões probabilísticas.

**Fórmula:**

$$
B = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

* `p_i` = confiança do modelo (ex: 0.75)
* `y_i` = ground truth (`1` se correto, `0` se errado)

**Código:**

```python
from sklearn.metrics import brier_score_loss

# Ground truth: se o código está certo (via testes)
y_true = [1, 0, 1, 1, 0]

# Confiança reflexiva do modelo (probabilidade de "True")
y_prob = [0.9, 0.3, 0.8, 0.95, 0.2]

brier = brier_score_loss(y_true, y_prob)
print("Brier Score:", brier)
```

---

#### 2. **Skill Score (SS)**

**O que mede:**
Compara o desempenho do modelo com um **baseline ingênuo**, que sempre prevê a média de acertos (base rate).
Quanto maior, melhor.

**Fórmula:**

$$
SS = \frac{B_{\text{ref}} - B_{\text{model}}}{B_{\text{ref}}}
$$

* $B_{\text{ref}} = p(1 - p)$, onde `p` = taxa média de acertos no dataset

**Código:**

```python
import numpy as np

# Ground truth
y_true = np.array([1, 0, 1, 1, 0])

# Modelo
y_prob = np.array([0.9, 0.3, 0.8, 0.95, 0.2])

# Brier do modelo
brier_model = brier_score_loss(y_true, y_prob)

# Brier do baseline (sempre chutar p = média de y_true)
p_base = y_true.mean()
brier_ref = p_base * (1 - p_base)

# Skill Score
skill_score = (brier_ref - brier_model) / brier_ref

print(f"Skill Score: {skill_score:.3f}")
```

---

#### 3. **Expected Calibration Error (ECE)**

**O que mede:**
Quanto, em média, a **confiança prevista** difere da **frequência real de acerto**, em cada faixa de confiança.

**Como funciona:**

1. Divide os exemplos em **bins** por faixa de confiança (ex: 0.0–0.1, 0.1–0.2, ..., 0.9–1.0)
2. Para cada bin, compara:

   * média da confiança dos exemplos (`conf`)
   * frequência real de acerto (`acc`)
3. Calcula o erro médio ponderado:

$$
ECE = \sum_{i=1}^{m} \frac{|S_i|}{n} \cdot |\text{acc}(S_i) - \text{conf}(S_i)|
$$

**Código:**

```python
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
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

# Exemplo
y_true = [1, 0, 1, 1, 0]
y_prob = [0.9, 0.3, 0.8, 0.95, 0.2]

ece = compute_ece(y_true, y_prob)
print("ECE:", ece)
```

---

#### Observações

* **Brier** penaliza erros com mais severidade (é um erro quadrático).
* **Skill Score** mostra o quanto seu modelo melhora em relação ao puro chute.
* **ECE** mostra a **calibração**, ou seja, se a confiança prevista bate com a frequência real.