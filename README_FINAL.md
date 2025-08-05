# RAG para Calibração de Confiança em Autocompletadores de Código

## 🎯 **Objetivo**

Este projeto investiga métodos de **RAG (Retrieval-Augmented Generation)** para melhorar a **calibração de confiança** em modelos de completamento de código. O foco é fazer com que os modelos sejam mais precisos ao estimar sua própria confiança nas predições, um aspecto crucial para ferramentas de programação assistida em ambientes reais.

### **Questão de Pesquisa Principal**
**RQ4: Few-shot prompting melhora a calibração reflexiva da confiança do modelo?**

## 🏗️ **Arquitetura do Sistema**

O projeto implementa um pipeline completo de avaliação de calibração de confiança:

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   Dataset       │    │   Geração de     │    │   Calibração       │
│   Processing    │───▶│   Predições      │───▶│   Reflexiva        │
│   (step1.py)    │    │   Base           │    │   (step2.py)       │
└─────────────────┘    └──────────────────┘    └────────────────────┘
                                                           │
                                                           ▼
                                               ┌────────────────────┐
                                               │   Análise e        │
                                               │   Visualização     │
                                               │   (calibration_    │
                                               │   analyzer.py)     │
                                               └────────────────────┘
```

### **Componentes Principais:**

1. **Geração Base** - Produz completamentos com métricas de confiança intrínsecas
2. **Sistema RAG** - Implementa múltiplas estratégias de retrieval (Random, BM25, CodeBERT)
3. **Calibração Reflexiva** - Prompts que induzem auto-avaliação de confiança
4. **Reescalonamento** - Isotonic Regression e Platt Scaling para calibração pós-hoc
5. **Análise** - Métricas ECE, Brier Score, Skill Score com visualizações

## 📊 **Dataset e Metodologia**

### **Dataset: DyPyBench**
- **50 projetos reais** de código aberto em Python
- Cada exemplo contém:
  - **Função completa** com uma linha removida
  - **Docstring** e contexto
  - **Testes automatizados** (ground truth para corretude)

### **Modelo Utilizado**
- **DeepSeek Coder 1.3B** (`deepseek-ai/deepseek-coder-1.3b-base`)
- Modelo instruction-tuned otimizado para tarefas de código
- Suporte para extração de logprobs para métricas de confiança

### **Estratégias de Retrieval Testadas**
1. **0-shot**: Sem exemplos few-shot
2. **Random**: Seleção aleatória de exemplos
3. **BM25**: Similaridade léxica baseada em TF-IDF
4. **RAG CodeBERT**: Similaridade semântica com embeddings especializados

## 🚀 **Como Reproduzir o Experimento**

### **Pré-requisitos**

```bash
# Dependências principais
pip install torch transformers datasets
pip install sentence-transformers faiss-cpu
pip install rank-bm25 scikit-learn
pip install matplotlib numpy tqdm
```

### **Estrutura de Arquivos**
```
TAES-2025.1/
├── config.py                    # Configurações centralizadas
├── data_structures.py           # Estruturas de dados
├── models.py                    # Gerenciamento de modelos
├── dataset_processor.py         # Processamento de datasets
├── evaluator.py                 # Avaliação de modelos
├── step1.py                     # Pipeline geração inicial
├── step2.py                     # Sistema RAG e calibração
├── calibration_analyzer_pavg.py # Análise e visualização
├── results/                     # Resultados experimentais
└── plots/                       # Gráficos de calibração
```

### **Execução Passo a Passo**

#### **Etapa 1: Geração de Predições Base**
```bash
# Gera predições iniciais com métricas de confiança intrínsecas
python step1.py
```

**O que acontece:**
- Carrega dataset DyPyBench filtrado (≥2 linhas executadas)
- Para cada função, remove uma linha aleatória
- Modelo gera completamento da linha
- Calcula métricas de confiança (`pavg`, `ptot`) via logprobs
- Salva resultados em `results/deepseek_predictions.json`

#### **Etapa 2: Calibração Reflexiva com RAG**
```bash
# Aplica diferentes estratégias RAG para calibração reflexiva
python step2.py
```

**O que acontece:**
- Carrega predições da Etapa 1
- Para cada estratégia (0-shot, random, BM25, RAG):
  - Seleciona k=5 exemplos few-shot
  - Monta prompts reflexivos
  - Avalia confiança via "True/False" e scores verbalizados
  - Calcula métricas de calibração
- Salva resultados em `results/confidence_results_*.json`

#### **Etapa 3: Análise e Visualização**
```bash
# Gera reliability plots e aplica reescalonamento
python calibration_analyzer_pavg.py
```

**O que acontece:**
- Carrega resultados da calibração
- Aplica Isotonic Regression e Platt Scaling
- Gera reliability plots (confiança vs acurácia)
- Calcula métricas finais (ECE, Brier, Skill Score)
- Salva plots em `plots/`

### **Configuração Personalizada**

#### **Modificar Parâmetros em `config.py`:**
```python
# Dataset
DATASET_NAME = "claudios/dypybench_functions"
SAMPLE_SIZE = 100  # Para experimentos rápidos (None = todos)

# Modelo
LLM_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
MAX_TOKENS = 32
TEMPERATURE = 0  # Geração determinística

# Few-shot
k_shots = 5  # Número de exemplos (modifique em step2.py)
```

#### **Testar Outros Modelos:**
```python
# Em config.py, altere:
LLM_BACKEND = "openai"  # Para usar GPT
LLM_MODEL = "gpt-3.5-turbo-instruct"
OPENAI_API_KEY = "sua_chave_aqui"
```

## 📈 **Métricas de Avaliação**

### **1. Expected Calibration Error (ECE)**
Mede a diferença entre confiança prevista e acurácia real:
```
ECE = Σ (|bin_size| / n) × |accuracy - confidence|
```

### **2. Brier Score**
Erro quadrático médio das probabilidades:
```
Brier = (1/N) × Σ (p_predicted - y_true)²
```

### **3. Skill Score**
Melhoria sobre baseline aleatório:
```
SS = (Brier_baseline - Brier_model) / Brier_baseline
```

## 📊 **Resultados Esperados**

### **Arquivos de Saída:**
- `results/deepseek_predictions.json`: Predições base
- `results/confidence_results_*.json`: Resultados por estratégia RAG
- `plots/*_reliability_*.png`: Reliability plots
- `metricas_calibracao_*.txt`: Métricas sumarizadas

### **Formato dos Resultados:**
```json
{
  "sample_id": 123,
  "context": "def function():\n    # código...",
  "predicted_line": "return result",
  "original_line": "return result", 
  "pavg": 0.85,         // Confiança intrínseca (média)
  "ptot": 0.72,         // Confiança intrínseca (produto)
  "pv": 0.78,           // Confiança verbalizada
  "ask_tf": 0.82,       // Probabilidade True/False
  "ask_tf_n": 0.79,     // Normalizada (para calibração)
  "is_correct": true    // Ground truth via testes
}
```

## 🎯 **Principais Inovações**

1. **Calibração Reflexiva**: Modelo estima própria confiança via self-questioning
2. **RAG Especializado**: CodeBERT para similaridade semântica de código  
3. **Avaliação Sistemática**: Comparação de múltiplas estratégias de retrieval
4. **Reescalonamento**: Técnicas pós-hoc para melhorar calibração
5. **Ground Truth Rigoroso**: Execução de testes para validação de corretude

## 🔬 **Experimentos e Análises**

### **Hipóteses Testadas:**
- **H1**: RAG com CodeBERT > BM25 > Random para calibração
- **H2**: Few-shot prompting melhora calibração vs 0-shot
- **H3**: Reescalonamento (Isotonic/Platt) melhora métricas finais

### **Visualizações Geradas:**
- **Reliability Plots**: Confiança vs Acurácia por bins
- **Histogramas**: Distribuição de scores de confiança
- **Comparações**: Métricas antes/depois do reescalonamento

## 👥 **Equipe**

- José Vinicius de Santana Souza ([jvss2](https://github.com/jvss2))
- Camila Barbosa Vieira ([cbv2](https://github.com/cbv2))  
- Rafael dos Reis Labio ([rrl3](https://github.com/rrl3))
- Matheus Lafayette Vasconcelos ([MLV](https://github.com/MLV))

## 📝 **Citação**

```bibtex
@misc{rag-calibracao-codigo-2025,
  title={RAG para Calibração de Confiança em Autocompletadores de Código},
  author={Souza, José Vinicius and Vieira, Camila Barbosa and Labio, Rafael dos Reis and Vasconcelos, Matheus Lafayette},
  year={2025},
  note={Projeto da disciplina Inovação com LLMs}
}
```

## 🛠️ **Solução de Problemas**

### **Erro de Memória GPU:**
```python
# Em config.py, reduza batch size ou use CPU:
HF_DEVICE = "cpu"  # Força uso de CPU
```

### **Timeout de API:**
```python
# Adicione delays entre chamadas no step2.py
import time
time.sleep(1)  # Entre requests
```

### **Dataset não encontrado:**
```bash
# Verifique conectividade com HuggingFace Hub
huggingface-cli login
```

---

**Este projeto demonstra como RAG pode melhorar não apenas a qualidade das predições, mas também a confiabilidade das estimativas de confiança em modelos de código - um aspecto crucial para adoção em ferramentas reais de programação assistida.**