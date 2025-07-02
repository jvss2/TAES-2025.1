# RAG para Calibração de Confiança em Autocompletadores de Código

## 🎯 Objetivo

Investigar um método de RAG (Retrieval-Augmented Generation) mais eficaz para calibrar a confiança de modelos de completamento de código. Resultados preliminares mostram que a seleção inteligente de exemplos — como via BM25 — já melhora significativamente a calibração reflexiva, especialmente quando combinada com reescalonamento. Acreditamos que abordagens baseadas em embeddings semânticos especializados em código podem superar esse baseline.

## 🧩 Atividade do Ciclo de Vida Envolvida

**Implementação e Evolução de Modelos de Assistência à Programação** — esta atividade envolve melhorar a confiabilidade de ferramentas de autocompletamento de código, com foco em usabilidade, confiança e segurança em ambientes reais.

## 🤖 Proposta com LLM e RAG

Propomos desenvolver um pipeline RAG com:

- **Embeddings Semânticos Específicos de Código**: Usaremos CodeBERT, GraphCodeBERT ou embeddings treinados in-house.
- **Re-rankers Neurais**: Para ordenar os exemplos recuperados com base na similaridade contextual real.
- **Componente de calibração reflexiva**: O modelo será induzido a estimar sua própria confiança após receber sugestões baseadas nos exemplos mais relevantes.

Essa abordagem fornecerá ao modelo instruções mais alinhadas ao problema atual, permitindo julgamentos reflexivos mais confiáveis.

## 🧪 Experimentos

### Vantagens Esperadas

- **Cenário:** Completamento de código com suporte a calibragem reflexiva.
- **Método A (Controle):** RAG com BM25 e sem re-rankers.
- **Método B (Proposto):** RAG com embeddings especializados e re-rankers.
- **Métricas:** ECE (Expected Calibration Error), cobertura, F1 do completamento, acurácia da autoconfiança.

<!-- ### Limitações Investigadas

- Robustez a exemplos ruidosos no repositório.
- Dependência da qualidade dos embeddings.
- Custo computacional adicional do re-ranking.

## 📦 Reprodutibilidade

- Usaremos datasets públicos (ex: HumanEval, CodeXGLUE).
- Embeddings e repositórios base serão documentados.
- Scripts de recuperação, re-ranking e avaliação serão versionados.

## 🛠️ Como Rodar o Pipeline (exemplo futuro)

1. Clone este repositório:

   ```bash
   git clone https://github.com/jvss2/rag-calibracao-codigo.git
   cd rag-calibracao-codigo
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o pipeline:

   ```bash
   python run_pipeline.py --dataset humaneval
   ```

## 📁 Estrutura do Repositório

```
.
├── README.md
├── run_pipeline.py
├── embeddings/
│   └── codebert/
├── retrieval/
│   ├── bm25.py
│   └── reranker.py
├── evaluation/
│   └── metrics.py
└── resultados/
    ├── ece_comparativo.csv
    └── analise_qualidade.md
``` -->

## 👥 Equipe

- José Vinicius de Santana Souza ([jvss2](https://github.com/jvss2))
- Camila Barbosa Vieira ([cbv2](https://github.com/cbv2))
- Rafael dos Reis Labio ([rrl3](https://github.com/rrl3))
- Matheus Lafayette Vasconcelos ([MLV](https://github.com/MLV))

---

Este projeto é parte da disciplina de Inovação com LLMs e busca ampliar a segurança e aplicabilidade de modelos generativos para assistentes de programação.

