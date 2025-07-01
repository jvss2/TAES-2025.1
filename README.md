# Gerador de Documentação para Novos Laboratórios

## 🎯 Objetivo

Este projeto propõe o uso de Large Language Models (LLMs) para automatizar a criação de documentação inicial (README.md) de novos laboratórios ou componentes de software. A iniciativa visa melhorar a qualidade da documentação, reduzir o esforço necessário para produzi-la e padronizar a experiência do desenvolvedor (DevEx).

## 🗉 Atividade do Ciclo de Vida Envolvida

**Documentação de Software** – atividade fundamental para facilitar o uso, manutenção e extensão de sistemas. Este projeto foca em como LLMs podem contribuir para essa tarefa, especialmente em contextos educacionais ou de código aberto, onde a documentação costuma ser negligenciada.

## 🤖 Proposta com LLM

Utilizamos um script que, a partir do código-fonte de um laboratório, gera um rascunho inicial de documentação (README.md). O objetivo é que o desenvolvedor apenas revise e edite esse rascunho, em vez de começar do zero.

## 🧪 Experimentos

### Vantagens (Produtividade e Qualidade)

- **Cenário:** Gerar documentação para um novo laboratório.
- **Método A (Controle):** Contribuidor escreve README.md do zero.
- **Método B (Experimental):** O mesmo contribuidor usa o script com LLM para gerar um rascunho e edita o texto.
- **Métricas:** Tempo total gasto e avaliação qualitativa da documentação final (clareza, cobertura, organização).

### Limitações (Completude)

Análise da documentação gerada por LLM para três laboratórios com níveis de complexidade distintos, visando identificar padrões de omissão (ex: detalhes de configuração, dependências específicas, instruções de instalação).

<!-- ## 📦 Reprodutibilidade

- Todos os laboratórios utilizados são públicos.
- O script de geração e o prompt LLM estão neste repositório.
- Os critérios de avaliação são descritos em um formulário incluído aqui.
- As instruções para rodar os experimentos estão na seção abaixo. -->

<!-- ## 🛠️ Como Rodar o Script

1. Clone este repositório:

   ```bash
   git clone https://github.com/jvss2/gerador-documentacao-labs.git
   cd gerador-documentacao-labs
   ```

2. Instale as dependências (exemplo para Python):

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o script de geração:

   ```bash
   python gerar_documentacao.py --path ./laboratorios/lab1
   ```

4. O rascunho será salvo como `README_GERADO.md` no diretório do laboratório.

## 📁 Estrutura do Repositório

```
.
├── README.md
├── gerar_documentacao.py
├── prompt.txt
├── avaliacao/
│   └── formulario.md
├── laboratorios/
│   ├── lab1/
│   ├── lab2/
│   └── lab3/
└── resultados/
    ├── tempo_comparativo.csv
    └── analise_qualitativa.md
``` -->

## 👥 Equipe

- José Vinicius de Santana Souza ([jvss2](https://github.com/jvss2))
- Camila Barbosa Vieira ([cbv2](https://github.com/cbv2))
- Rafael dos Reis Labio ([rrl3](https://github.com/rrl3))
- Matheus Lafayette Vasconcelos ([MLV](https://github.com/MLV))

---

Este projeto é parte da disciplina de TAES e visa explorar contribuições práticas de IA generativa no ciclo de vida do software.

