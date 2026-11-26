# Análise de Sentimentos em Avaliações de Marketplace utilizando BERTimbau

Projeto acadêmico de Redes Neurais para classificação binária de sentimentos em avaliações de produtos de um marketplace brasileiro. O objetivo é comparar um baseline tradicional com um modelo de linguagem em português, identificando avaliações positivas e negativas.

## Problema

Avaliações de consumidores podem indicar a percepção sobre os produtos e serviços oferecidos em um marketplace. Neste projeto, o texto das avaliações será usado para prever se o sentimento expresso é negativo ou positivo.

## Dataset

Será utilizado o dataset **B2W-Reviews01**, um conjunto de avaliações de produtos de um marketplace brasileiro. A preparação dos dados, incluindo limpeza, rotulagem e divisão em conjuntos, será documentada e realizada durante o desenvolvimento do projeto.

O dataset não está incluído neste repositório. Sua origem e as instruções para obtenção serão registradas posteriormente conforme as orientações da disciplina e as condições de uso da fonte original.

## Estratégia de rotulagem

As avaliações serão convertidas para uma tarefa de classificação binária:

- 1 e 2 estrelas: sentimento negativo (`0`);
- 4 e 5 estrelas: sentimento positivo (`1`);
- 3 estrelas: excluídas da análise por representarem uma avaliação intermediária.

## Modelos

Serão avaliados dois modelos:

1. **Baseline:** representação TF-IDF dos textos combinada com Regressão Logística.
2. **Modelo principal:** BERTimbau (`neuralmind/bert-base-portuguese-cased`) com fine-tuning para classificação de sentimentos, utilizando PyTorch e a biblioteca Transformers.

Os modelos ainda não foram implementados neste estágio inicial.

## Métricas de avaliação

O desempenho será analisado utilizando:

- acurácia (accuracy);
- precisão (precision);
- revocação (recall);
- F1-score;
- matriz de confusão.

Os resultados serão comparados entre o baseline e o BERTimbau, acompanhados de uma análise de erros.

## Estrutura do repositório

```text
data/
  raw/                 # Dados originais, não versionados
  processed/           # Dados preparados, não versionados

notebooks/             # Explorações e experimentos em Jupyter

src/
  data_preparation/    # Limpeza, rotulagem e divisão do dataset
  baseline/            # TF-IDF + Regressão Logística
  bertimbau/           # Fine-tuning e inferência com BERTimbau
  evaluation/          # Métricas, matrizes e análise de erros

demo/                  # Materiais para demonstração
results/
  baseline/            # Resultados do baseline
  bertimbau/           # Resultados do BERTimbau
docs/                  # Documentação teórica e do projeto
```

Datasets, checkpoints, logs e resultados gerados não são versionados por padrão, pois podem ser grandes ou reproduzíveis.

## Responsabilidades da equipe

- **Pessoa 1:** preparação, limpeza, rotulagem e divisão do dataset.
- **Pessoa 2:** pesquisa teórica e apresentação.
- **Pessoa 3:** baseline TF-IDF + Regressão Logística, organização do GitHub e README.
- **Pessoa 4:** implementação e treinamento do BERTimbau.
- **Pessoa 5:** avaliação final, testes, análise de erros e demonstração.

## Resultados

Os resultados ainda serão preenchidos após a implementação, o treinamento e a avaliação dos modelos. Nenhum valor de desempenho foi definido ou inventado nesta etapa.

## Instalação das dependências

Recomenda-se utilizar Python 3.10 ou superior em um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instale as dependências do projeto:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O projeto utiliza PyTorch como framework de aprendizado profundo e não utiliza TensorFlow.
