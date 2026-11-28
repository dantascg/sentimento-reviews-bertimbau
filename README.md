# Análise de Sentimentos em Avaliações de Marketplace utilizando BERTimbau

Projeto acadêmico da disciplina de Redes Neurais que investiga a
classificação de sentimentos em avaliações de produtos de um marketplace
brasileiro. O trabalho compara um modelo tradicional de aprendizado de
máquina com um modelo baseado em Transformer para identificar avaliações
positivas e negativas.

## Objetivo

Classificar o sentimento expresso no texto de avaliações de consumidores
e comparar o desempenho de:

-   **Baseline:** TF-IDF + Regressão Logística.
-   **Modelo principal:** BERTimbau, com fine-tuning para classificação
    binária.

Os dois modelos utilizam os mesmos conjuntos de treino, validação e
teste, permitindo uma comparação justa.

## Dataset

Foi utilizado o **B2W-Reviews01**, conjunto público de avaliações de
produtos de um marketplace brasileiro.

-   Repositório oficial:
    https://github.com/americanas-tech/b2w-reviews01

O dataset original não é armazenado neste repositório. Os arquivos
processados utilizados nos experimentos estão em `data/processed/`.

**Atribuição e licença:** o dataset B2W-Reviews01 é disponibilizado sob
a licença CC BY-NC-SA 4.0. Consulte o repositório oficial para os termos
completos e atribua a autoria à Americanas S.A. ao reutilizar os dados.

## Preparação e rotulagem

A tarefa foi definida como classificação binária a partir da nota da
avaliação:

  Nota original       Classe Sentimento
  ----------------- -------- ------------
  1 ou 2 estrelas          0 Negativo
  3 estrelas             --- Excluída
  4 ou 5 estrelas          1 Positivo

Os dados foram divididos de forma estratificada em treino, validação e
teste.

  Conjunto        Registros
  ----------- -------------
  Treino             77.355
  Validação          16.576
  Teste              16.577
  **Total**     **110.508**

Todos os arquivos possuem as colunas `texto` e `rotulo`.

## Modelos

### Baseline --- TF-IDF + Regressão Logística

O baseline transforma os textos em representações numéricas usando
`TfidfVectorizer` e realiza a classificação com `LogisticRegression` do
scikit-learn.

O vetorizador é ajustado exclusivamente com os textos do conjunto de
treino. Validação e teste são transformados com o mesmo vetorizador, sem
reajuste. O classificador utiliza `max_iter=1000` e `random_state=42`.

O script de treinamento está em `src/baseline/train_baseline.py`. Os
artefatos gerados incluem o vetorizador, o classificador, as métricas em
JSON e as matrizes de confusão em CSV e PNG.

### Modelo principal --- BERTimbau

Será utilizado o modelo pré-treinado
`neuralmind/bert-base-portuguese-cased`, com fine-tuning para a tarefa
de classificação binária. A implementação e o treinamento estão em
desenvolvimento.

## Resultados do baseline

Resultados registrados em `results/baseline/metrics.json`:

  Métrica                  Validação    Teste
  ---------------------- ----------- --------
  Acurácia (Accuracy)         94,17%   94,05%
  Precisão (Precision)        95,55%   95,63%
  Revocação (Recall)          96,21%   95,95%
  F1-score                    95,88%   95,79%

As métricas de precisão, revocação e F1-score apresentadas consideram a
classe positiva (rótulo 1), conforme a configuração padrão das métricas
utilizadas no script.

As matrizes de confusão correspondentes estão disponíveis em
`results/baseline/`. A comparação final com o BERTimbau será
acrescentada após o treinamento e a avaliação desse modelo.

## Estrutura do repositório

``` text
data/
  raw/                     # Dataset original (não versionado)
  processed/
    treino.csv             # Dados de treino
    validacao.csv          # Dados de validação
    teste.csv              # Dados de teste

src/
  data_preparation/        # Preparação dos dados
  baseline/
    train_baseline.py      # Treinamento e avaliação do baseline
  bertimbau/               # Implementação do BERTimbau
  evaluation/              # Avaliação comparativa e análise de erros

results/
  baseline/
    metrics.json           # Métricas de validação e teste
    matriz_confusao_*.csv  # Matrizes de confusão em CSV
    matriz_confusao_*.png  # Matrizes de confusão em imagem
    *.joblib               # Vetorizador e classificador treinados
  bertimbau/               # Resultados do BERTimbau

notebooks/                 # Experimentos e análises
demo/                      # Materiais para demonstração
docs/                      # Documentação do projeto
```

## Tecnologias

-   Python
-   Pandas e NumPy
-   scikit-learn
-   PyTorch
-   Hugging Face Transformers e Datasets
-   Matplotlib e Seaborn
-   Jupyter

## Como executar o baseline

Recomenda-se Python 3.10 ou superior.

1.  Clone o repositório e acesse a pasta do projeto.
2.  Crie um ambiente virtual:

``` bash
python -m venv .venv
```

No Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

No Linux/macOS:

``` bash
source .venv/bin/activate
```

3.  Instale as dependências:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4.  Execute o treinamento e a avaliação do baseline a partir da raiz do
    repositório:

``` bash
python src/baseline/train_baseline.py
```

O script utiliza por padrão os arquivos em `data/processed/` e salva os
resultados em `results/baseline/`. Os caminhos podem ser personalizados
pelos argumentos `--train`, `--validation`, `--test` e `--output-dir`.

## Equipe e responsabilidades

-   **Leonardo:** preparação, limpeza, rotulagem e divisão do dataset.
-   **Alex:** pesquisa teórica e apresentação.
-   **Yan:** baseline TF-IDF + Regressão Logística, organização do
    GitHub e documentação.
-   **Karine:** implementação e treinamento do BERTimbau.
-   **João:** avaliação final, testes, análise de erros e demonstração.

## Status do projeto

-   [x] Preparação e divisão do dataset.
-   [x] Implementação e avaliação do baseline TF-IDF + Regressão
    Logística.
-   [ ] Implementação e treinamento do BERTimbau.
-   [ ] Comparação final dos modelos e análise de erros.
-   [ ] Demonstração do projeto.
