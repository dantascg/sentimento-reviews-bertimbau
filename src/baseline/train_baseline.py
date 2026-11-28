"""Treina e avalia o baseline TF-IDF + Regressão Logística.

Os arquivos de treino, validação e teste devem conter as colunas ``texto`` e
``rotulo``. O vectorizer é ajustado exclusivamente com os textos de treino.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


TEXT_COLUMN = "texto"
LABEL_COLUMN = "rotulo"


def load_split(path: Path) -> pd.DataFrame:
    """Carrega um split e verifica as colunas necessárias."""
    dataframe = pd.read_csv(path, encoding="utf-8")
    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{path} não contém as colunas obrigatórias: "
            f"{', '.join(sorted(missing_columns))}"
        )

    dataframe = dataframe[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    dataframe[TEXT_COLUMN] = dataframe[TEXT_COLUMN].astype(str)
    dataframe[LABEL_COLUMN] = dataframe[LABEL_COLUMN].astype(int)
    return dataframe


def calculate_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Calcula as métricas principais da classificação binária."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_confusion_matrix(matrix, split_name: str, output_dir: Path) -> None:
    """Salva a matriz de confusão em CSV e em imagem PNG."""
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=["real_0", "real_1"],
        columns=["predito_0", "predito_1"],
    )
    matrix_dataframe.to_csv(output_dir / f"matriz_confusao_{split_name}.csv")

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negativo (0)", "Positivo (1)"],
        yticklabels=["Negativo (0)", "Positivo (1)"],
        xlabel="Classe predita",
        ylabel="Classe real",
        title=f"Matriz de confusão - {split_name}",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / f"matriz_confusao_{split_name}.png", dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/treino.csv"))
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/validacao.csv"),
    )
    parser.add_argument("--test", type=Path, default=Path("data/processed/teste.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.train)
    validation = load_split(args.validation)
    test = load_split(args.test)

    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train[TEXT_COLUMN])
    validation_features = vectorizer.transform(validation[TEXT_COLUMN])
    test_features = vectorizer.transform(test[TEXT_COLUMN])

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_features, train[LABEL_COLUMN])

    metrics = {
        "config": {
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
            "vectorizer": "TfidfVectorizer",
            "classifier": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "test_samples": len(test),
        },
        "validation": {},
        "test": {},
    }

    for split_name, dataframe, features in (
        ("validation", validation, validation_features),
        ("test", test, test_features),
    ):
        predictions = classifier.predict(features)
        metrics[split_name] = calculate_metrics(dataframe[LABEL_COLUMN], predictions)
        matrix = confusion_matrix(
            dataframe[LABEL_COLUMN], predictions, labels=[0, 1]
        )
        save_confusion_matrix(matrix, split_name, args.output_dir)

    joblib.dump(vectorizer, args.output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, args.output_dir / "logistic_regression.joblib")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
