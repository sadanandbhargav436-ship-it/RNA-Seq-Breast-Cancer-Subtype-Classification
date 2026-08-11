"""Command-line entrypoint for training a disease classification model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data import (
    align_expression_and_metadata,
    load_gene_expression_matrix,
    load_sample_metadata,
    preprocess_expression,
    train_test_split_data,
)
from src.model import evaluate_model, save_model, train_model
from src.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RNA-Seq disease classification model."
    )
    parser.add_argument(
        "--expression", required=True, help="Path to expression matrix CSV/TSV"
    )
    parser.add_argument(
        "--metadata", required=True, help="Path to sample metadata CSV/TSV"
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Metadata column containing the target labels",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples to reserve for testing",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "xgboost", "logistic_regression", "svm"],
        help="Classifier type to train",
    )
    parser.add_argument(
        "--output",
        default="models/disease_classifier.joblib",
        help="Output path for the trained model",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)

    print(f"Loading expression matrix from: {args.expression}")
    expression = load_gene_expression_matrix(args.expression)

    print(f"Loading metadata from: {args.metadata}")
    metadata = load_sample_metadata(args.metadata)
    metadata = (
        metadata.rename(columns={args.label_column: "label"})
        if args.label_column != "label"
        else metadata
    )

    features, labels = align_expression_and_metadata(expression, metadata)
    features = preprocess_expression(features)

    X_train, X_test, y_train, y_test = train_test_split_data(
        features,
        labels,
        test_size=args.test_size,
        stratify=True,
        random_state=args.random_state,
    )

    print(f"Training '{args.model}' model...")
    model = train_model(
        X_train, y_train, method=args.model, random_state=args.random_state
    )

    print("Evaluating model...")
    evaluation = evaluate_model(model, X_test, y_test)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output_path)

    print("\n-------------------------------------------")
    print("Trained model saved to:", output_path)
    print("\nClassification report:")
    for label, metrics in evaluation["classification_report"].items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        print(
            f"  - {label}: precision={metrics['precision']:.3f},"
            f" recall={metrics['recall']:.3f},"
            f" f1-score={metrics['f1-score']:.3f}"
        )
    print("\nOverall Accuracy:", evaluation["classification_report"]["accuracy"])
    print("\nConfusion Matrix:")
    print(evaluation["confusion_matrix"])
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()
