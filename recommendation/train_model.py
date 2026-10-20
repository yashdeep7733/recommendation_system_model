"""Train the LibSphere content-based book recommendation model.

Run from the project root:
    python -m recommendation.train_model

Input:
    LibSphere_Academic_Textbooks_1000.csv (project root)

Output:
    recommendation/artifacts/libsphere_recommender.joblib
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "LibSphere_Academic_Textbooks_1000.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "recommendation" / "artifacts" / "libsphere_recommender.joblib"

REQUIRED_COLUMNS = {"title"}
TEXT_COLUMNS = ["title", "authors", "programme", "subject_area", "subjects", "publisher", "language"]


def clean_value(value: object) -> str:
    """Convert missing/odd CSV values to clean text."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return ""
    return re.sub(r"\s+", " ", text)


def normalise_isbn(value: object) -> str:
    return re.sub(r"[^0-9Xx]", "", clean_value(value)).upper()


def normalise_title(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_value(value).lower()).strip()


def build_document(row: pd.Series) -> str:
    """Weight useful fields by repeating them in a small, transparent way."""
    title = clean_value(row.get("title"))
    authors = clean_value(row.get("authors"))
    subject_area = clean_value(row.get("subject_area"))
    subjects = clean_value(row.get("subjects"))
    programme = clean_value(row.get("programme"))
    publisher = clean_value(row.get("publisher"))
    return " ".join(part for part in [title, title, authors, subject_area, subject_area, subjects, subjects, programme, publisher, clean_value(row.get("language"))] if part)


def train(csv_path: Path = DEFAULT_CSV, output_path: Path = DEFAULT_OUTPUT) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Book CSV not found: {csv_path}\nPlace LibSphere_Academic_Textbooks_1000.csv in the project root, "
            "or pass --csv /path/to/file.csv."
        )

    books = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    books.columns = [str(c).strip().lower() for c in books.columns]
    missing = REQUIRED_COLUMNS - set(books.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    for col in TEXT_COLUMNS + ["book_id", "isbn", "cover_url", "publication_year", "edition", "source_record_url"]:
        if col not in books.columns:
            books[col] = ""
        books[col] = books[col].map(clean_value)

    books = books[books["title"].str.strip().ne("")].copy()
    blank_isbn = books["isbn"].eq("")
    with_isbn = books.loc[~blank_isbn].drop_duplicates(subset=["isbn"], keep="first")
    without_isbn = books.loc[blank_isbn].drop_duplicates(
        subset=["title", "authors", "publication_year"], keep="first"
    )
    books = pd.concat([with_isbn, without_isbn]).sort_index().reset_index(drop=True)

    books["_normalised_isbn"] = books["isbn"].map(normalise_isbn)
    books["_normalised_title"] = books["title"].map(normalise_title)
    books["_document"] = books.apply(build_document, axis=1)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    matrix = vectorizer.fit_transform(books["_document"].tolist())

    artifact = {
        "version": 1,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "books": books.to_dict(orient="records"),
        "csv_path_name": csv_path.name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    print(f"Trained model on {len(books):,} books.")
    print(f"TF-IDF features: {matrix.shape[1]:,}")
    print(f"Saved model: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LibSphere book recommendation model.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the 1,000-book CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the model artifact")
    args = parser.parse_args()
    train(args.csv.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
