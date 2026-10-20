from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "recommendation" / "artifacts" / "libsphere_recommender.joblib"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null", "n/a", "na"} else re.sub(r"\s+", " ", value)


def _norm_isbn(value: Any) -> str:
    return re.sub(r"[^0-9Xx]", "", _text(value)).upper()


def _norm_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).lower()).strip()


def _split_terms(value: Any) -> set[str]:
    return {part.strip().lower() for part in re.split(r"[;,|/]", _text(value)) if part.strip()}


class BookRecommender:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {path}. Run `python -m recommendation.train_model` first."
            )
        artifact = joblib.load(path)
        if artifact.get("version") != 1:
            raise ValueError("Unsupported LibSphere model artifact version; retrain the model.")
        self.vectorizer = artifact["vectorizer"]
        self.matrix = artifact["matrix"]
        self.books = pd.DataFrame(artifact["books"]).fillna("")
        if self.books.empty:
            raise ValueError("The saved recommendation model contains no books.")
        self._isbn_to_idx: dict[str, int] = {}
        self._title_to_idx: dict[str, int] = {}
        for idx, row in self.books.iterrows():
            isbn = _norm_isbn(row.get("_normalised_isbn") or row.get("isbn"))
            title = _norm_title(row.get("_normalised_title") or row.get("title"))
            if isbn:
                self._isbn_to_idx.setdefault(isbn, int(idx))
            if title:
                self._title_to_idx.setdefault(title, int(idx))

    def _match_history_item(self, item: Any) -> Optional[int]:
        if isinstance(item, str):
            return self._title_to_idx.get(_norm_title(item))
        if not isinstance(item, Mapping):
            return None
        isbn = _norm_isbn(item.get("isbn") or item.get("ISBN"))
        if isbn and isbn in self._isbn_to_idx:
            return self._isbn_to_idx[isbn]
        title = _norm_title(item.get("title") or item.get("book_title") or item.get("name"))
        return self._title_to_idx.get(title) if title else None

    def recommend_for_user(
        self,
        borrowed_books: Iterable[Any],
        *,
        programme: str | None = None,
        subject_area: str | None = None,
        limit: int = 8,
        exclude_isbns: Iterable[str] = (),
        exclude_titles: Iterable[str] = (),
        available_isbns: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 8

        history = list(borrowed_books or [])
        matched_indices = []
        for item in history:
            idx = self._match_history_item(item)
            if idx is not None and idx not in matched_indices:
                matched_indices.append(idx)

        excluded = set(matched_indices)
        excluded_isbns = {_norm_isbn(x) for x in exclude_isbns if _norm_isbn(x)}
        excluded_titles = {_norm_title(x) for x in exclude_titles if _norm_title(x)}
        for item in history:
            if isinstance(item, Mapping):
                isbn = _norm_isbn(item.get("isbn") or item.get("ISBN"))
                title = _norm_title(item.get("title") or item.get("book_title") or item.get("name"))
                if isbn:
                    excluded_isbns.add(isbn)
                if title:
                    excluded_titles.add(title)
            elif isinstance(item, str) and _norm_title(item):
                excluded_titles.add(_norm_title(item))

        if matched_indices:
            profile = np.asarray(self.matrix[matched_indices].mean(axis=0))
            scores = cosine_similarity(profile, self.matrix).ravel()
            strategy = "borrowing_history"
        else:
            query = " ".join(x for x in [_text(programme), _text(subject_area)] if x)
            if query:
                profile = self.vectorizer.transform([query])
                scores = cosine_similarity(profile, self.matrix).ravel()
                strategy = "programme_or_subject"
            else:
                scores = [0.0] * len(self.books)
                strategy = "catalogue_fallback"

        rows = []
        for idx, row in self.books.iterrows():
            isbn = _norm_isbn(row.get("isbn"))
            title_norm = _norm_title(row.get("title"))
            if int(idx) in excluded or (isbn and isbn in excluded_isbns) or (title_norm and title_norm in excluded_titles):
                continue
            if available_isbns is not None and isbn and isbn not in {_norm_isbn(x) for x in available_isbns}:
                continue
            score = float(scores[int(idx)])
            if strategy in {"programme_or_subject", "catalogue_fallback"}:
                wanted_programme = _text(programme).lower()
                wanted_subject = _text(subject_area).lower()
                if wanted_programme and wanted_programme in _text(row.get("programme")).lower():
                    score += 0.15
                if wanted_subject and wanted_subject in _text(row.get("subject_area")).lower():
                    score += 0.25
            rows.append((score, int(idx), row))

        rows.sort(key=lambda item: (-item[0], _text(item[2].get("title")).lower()))
        output = []
        for score, idx, row in rows[:limit]:
            title = _text(row.get("title"))
            subject = _text(row.get("subject_area"))
            authors = _text(row.get("authors"))
            if strategy == "borrowing_history":
                reason = f"Similar to subjects in your borrowing history{': ' + subject if subject else ''}."
            elif strategy == "programme_or_subject":
                reason = f"Selected using your programme/subject profile{': ' + (subject or _text(subject_area)) if (subject or subject_area) else ''}."
            else:
                reason = "A catalogue starter recommendation. Borrow a few books or choose interests for more personalised suggestions."
            output.append({
                "book_id": _text(row.get("book_id")),
                "title": title,
                "authors": authors,
                "isbn": _text(row.get("isbn")),
                "programme": _text(row.get("programme")),
                "subject_area": subject,
                "subjects": _text(row.get("subjects")),
                "publisher": _text(row.get("publisher")),
                "publication_year": _text(row.get("publication_year")),
                "edition": _text(row.get("edition")),
                "language": _text(row.get("language")),
                "cover_url": _text(row.get("cover_url")),
                "source_record_url": _text(row.get("source_record_url")),
                "score": round(max(0.0, min(1.0, score)), 4),
                "reason": reason,
                "strategy": strategy,
            })
        return output


_default_instance: BookRecommender | None = None


def get_recommender(model_path: str | Path = DEFAULT_MODEL_PATH) -> BookRecommender:
    """Lazy-load once per worker; do not reload the model on every API request."""
    global _default_instance
    if _default_instance is None or Path(model_path) != DEFAULT_MODEL_PATH:
        _default_instance = BookRecommender(model_path)
    return _default_instance


def recommend_for_user(
    borrowed_books: Iterable[Any],
    *,
    programme: str | None = None,
    subject_area: str | None = None,
    limit: int = 8,
    exclude_isbns: Iterable[str] = (),
    exclude_titles: Iterable[str] = (),
    available_isbns: Optional[set[str]] = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    """Convenience wrapper for the existing Flask backend."""
    return get_recommender(model_path).recommend_for_user(
        borrowed_books,
        programme=programme,
        subject_area=subject_area,
        limit=limit,
        exclude_isbns=exclude_isbns,
        exclude_titles=exclude_titles,
        available_isbns=available_isbns,
    )
