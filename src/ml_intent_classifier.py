from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

@dataclass
class IntentPrediction:
    intent: str
    confidence: float

class IntentClassifier:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._label_encoder: LabelEncoder | None = None

    def train_from_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        intents = df["intent"].astype(str).tolist()

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(intents)

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

        self._pipeline.fit(texts, y)

    def predict(self, text: str) -> IntentPrediction:
        if self._pipeline is None or self._label_encoder is None:
            raise RuntimeError("Model not trained. Call train_from_csv() first.")

        probs = self._pipeline.predict_proba([text])[0]
        best_idx = probs.argmax()
        intent = self._label_encoder.inverse_transform([best_idx])[0]
        confidence = float(probs[best_idx])
        return IntentPrediction(intent=intent, confidence=confidence)

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        return [self.predict(t) for t in texts]
