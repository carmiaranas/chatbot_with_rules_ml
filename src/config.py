from dataclasses import dataclass, field

@dataclass
class MLConfig:
    model_path: str = "models/intent_classifier.joblib"
    min_confidence: float = 0.6  # below this, treat as low confidence

@dataclass
class ChatbotConfig:
    # Use default_factory so each ChatbotConfig gets its own MLConfig instance
    ml: MLConfig = field(default_factory=MLConfig)
