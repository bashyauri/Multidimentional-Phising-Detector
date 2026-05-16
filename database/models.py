from datetime import datetime

from database.db import db


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    source_type = db.Column(db.String(30), nullable=False)
    input_text = db.Column(db.Text, nullable=True)
    prediction_label = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    response_time_ms = db.Column(db.Float, nullable=False)
    fusion_used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "source_type": self.source_type,
            "prediction_label": self.prediction_label,
            "confidence": self.confidence,
            "response_time_ms": self.response_time_ms,
            "fusion_used": self.fusion_used,
            "created_at": self.created_at.isoformat(),
        }
