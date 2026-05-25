import os
from pathlib import Path

from flask import Flask

from database.db import init_db
from routes.detection import detection_bp
from routes.admin import admin_bp


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "database" / "research_app.db"

    app = Flask(__name__)
    app.config["SECRET_KEY"] = "academic-phishing-prototype"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path.as_posix()}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["SMS_PREFER_TRANSFORMER"] = _env_flag("SMS_PREFER_TRANSFORMER", False)
    app.config["EMAIL_PREFER_TRANSFORMER"] = _env_flag("EMAIL_PREFER_TRANSFORMER", False)

    init_db(app)

    app.register_blueprint(detection_bp)
    app.register_blueprint(admin_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
