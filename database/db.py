from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def init_db(app) -> None:
    db.init_app(app)
    with app.app_context():
        from database import models  # noqa: F401

        db.create_all()
