# Famigo UI

## how to run:

* with uv

```shell
uv sync
uv run streamlit run app.py
```

* with pip

```shell
pip install -r requirements.txt
streamlit run app.py
```

## setup db
```shell
createdb voice_face_rag
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/voice_face_rag"
alembic -c app/db/migrations/alembic.ini upgrade head
```
