# Migrations

Alembic migration files will live here.

Generate an initial migration after dependencies are installed and the database URL is configured:

```bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```
