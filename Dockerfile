FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e ".[dev]"

COPY . .

CMD ["python", "-m", "pytest", "tests/", "-m", "not slow and not benchmark", "-v"]
