# Passo 1: Imagem base
FROM python:3.12

# Definir o diretório de trabalho
WORKDIR /app

# Passo 2: Instalação do Poetry
RUN pip install poetry

# Passo 3: Copiar os arquivos de dependência primeiro para aproveitar o cache do Docker
COPY pyproject.toml poetry.lock* /app/

# Passo 4: Instalação das dependências
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Passo 5: Copiar o restante do código do projeto
COPY ./iron_man_models /app

WORKDIR /app

# Passo 6: Definir variáveis de ambiente
# ENV INSIDE_DOCKER="true"
ENV PYTHONPATH "${PYTHONPATH}:/app"


# Passo 7: Comando para executar o script
CMD ["python", "iron_man_models/pipeline.py"]
