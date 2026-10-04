# Usar una imagen oficial de Python ligera
FROM python:3.12-slim

# Evitar que Python escriba archivos .pyc y forzar logs sin buffer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema y uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración de uv (pyproject.toml y uv.lock)
COPY pyproject.toml uv.lock ./

# Instalar las dependencias directamente en el entorno del sistema
# Streamlit se agregará aquí, pero uv lo manejará
RUN uv pip install --system streamlit && uv sync --frozen

# Copiar todo el proyecto
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
