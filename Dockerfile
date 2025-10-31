# Imagen base liviana
FROM python:3.11-slim

# Evita prompts de apt y reduce ruido
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Dependencias del sistema:
# - libgomp1: necesario para FAISS / numpy
# - tesseract-ocr + idioma español
# - libs de imagen comunes para Pillow (JPEG/PNG/TIFF/JP2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr tesseract-ocr-spa \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libtiff5 \
    libopenjp2-7 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# (Opcional) Si usás Tesseract con modelos custom
# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# Instala Python deps primero (capa cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Config de CORS por defecto (podés sobreescribir en deploy)
ENV OCR_ENABLED=false \
    CORS_ORIGINS="http://localhost:5173,http://localhost:3000,https://runner-py-ia.vercel.app"

# IMPORTANTE: usá el puerto provisto por la plataforma si existe
# (Cloud Run/Render/Railway inyectan $PORT)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn servidor:app --host 0.0.0.0 --port ${PORT:-8000}"]
