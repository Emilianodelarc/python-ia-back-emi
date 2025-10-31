FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Paquetes del sistema mínimos y estables:
# - libgomp1: FAISS / numpy
# - tesseract-ocr (binario); el idioma ES lo bajamos por curl (evita tesseract-ocr-spa)
# - libjpeg + zlib: compatibilidad con Pillow
# - ca-certificates + curl: HTTPS y descarga del modelo spa.traineddata
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    libjpeg62-turbo \
    zlib1g \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Descargar datos de idioma español para Tesseract (en vez de tesseract-ocr-spa)
# Usamos tessdata_best por calidad; podés cambiar a tessdata si querés más liviano.
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata && \
    curl -L -o /usr/share/tesseract-ocr/4.00/tessdata/spa.traineddata \
      https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata

WORKDIR /app

# Dependencias de Python (capa cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

# Env por defecto (podés sobreescribirlos en Render)
ENV OCR_ENABLED=false \
    CORS_ORIGINS="http://localhost:5173,http://localhost:3000,https://runner-py-ia.vercel.app"

# Exponer y arrancar en el puerto que inyecta la plataforma (Render/Railway/Cloud Run)
EXPOSE 8000
CMD ["sh", "-c", "uvicorn servidor:app --host 0.0.0.0 --port ${PORT:-8000}"]
