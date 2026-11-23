# Gunakan image Python versi slim agar lebih ringan dan cepat
FROM python:3.11-slim

# Set environment variables agar output Python tidak dibuffer
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Tentukan direktori kerja di dalam kontainer
WORKDIR /app

# Install system runtime dependencies (libgomp1 for LightGBM, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk memanfaatkan cache layer Docker
COPY requirements.txt .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke dalam kontainer
COPY . .

# Beri tahu Docker bahwa aplikasi ini berjalan di port 8000 (secara default)
EXPOSE 8000

# Healthcheck untuk memantau kesiapan kontainer
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/status || exit 1

# Jalankan server menggunakan Gunicorn (production-grade server)
# Kita menggunakan bind port dari env var $PORT agar kompatibel dengan Render
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --worker-class gthread --workers 1 --threads 32 --timeout 120 --graceful-timeout 30 --keep-alive 5 app:app"]
