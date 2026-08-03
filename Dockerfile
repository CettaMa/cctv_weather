# Gunakan image Python versi slim agar lebih ringan dan cepat
FROM python:3.11-slim

# Set environment variables agar output Python tidak dibuffer
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Tentukan direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements terlebih dahulu untuk memanfaatkan cache layer Docker
COPY requirements.txt .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek ke dalam kontainer
COPY . .

# Beri tahu Docker bahwa aplikasi ini berjalan di port 8000 (secara default)
EXPOSE 8000

# Jalankan server menggunakan Gunicorn (production-grade server)
# Kita menggunakan bind port dari env var $PORT agar kompatibel dengan Render
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} app:app"]
