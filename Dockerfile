# Tahap ini hanya sumber pustaka sistem: image penuh sudah memuat libgomp, sehingga build tidak perlu
# apt-get (server di jaringan yang tidak bisa menjangkau deb.debian.org tetap bisa membangun image)
FROM python:3.11-trixie AS libs

# Image Python slim agar ringan; rilis Debian-nya harus sama dengan tahap libs
FROM python:3.11-slim-trixie

# TZ: jam "WIB" yang ditampilkan aplikasi berasal dari datetime.now()
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Jakarta

WORKDIR /app

# libgomp1 dibutuhkan LightGBM
COPY --from=libs /usr/lib/x86_64-linux-gnu/libgomp.so.1* /usr/lib/x86_64-linux-gnu/

# Aplikasi berjalan sebagai user biasa, bukan root
RUN useradd --system --uid 10001 --no-create-home app

# Dependensi lebih dulu supaya layer-nya tetap ter-cache saat kode berubah
COPY requirements.txt .
RUN pip install -r requirements.txt

# Kode milik root (tidak bisa diubah aplikasi); hanya csv/ dan outputs/ yang boleh ditulis
COPY . .
RUN chown -R app:app csv outputs

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/refresh-state' % os.environ.get('PORT','8000'), timeout=4)"]

# Port diatur lewat env PORT (kompatibel dengan Render); lihat gunicorn.conf.py
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
