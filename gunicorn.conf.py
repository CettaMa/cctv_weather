"""Konfigurasi Gunicorn untuk container."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Harus tetap 1 worker: status kamera, cache segmen, dan thread latar belakang disimpan di memori
# proses. Naikkan WEB_THREADS bila penonton bersamaan bertambah (tiap stream MJPEG memegang 1 thread).
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("WEB_THREADS", "32"))

timeout = 120
# lebih pendek dari stop_grace_period compose, supaya stream yang masih terbuka tidak berakhir SIGKILL
graceful_timeout = 8
keepalive = 5

# berkas heartbeat di RAM agar I/O disk yang lambat tidak membuat worker dianggap macet
worker_tmp_dir = "/dev/shm"
errorlog = "-"
