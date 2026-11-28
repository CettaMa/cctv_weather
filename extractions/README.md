# Ekstraksi fitur PKJI 2023 dari video CCTV

Paket `pkji/` menggantikan `yolo_onnx_pkji2_video.py`. Script lama tidak diubah
dan tetap bisa dipakai sebagai pembanding.

## Lisensi model

`models/v-1-1.onnx` adalah model deteksi **Ultralytics YOLO26n** (kelas motor, mobil, truk, orang,
bis) yang dilatih pada data Terusan Buah Batu, diekspor dengan Ultralytics 8.4.209 pada
12 November 2026. Menurut metadata di dalam berkas, model ini berlisensi
**[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)**
([ketentuan Ultralytics](https://ultralytics.com/license)). Siapa pun yang memakai atau
membagikannya, termasuk sebagai bagian dari layanan jaringan, perlu mengikuti syarat lisensi
tersebut; untuk penggunaan di luar syarat itu Ultralytics menyediakan lisensi komersial. Paket
Python `ultralytics` yang dipakai kode di folder ini juga berlisensi AGPL-3.0.

## Instalasi

Butuh GPU NVIDIA dan [uv](https://docs.astral.sh/uv/). Jalankan dari folder `extractions`.

```powershell
uv venv .venv --python 3.12
```

```powershell
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt
```

## Membuat engine TensorRT

Sekali saja, dari `model-v1-1.pt`. Engine hanya berlaku untuk GPU dan versi
TensorRT tempat ia dibangun.

```powershell
.venv\Scripts\python.exe -m pkji.export_model
```

## Menjalankan

```powershell
.venv\Scripts\python.exe -m pkji --input dataset --recursive
```

Hasil ditulis ke `results_video/`: `video_summary_pkji2.csv`,
`subsession_1min_pkji2.csv`, ringkasan 5-60 menit, dan `run_manifest.json`
(metode dan model yang dipakai). Nama file video harus
`<LOKASI>_<YYYYMMDDTHHMMSSZ>.mp4`.

| Opsi | Arti |
|---|---|
| `--legacy` | Cara hitung script lama (lihat di bawah) |
| `--backend onnx` | Pakai ONNX Runtime dan `models/v-1-1.onnx` |
| `--workers N` | Proses decode paralel (default 4; 0 = satu proses) |
| `--servers 2` | Dua inference server; sedikit lebih cepat, VRAM 2x |
| `--cache-dir` | Lokasi cache deteksi (default `cache/`) |
| `--redetect` | Abaikan cache |
| `--inference-interval` | Jarak antar sampel dalam detik (default 0.04; 0.2 dengan `--legacy`) |
| `--roi` | File ROI, atau folder berisi `<LOKASI>.json` |
| `--append` | Tambah ke CSV lama; ditolak bila metode atau model berbeda |

### Pause dan lanjutkan

Tekan **Ctrl+C** (atau Ctrl+Break) kapan saja. Program menyimpan checkpoint lalu
berhenti dan melepas GPU. Jalankan ulang perintah yang sama untuk melanjutkan;
jangan sertakan `--redetect`.

| Tahap | Disimpan | Yang diulang saat lanjut |
|---|---|---|
| Deteksi | Saat pause, dan tiap 15 detik (`*.partial.parquet`) | Bagian video yang sudah selesai hanya di-decode ulang tanpa GPU (paling lama sekitar setengah menit) |
| Fitur | Setelah tiap video (`checkpoints/`) | Video yang sedang dihitung saat pause |

Bila proses mati paksa (listrik, jendela ditutup), paling banyak sekitar 15 detik
deteksi per video yang hilang. Checkpoint ada di `--cache-dir`. Mengubah ROI,
parameter PKJI, `--legacy`, atau kode `pkji` membuat tahap fitur dihitung ulang,
tetapi tidak memerlukan deteksi ulang. CSV baru ditulis setelah semua video
selesai.

## Perhitungan default vs `--legacy`

| Hal | Default | `--legacy` |
|---|---|---|
| Baris 1 menit | Satu per menit kalender; bagian video yang tumpang tindih dibuang | Per video |
| Agregasi 5-60 menit | Rata-rata berbobot jumlah sampel | Rata-rata per baris |
| Status | `PARTIAL` bila frame terbaca < 98% | `OK` |
| Tracker | Prediksi gerak (`MotionTracker`) | Centroid |
| Arus | Garis hitung bila ada `count_line`; bila tidak, ID baru tanpa kendaraan yang sudah ada di awal | Semua ID baru |
| Awal sesi | 30 detik pertama tidak masuk rata-rata arus | Diekstrapolasi |
| Waktu | Timestamp container | `frame_idx / fps` |
| Interval sampel | 0.04 detik (praktis semua frame) | 0.2 detik |

Video yang bersambung (jeda <= 5 detik) diperlakukan sebagai satu pengamatan.

### Tracker mode default

Pada 5 sampel/detik, ByteTrack gagal memberi ID pada sekitar sepertiga deteksi.
`MotionTracker` mencocokkan deteksi dengan posisi prediksi (jarak ternormalisasi
ukuran kotak, IoU, perubahan luas). Diukur terhadap acuan yang dilacak pada semua
frame (3 klip 5 menit; dua untuk menyetel parameter, satu untuk validasi):

| | Deteksi ber-ID | IDF1 klip setel | IDF1 klip validasi |
|---|---|---|---|
| ByteTrack | 54-66% | 0,62 | 0,48 |
| Centroid (`--legacy`) | 100% | 0,56 | 0,53 |
| `MotionTracker` | 100% | 0,84 | 0,78 |

Acuan itu juga hasil tracker otomatis, bukan anotasi manusia. Parameter:
`track_max_age_sec`, `track_gate`, `track_new_gate` (disetel pada 5 sampel/detik).

Pada interval default 0.04, IDF1 `MotionTracker` terhadap acuan IoU 0,90 (klip
setel) dan 0,85 (validasi). Terhadap acuan ByteTrack 0,79 dan 0,69, sedikit lebih
rendah daripada pada 0.2 (0,81 dan 0,75). Kedua acuan memakai semua frame.

## Konfigurasi ROI

```json
{
  "frame_width": 640,
  "frame_height": 480,
  "points": [[0, 0], [600, 0], [600, 400]],
  "count_line": [[0, 300], [600, 300]],
  "pkji": {"active_lanes_in_roi": 2, "roi_length_meters": 90}
}
```

`count_line` dan `pkji` opsional. Kunci `pkji` yang tersedia ada di
`PKJIParams` (`pkji/config.py`); kunci yang tidak dikenal ditolak.

### Usulan ROI per arah (`config/usulan/`)

Satu file per arah, masing-masing dengan `count_line`. Jalankan sekali per arah
dengan folder hasil terpisah; deteksi dari cache dipakai bersama.

```powershell
.venv\Scripts\python.exe -m pkji --input dataset --recursive --roi config\usulan\arah_menjauhi.json --output-dir results_menjauhi
```

```powershell
.venv\Scripts\python.exe -m pkji --input dataset --recursive --roi config\usulan\arah_mendekati.json --output-dir results_mendekati
```

Periksa `pratinjau.png`, lalu hitung manual kendaraan yang melintasi garis pada
`verifikasi_60s.mp4`. `active_lanes_in_roi` dan `roi_length_meters` masih nilai
bawaan dan perlu diukur di lapangan.

Hitungan program dibanding hitungan manual pada video 60 detik tersebut (manual:
124 menjauhi, 49 mendekati):

| `--inference-interval` | Menjauhi | Mendekati |
|---|---|---|
| 0.2 (`--legacy` dan script lama) | 119 (96%) | 35 (71%) |
| 0.12 | 120 (97%) | 41 (84%) |
| 0.04 (default) | 123 (99%) | 43 (88%) |

Pada satu video satu jam yang dijalankan sendiri, waktunya sekitar 55 detik
(0.2), 59 detik (0.12), dan 147 detik (0.04). Cache deteksi pada 0.04 sekitar
14 MB per jam video. Jalur mendekati terhalang pohon dan
tiang. Tabel ini membandingkan hitungan garis pada
mode default; hanya intervalnya yang diubah. Perbandingan terhadap acuan otomatis semua frame (94% dan 89% pada 0.2)
terlalu optimistis untuk jalur itu, karena acuannya juga melewatkan kendaraan.

## Batasan

- **Arus baru dibandingkan dengan hitungan manual pada satu klip 60 detik** (mode
  default, tabel di atas). Status gerak dan kondisi lalu lintas belum divalidasi.
- **ROI contoh mencakup dua arah jalan**, sedangkan kapasitas dihitung untuk 2
  lajur. Arus kedua arah dibandingkan dengan kapasitas satu arah, sehingga DJ
  cenderung di atas 1. Sesuaikan ROI atau `active_lanes_in_roi`.
- Tanpa `count_line`, arus dihitung dari ID baru. Pada klip 5 menit hasilnya
  sekitar 40% lebih tinggi daripada hitungan lintasan di tengah ROI (track yang
  terputus ikut terhitung). Tambahkan `count_line` bila memungkinkan.
- Beberapa video sumber punya frame yang tidak bisa di-decode (misalnya
  `20261111T092535Z`, 96% frame terbaca) sehingga berstatus `PARTIAL`.
- Input rect (`--shape rect`) lebih ringan tetapi gagal uji paritas: deteksi bis
  turun sekitar 50%. Default-nya persegi 640x640.
- Tahap fitur berjalan di satu proses, sekitar 23 detik per jam video pada
  interval 0.04.
- Inferensi GPU tidak sepenuhnya deterministik, dan video sumber memuat frame
  H.264 rusak, jadi dua run bisa berselisih kecil (sekitar 0,1-0,3%).

## Analisis eksploratif (`eda/`)

Setelah ekstraksi kedua arah selesai (`results_menjauhi/`, `results_mendekati/`):

```powershell
.venv\Scripts\python.exe eda\pindai_mutu.py
```

```powershell
.venv\Scripts\python.exe eda\pindai_ulang.py
```

```powershell
..\.venv\Scripts\python.exe eda\analisis.py
```

```powershell
..\.venv\Scripts\python.exe eda\buat_laporan.py
```

Dua perintah terakhir butuh pandas dan matplotlib. Hasil: `eda/laporan_eda.html` dan
`eda/data/dataset_menit_berflag.csv`.

## Forecasting (`forecast/`)

Meramal arus (kendaraan/jam) per arah pada resolusi 5 menit, horizon 5 menit sampai 6 jam, dan
menguji hubungan cuaca (OpenWeatherMap) dengan arus. Butuh venv induk (`..\.venv`) dengan pandas,
lightgbm, xgboost, torch CUDA, neuralforecast, optuna, statsmodels. Jalankan berurutan dari folder
`extractions`:

```powershell
..\.venv\Scripts\python.exe -m forecast.cuaca
```

```powershell
..\.venv\Scripts\python.exe -m forecast.siapkan_data
```

```powershell
..\.venv\Scripts\python.exe -m forecast.model_gbdt
```

```powershell
..\.venv\Scripts\python.exe -m forecast.model_deep
```

```powershell
..\.venv\Scripts\python.exe -m forecast.evaluasi
```

```powershell
..\.venv\Scripts\python.exe -m forecast.cuaca_vs_arus
```

```powershell
..\.venv\Scripts\python.exe -m forecast.jeda_cuaca
```

```powershell
..\.venv\Scripts\python.exe -m forecast.buat_laporan
```

### Panel ramalan di aplikasi peta

Aplikasi Flask (`../app.py`) menampilkan panel ramalan saat marker **CCTV TERUSAN BUAH BATU-JEMBATAN
TOL 02** diklik (atau lewat tombol di popup TOL 01/02, yang markernya bertumpuk). Datanya dibuat
dengan:

```powershell
..\.venv\Scripts\python.exe -m forecast.ekspor_app
```

Hasilnya `../outputs/buahbatu_forecast.json`. Isinya mode replay 20-26 Nov: ramalan arus dan SMP
+5 sampai +360 menit dari LightGBM tanpa fitur cuaca (out-of-sample), dan cuaca Open-Meteo per 5
menit di titik CCTV (butuh `forecast.cuaca --sumber openmeteo`). Endpoint:
`/api/buahbatu/forecast/meta` dan `/api/buahbatu/forecast?asal=<ISO>`. Tampilan ada di
`../static/buahbatu_forecast.js` dan `.css`; buka langsung dengan `/?ramalan=buahbatu`. Kondisi
Lancar/Padat/Macet di panel adalah DJ = SMP/jam ÷ kapasitas bawaan PKJI, jadi hanya perkiraan.

Layer peta: lingkaran hujan berjari-jari 1 km (nilai satu titik, bukan peta sebaran) dan heatmap
kepadatan selebar badan Jalan Buah Batu dari `../static/buahbatu_jalan.json` (geometri
© kontributor OpenStreetMap, ODbL). Warna tiap arah memakai ramalan di titik kamera untuk seluruh
ruas. Rentang warnanya (kend/jam) bisa diatur di panel.

### Mutu video dan galat ramalan

Sebagian video memutar ulang cuplikan yang sama (25 Nov sekitar 10.40-11.40 WIB hanya berisi
rekaman jam overlay 10.26-10.28). Deteksinya lewat jam overlay, jalankan sebelum `siapkan_data`:

```powershell
.venv\Scripts\python.exe eda\pindai_ulang.py
```

`eda/analisis.py` dan `forecast.siapkan_data` membuang menit yang ditandai
(`eda/data/ulang_per_menit.csv`).
Gambar rusak (galat dekode H.264) belum difilter: video dengan galat dekode lebih banyak cenderung
menghasilkan arus lebih rendah (Spearman -0,43 pada 79 video siang), terutama sejak 20 Nov, sehingga
ramalan terlihat jauh di atas aktual pada periode itu.

### Detektor hujan dari video

Waktu mulai hujan per menit diambil dari video CCTV (cuaca API hanya per jam). Fitur latar jalan per
menit diekstrak dengan venv ekstraksi (sekitar 20 menit untuk 188 video), lalu detektor dilatih dan
arus di sekitar awal hujan dianalisis dengan venv induk:

```powershell
.venv\Scripts\python.exe -m forecast.hujan_video
```

```powershell
..\.venv\Scripts\python.exe -m forecast.deteksi_hujan
```

Label kering/basah ada di `forecast/label_hujan_video.csv` dan awal hujan yang sudah dicek di
`forecast/awal_hujan_terverifikasi.csv`; keduanya hasil pemeriksaan visual frame. Untuk memeriksa
kandidat baru: `forecast.frame_cek <video> <menit> ...` (frame resolusi penuh) dan
`forecast.lembar_hujan <video>` (latar per menit). Detektor hanya untuk siang hari; deteksi palsu
umumnya akibat bayangan matahari pagi atau senja.

`forecast.jeda_cuaca` menguji jeda antara hujan dan perubahan arus dan butuh `cuaca_owm.csv` dan
`cuaca_openmeteo.csv` (jalankan `forecast.cuaca` sekali dengan `--sumber openmeteo`, lalu sekali
lagi tanpa opsi agar `cuaca_5menit.csv` kembali dari OpenWeatherMap).

`forecast.cuaca` mengambil cuaca per jam dari OpenWeatherMap One Call 4.0 di titik CCTV
(-6.9652846, 107.6381471). API key dibaca dari `$env:OPENWEATHER_API_KEY` atau dari
`thesis\forecasting\.env` (ubah dengan `--env`). Hanya jam yang belum ada di
`forecast/data/cuaca_owm.csv` yang diminta. `--sumber openmeteo` memakai Open-Meteo tanpa API key. Split kronologis ada di `forecast/common.py` (latih sampai
19 Nov, validasi 20-22 Nov, uji 23-26 Nov). Hasil: `forecast/laporan_forecast.html`, tabel di
`forecast/hasil/`. Model deep (24 run) di RTX 3060 6 GB butuh sekitar 15 menit; hasil
tiap model disimpan di `forecast/hasil/deep_cache/` sehingga run yang terputus bisa dilanjutkan (hapus
folder itu untuk melatih ulang).

## Pengujian

```powershell
.venv\Scripts\python.exe -m pytest tests
```

```powershell
..\.venv\Scripts\python.exe -m pytest tests\test_forecast.py
```

```powershell
.venv\Scripts\python.exe -m pkji.parity --video dataset\2026-11-11\TERUSAN_BUAHBATU_20261111T102531Z.mp4
```

```powershell
.venv\Scripts\python.exe -m pkji.compare <folder_hasil_acuan> <folder_hasil_baru>
```
