# Laporan — Klasifikasi Cuaca CCTV Kordon (Hujan / Cerah) dengan ML & DL

**Cakupan:** Kamera Perempatan Kordon (CCTV-02), klasifikasi biner **hujan (rain)** vs **cerah (sunny)**, komputasi luring CPU. Seluruh dataset, model, grafik, dan laporan terisolasi penuh di bawah `weather/kordon/` tanpa mengubah berkas SP Buah Batu.
**Data:**
- Hujan: `weather/kordon/dataset/video/cctv_perempatan_kordon_02_20260928T070904Z_rain.mp4` (8,7 MiB, 640×480, 24,95 fps, 520 dtk, rekaman 28-09-2026 ~13:50 WIB).
- Cerah: `weather/kordon/dataset/video/cctv_perempatan_kordon_02_20261004T054311Z_sunny.mp4` (10,1 MiB, 640×480, 24,97 fps, 600 dtk, rekaman 04-10-2026 ~12:25 WIB).

**Peralatan:** `script_tools/weather_ml.py`, `script_tools/weather_tabular.py`, dan `script_tools/weather_mog.py` dengan profil `--dataset kordon`.
**Keluaran:** 10 model klasifikasi (5 ML: RF, XGBoost, LightGBM, CatBoost, MOG; 5 DL: MLP, TabNet, FT-Transformer, CNN, CNN_paper) dievaluasi pada split temporal yang sama.

---

## Langkah 1 — Pemotongan Frame (Slice Seimbang)

**Perintah:**
```bash
python -m script_tools.weather_ml slice --dataset kordon
```
**Hasil:**
- Target ekstraksi: 600 frame per kelas.
- Hasil pembacaan video: 598 frame hujan dan 598 frame cerah (pembuat sampel menghasilkan 598 frame karena pembacaan bitstream video berakhir sebelum mencapai dua indeks target terakhir).
- Manifest indeks video: disimpan di `weather/kordon/dataset_frames/frame_index.json`, memetakan setiap nama berkas ke nomor frame asli di dalam berkas MP4.
- Penyimpanan: `weather/kordon/dataset_frames/all/{rain,sunny}/frame_*.jpg` (JPEG kualitas 90).

**Keputusan teknis:**
1. **Sampling seragam sepanjang klip:** Frame diambil dengan linspace berjarak tetap dari awal hingga akhir durasi (520 dtk dan 600 dtk). Dengan 598 frame, jarak antar-frame berturutan sekitar 0,9 hingga 1,0 detik.
2. **Crop eliminasi teks:** Koordinat crop disetel ke `x∈[0,02 ; 0,98]` dan `y∈[0,08 ; 0,86]` dari resolusi asli 640×480. Bagian atas dipotong 8% untuk menghapus teks nama kamera dan stempel waktu (28-09-2026 vs 04-10-2026). Bagian bawah dipotong 14% untuk menghapus teks watermark "CCTV-02 / Arah Pertigaan Ps. Kordon / Diskominfo Kota Bandung - 2022". Langkah ini mencegah kebocoran informasi tanggal dan teks statis ke model penglihatan komputer.
3. **Region of Interest (ROI) jalan:** Didefinisikan poligon cembung 5 titik pada koordinat relatif frame penuh: `(0,02, 0,50)`, `(0,72, 0,56)`, `(0,90, 0,70)`, `(0,90, 0,86)`, dan `(0,02, 0,86)`. Area ini mencakup permukaan aspal jalan raya serta menyingkirkan pulau tanaman di sisi kiri atas dan trotoar di sisi kanan.

![Grid Frame Kordon](output_picture/_ml_slice_grid.png)

---

## Langkah 2 — Pembagian Data Temporal (Temporal Block Split)

**Perintah:**
```bash
python -m script_tools.weather_ml split --dataset kordon
```
**Hasil:**
- Data latih (train): 459 frame hujan, 459 frame cerah (total 918 frame).
- Data uji (test): 119 frame hujan, 119 frame cerah (total 238 frame).
- Data terbuang (purged gap): 20 frame hujan, 20 frame cerah (total 40 frame).

**Alasan metodologis:**
Pada data video CCTV berurutan, split acak berlapis (random stratified) menimbulkan kebocoran informasi akibat autokorelasi temporal: frame $t$ dan $t+1$ (jarak ~1 detik) praktis identik. Model yang diuji pada split acak akan mendapatkan skor tinggi semu karena sekadar mengenali frame tetangga.

Untuk mengatasinya, diterapkan **temporal block split**:
1. Linimasa 598 frame dibagi rata menjadi 10 blok temporal per kelas (~60 frame per blok).
2. Blok ke-3 dan blok ke-8 dijadikan himpunan uji (test set), mencakup tepat 20% dari keseluruhan rentang waktu video.
3. Diterapkan batas pengaman (purge gap) sebanyak 5 frame pada setiap sisi tepi blok uji. Sebanyak 20 frame per kelas yang bertetangga langsung dengan data uji disingkirkan ke folder `purged/` dan tidak dimasukkan ke dalam data latih maupun uji.

![Distribusi Split Temporal](output_picture/_ml_split_dist.png)

---

## Langkah 3 — Model Machine Learning Tabular

Setiap frame diekstraksi ke dalam 24 fitur statistik citra pada area ROI jalan:
- Indikator pantulan basah: `spec_ratio` (spekular global), `spec_ratio_roi` (spekular di aspal), `dark_ratio`, `blue_ratio`.
- Statistik kanal warna: rerata dan deviasi standar HSV (`hue_mean`, `hue_std`, `sat_mean`, `sat_std`, `val_mean`, `val_std`) serta RGB (`r_mean`, `g_mean`, `b_mean`, `r_std`, `g_std`, `b_std`).
- Distribusi intensitas dan tekstur: persentil abu-abu (`gray_p10`, `gray_p50`, `gray_p90`), kontras `gray_std`, ketajaman `lap_var`, gradien `sobel_mag_mean`, dan densitas tepi `edge_density`.

Empat model tabular klasik dilatih menggunakan fitur ini:

### 1. Random Forest (RF)
- Konfigurasi: 300 pohon keputusan, `random_state=42`.
- Waktu latih: 0,55 detik; latensi inferensi: 0,167 ms/frame.
- Akurasi: **0,992** (236/238 benar); Macro F1: **0,992**; ROC AUC: **1,000**.
- Rincian metrik per kelas:
  - Hujan: Precision 1,000; Recall 0,983; F1 0,992 (TP=117, FN=2).
  - Cerah: Precision 0,983; Recall 1,000; F1 0,992 (FP=0, TN=119).
- Kesalahan: 2 frame hujan salah diprediksi sebagai cerah (`frame_218.jpg` dengan $P(\text{hujan})=0,47$ dan `frame_509.jpg` dengan $P(\text{hujan})=0,34$).
- 3 Fitur terpenting: `road_gray` (15,0%), `gray_p50` (13,8%), `r_mean` (13,7%).

![RF Confusion Matrix](output_picture/_cm_rf.png)
![RF Metrics](output_picture/_metrics_rf.png)
![RF Feature Importance](output_picture/_featimp_rf.png)

### 2. XGBoost (XGB)
- Konfigurasi: 300 estimators, max depth 4, learning rate 0,05, subsample 0,9, colsample 0,9.
- Waktu latih: 1,50 detik; latensi inferensi: 0,004 ms/frame.
- Akurasi: **1,000** (238/238 benar); Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119 (sempurna).
- 3 Fitur terpenting: `gray_p50` (58,5%), `r_mean` (27,4%), `road_gray` (6,8%).

![XGB Confusion Matrix](output_picture/_cm_xgb.png)
![XGB Metrics](output_picture/_metrics_xgb.png)
![XGB Feature Importance](output_picture/_featimp_xgb.png)

### 3. LightGBM (LGBM)
- Konfigurasi: 300 estimators, num leaves 15, learning rate 0,05, min child samples 5.
- Waktu latih: **0,08 detik** (model latih tercepat); latensi inferensi: 0,008 ms/frame.
- Akurasi: **1,000**; Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119.
- 3 Fitur terpenting: `gray_p50` (11,6%), `spec_ratio` (10,5%), `gray_p10` (6,8%).

![LGBM Confusion Matrix](output_picture/_cm_lgbm.png)
![LGBM Metrics](output_picture/_metrics_lgbm.png)
![LGBM Feature Importance](output_picture/_featimp_lgbm.png)

### 4. CatBoost
- Konfigurasi: 500 iterasi, depth 4, learning rate 0,05.
- Waktu latih: 0,71 detik; latensi inferensi: 0,005 ms/frame.
- Akurasi: **1,000**; Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119.
- 3 Fitur terpenting: `gray_p50` (50,1%), `road_gray` (9,0%), `val_mean` (8,7%).

![CatBoost Confusion Matrix](output_picture/_cm_catboost.png)
![CatBoost Metrics](output_picture/_metrics_catboost.png)
![CatBoost Feature Importance](output_picture/_featimp_catboost.png)

---

## Langkah 4 — Model Deep Learning

Lima arsitektur dievaluasi: 3 model tabular berbasis PyTorch/TabNet dan 2 model CNN penglihatan citra.

### 1. Multi-Layer Perceptron (MLP)
- Arsitektur: `Linear(24, 64) → ReLU → Dropout(0,2) → Linear(64, 32) → ReLU → Dropout(0,2) → Linear(32, 2)`.
- Input: 24 fitur tabular dinormalisasi dengan `StandardScaler`.
- Pelatihan: Adam (lr $10^{-3}$, weight decay $10^{-4}$), batch 16, early stopping berhenti pada 43 epoch.
- Waktu latih: 3,85 detik; latensi inferensi: 0,002 ms/frame.
- Akurasi: **1,000**; Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119.

![MLP Confusion Matrix](output_picture/_cm_mlp.png)
![MLP Metrics](output_picture/_metrics_mlp.png)
![MLP Curves](output_picture/_curves_mlp.png)

### 2. TabNet
- Arsitektur: Pemilihan fitur sekuensial dengan perhatian renggang ($n_d=8, n_a=8$, 3 langkah keputusan, $\gamma=1,3$).
- Pelatihan: Batch 64, lr 0,02, early stopping berhenti pada 69 epoch.
- Waktu latih: 15,44 detik; latensi inferensi: 0,097 ms/frame.
- Akurasi: **1,000**; Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119.
- 3 Fitur terpenting: `gray_p50` (55,9%), `r_mean` (8,2%), `b_mean` (7,2%).

![TabNet Confusion Matrix](output_picture/_cm_tabnet.png)
![TabNet Metrics](output_picture/_metrics_tabnet.png)
![TabNet Feature Importance](output_picture/_featimp_tabnet.png)
![TabNet Curves](output_picture/_curves_tabnet.png)

### 3. FT-Transformer (Feature Tokenizer Transformer)
- Arsitektur: 3 blok transformer dengan 8 attention heads, dimensi token 64, FFN 128, dropout 0,1.
- Pelatihan: AdamW (lr $10^{-4}$, weight decay $10^{-5}$), batch 16, early stopping berhenti pada 117 epoch.
- Waktu latih: 104,29 detik; latensi inferensi: 0,087 ms/frame.
- Akurasi: **1,000**; Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119.

![FTT Confusion Matrix](output_picture/_cm_ftt.png)
![FTT Metrics](output_picture/_metrics_ftt.png)
![FTT Curves](output_picture/_curves_ftt.png)

### 4. Convolutional Neural Network (CNN Citra Mentah 64×64)
- Input: Citra RGB 64×64 hasil crop, dinormalisasi berdasarkan mean dan deviasi standar data latih.
- Arsitektur:
```
Conv2d(3 → 16, k=3, pad=1) → ReLU → MaxPool2d(2)
Conv2d(16 → 32, k=3, pad=1) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, k=3, pad=1) → ReLU → AdaptiveAvgPool2d(1)
Dropout(0,25) → Linear(64 → 2)
```
- Pelatihan: 20 epoch, Adam (lr $10^{-3}$, weight decay $10^{-4}$), batch 16, augmentasi flip horizontal pada data latih.
- Waktu latih: 13,27 detik; latensi inferensi: 0,624 ms/frame.
- Akurasi: **0,996** (237/238 benar); Macro F1: **0,996**; ROC AUC: **1,000**.
- Rincian metrik per kelas:
  - Hujan: Precision 1,000; Recall 0,992; F1 0,996 (TP=118, FN=1).
  - Cerah: Precision 0,992; Recall 1,000; F1 0,996 (FP=0, TN=119).
- Kesalahan: 1 frame hujan diprediksi sebagai cerah (`rain/frame_217.jpg` dengan $P(\text{hujan})=0,005$).

![CNN Confusion Matrix](output_picture/_cm_cnn.png)
![CNN Metrics](output_picture/_metrics_cnn.png)
![CNN Curves](output_picture/_curves_cnn.png)

### 5. CNN_paper (Hakim & Prawito, 2020)
- Referensi: Arif Luqman Hakim & Prawito, *Rain detection in image using convolutional neural network*, J. Phys.: Conf. Ser. 1528 012010 (2020), DOI: [10.1088/1742-6596/1528/1/012010](https://doi.org/10.1088/1742-6596/1528/1/012010). Berkas PDF tersedia di `weather/kordon/paper/Luqman_Hakim_2020_J._Phys.__Conf._Ser._1528_012010.pdf`.
- Arsitektur mengikuti konfigurasi terbaik pada Tabel 1 Baris 5 ("Training-5", akurasi makalah 98,30%):
```
Input 150×150 RGB (/255)
Conv2d(3 → 32, k=3) → ReLU → MaxPool2d(2)   # 150 → 148 → 74
Conv2d(32 → 64, k=2) → ReLU → MaxPool2d(2)  # 74 → 73 → 36
Linear(64×36×36=82944 → 2) → Softmax
```
- Konfigurasi pelatihan: 50 epoch penuh (tanpa early stopping), batch size 32, pengoptimal Adam (lr $10^{-3}$, weight decay $10^{-4}$), tanpa augmentasi citra. Bobot evaluasi diambil dari epoch dengan kerugian validasi terendah.
- Waktu latih: 334,07 detik (~5,5 menit pada CPU); latensi inferensi: 2,634 ms/frame.
- Akurasi: **1,000** (238/238 benar); Macro F1: **1,000**; ROC AUC: **1,000**.
- Matriks konfusi: TP=119, FN=0, FP=0, TN=119 (sempurna).

![CNN_paper Confusion Matrix](output_picture/_cm_cnn_paper.png)
![CNN_paper Metrics](output_picture/_metrics_cnn_paper.png)
![CNN_paper Curves](output_picture/_curves_cnn_paper.png)

---

## Langkah 5 — Deteksi Gerakan Mixture of Gaussians (MOG2)

Sebagai pembanding terhadap model berbasis fitur warna statis dan CNN, diimplementasikan model deteksi gerakan berbasis pengurangan latar belakang (*background subtraction*) Mixture of Gaussians. Implementasi mengacu pada struktur repositori [GuilanITS/mog-motion-detection](https://github.com/GuilanITS/mog-motion-detection).

**Metodologi:**
1. **Varian algoritma:** Menggunakan OpenCV MOG2 (`cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)`). Varian MOG orisinal tidak digunakan karena modul `cv2.bgsegm` tidak disertakan dalam distribusi bawaan paket opencv-python.
2. **Jendela temporal video:** Untuk setiap frame yang dievaluasi, MOG2 membaca sekuens video asli sepanjang 5 frame berturutan ($N=5$) pada resolusi kerja 320×240 piksel di dalam batas crop dan ROI jalan.
3. **Ekstraksi 6 fitur gerakan mask:**
   - `fg_ratio_roi`: Rasio piksel foreground terhadap total piksel ROI jalan.
   - `fg_ratio_full`: Rasio piksel foreground terhadap seluruh frame crop.
   - `small_blob_density`: Kerapatan komponen terhubung kecil (luas 2 hingga 40 piksel, mewakili tetesan air hujan atau percikan) per 10.000 piksel ROI.
   - `large_blob_density`: Kerapatan komponen terhubung besar (luas > 40 piksel, mewakili kendaraan melintas) per 10.000 piksel ROI.
   - `small_fg_share`: Proporsi luas foreground ROI yang disumbangkan oleh komponen kecil.
   - `fg_ratio_roi_std`: Deviasi standar fluktuasi `fg_ratio_roi` di sepanjang 5 frame jendela temporal.
4. **Klasifikator:** Fitur distandardisasi dengan `StandardScaler` lalu diklasifikasikan menggunakan `LogisticRegression(C=1.0, random_state=42)`. Pendekatan ini menghasilkan nilai probabilitas $P(\text{hujan})$ yang dapat diinterpretasikan koefisiennya.

**Hasil dan evaluasi:**
- Waktu ekstraksi fitur dari video: 87,17 detik untuk seluruh frame hujan dan cerah.
- Waktu latih regresi logistik: 0,01 detik; latensi inferensi logistik: 0,002 ms/frame.
- Akurasi: **0,702** (167/238 benar); Macro F1: **0,699**; ROC AUC: **0,800**.
- Rincian metrik per kelas:
  - Hujan: Precision 0,745; Recall 0,613; F1 0,673 (TP=73, FN=46).
  - Cerah: Precision 0,671; Recall 0,790; F1 0,726 (FP=25, TN=94).
- Bobot koefisien logistik terhadap kelas cerah (tanda negatif menandakan asosiasi kuat dengan kondisi hujan):
  - `small_fg_share`: **-0,8453** (pangsa blob kecil paling kuat menandakan hujan).
  - `small_blob_density`: **-0,8404** (densitas blob kecil berkorelasi tinggi dengan hujan).
  - `large_blob_density`: **+0,1548** (kendaraan besar mendominasi di kondisi cerah).
  - `fg_ratio_roi`: **+0,1441**.
  - `fg_ratio_roi_std`: **+0,0983**.
  - `fg_ratio_full`: **-0,0564**.

![MOG Masks](output_picture/_mog_masks.png)
![MOG Features](output_picture/_mog_features.png)
![MOG Confusion Matrix](output_picture/_cm_mog.png)
![MOG Metrics](output_picture/_metrics_mog.png)
![MOG Feature Importance](output_picture/_featimp_mog.png)

---

## Langkah 6 — Perbandingan Keseluruhan Model

**Perintah:**
```bash
python -m script_tools.weather_ml evaluate --dataset kordon
```

### Tabel Perbandingan Metrik Pengujian (N = 238 frame seimbang)

Urutan diurutkan berdasarkan Macro F1 tertinggi, kemudian waktu pelatihan tercepat:

| Model | Keluarga Modality | Akurasi | Macro F1 | F1 Hujan | F1 Cerah | ROC AUC | Waktu Latih (dtk) | Inferensi (ms/frame) |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** | Machine Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,08 | 0,008 |
| **CatBoost** | Machine Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,71 | 0,005 |
| **XGBoost** | Machine Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,50 | 0,004 |
| **MLP** | Deep Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 3,85 | 0,002 |
| **TabNet** | Deep Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 15,44 | 0,097 |
| **FT-Transformer** | Deep Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 104,29 | 0,087 |
| **CNN_paper** | Deep Learning | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 334,07 | 2,634 |
| **CNN** | Deep Learning | 0,996 | 0,996 | 0,996 | 0,996 | 1,000 | 13,27 | 0,624 |
| **Random Forest** | Machine Learning | 0,992 | 0,992 | 0,992 | 0,992 | 1,000 | 0,55 | 0,167 |
| **MOG** | Machine Learning | 0,702 | 0,699 | 0,673 | 0,726 | 0,800 | 0,01 | 0,002 |

*Catatan inferensi MOG: latensi 0,002 ms mencakup langkah inferensi regresi logistik. Proses decoding video dan pengurangan latar MOG2 membutuhkan rata-rata ekstraksi ~73 ms per frame pada resolusi 320×240.*

Laporan lengkap format Excel tersimpan di: `weather/kordon/model_comparison.xlsx`.

![Perbandingan Akurasi dan F1](output_picture/_compare_models.png)
![Kurva ROC Semua Model](output_picture/_roc_all.png)
![Kecepatan Latih dan Inferensi](output_picture/_speed_models.png)

### Analisis Sebaran Probabilitas dan Pola Kesalahan

Peta panas (heatmap) berikut menunjukkan nilai probabilitas $P(\text{hujan})$ yang dihasilkan ke-10 model untuk setiap frame uji, diurutkan menurut urutan waktu. Garis vertikal hitam memisahkan kelas hujan (kiri) dan kelas cerah (kanan), sedangkan garis putus-putus abu-abu menandai batas antarmuka dua blok waktu pengujian (blok 3 dan blok 8).

![Peta Panas Probabilitas Hujan](output_picture/_prain_heatmap.png)

**Temuan penting dari sebaran prediksi:**
1. **Pemisahan kelas sangat tajam pada model statistik jalan:** Model gradient boosted trees (XGB, LGBM, CatBoost), jaringan saraf tiruan tabular (MLP, TabNet, FTT), dan CNN_paper memisahkan kedua kelas secara sempurna tanpa kesalahan ($P(\text{hujan}) \approx 1,0$ pada blok hujan dan $\approx 0,0$ pada blok cerah).
2. **Pola kesalahan pada frame tertentu:**
   - CNN mengalami 1 kesalahan pada `rain/frame_217.jpg` ($P=0,005$).
   - Random Forest mengalami 2 kesalahan pada `rain/frame_218.jpg` ($P=0,47$) dan `rain/frame_509.jpg` ($P=0,34$).
   - Kedua model tersebut kesulitan pada frame transisi di blok uji ke-3 di mana pantulan spekular permukaan aspal berkurang.
3. **Karakteristik deteksi gerakan MOG:**
   - MOG menghasilkan akurasi 70,2% dan AUC 0,800. Model ini membedakan kondisi cuaca murni berdasarkan dinamika partikel bergerak di jalan raya.
   - Koefisien negatif terbesar pada `small_fg_share` (-0,85) dan `small_blob_density` (-0,84) membuktikan bahwa keberadaan partikel bergerak berukuran kecil secara statistik berkorelasi kuat dengan presipitasi hujan.
   - Keterbatasan utama MOG bersumber dari gangguan kendaraan melintas: lalu lintas padat memicu komponen gerak besar yang mengaburkan deteksi partikel hujan pada jendela waktu singkat.
4. **Efisiensi komputasi:** LightGBM tetap menjadi model paling efisien secara komputasi: waktu latih hanya 0,08 detik, latensi inferensi 0,008 ms per frame, dengan akurasi 100% pada data uji terisolasi.

---

## Keterbatasan Metodologi

1. **Keterbatasan kamera tunggal dan tanggal tunggal:** Klip hujan direkam pada 28 September 2026 (~13:50 WIB) dan klip cerah direkam pada 4 Oktober 2026 (~12:25 WIB). Generalisasi model terhadap kondisi cuaca mendung tipis, hujan malam hari, atau pencahayaan senja di hari lain belum teruji.
2. **Korelasi dalam rekaman yang sama:** Meskipun temporal block split dengan jarak 5 frame purge gap berhasil mencegah kebocoran frame tetangga langsung, data latih dan data uji tetap bersumber dari dua berkas rekaman yang sama.
3. **Validasi internal:** Pembagian train/validation untuk early stopping pada CNN, MLP, FT-Transformer, dan TabNet dilakukan secara acak bertingkat di dalam blok latih. Kurva validasi sedikit lebih optimistis dibandingkan kurva pengujian blok independen.
4. **Granularitas himpunan uji:** Dengan $N = 238$ frame uji, setiap kesalahan 1 frame bernilai $\frac{1}{238} \approx 0,42\%$ terhadap metrik akurasi.

---

## Langkah Reproduksi Penuh

Semua tahapan dapat dijalankan ulang dari akar repositori menggunakan Python environment proyek:

```bash
# 1. Ekstraksi 600 frame per kelas dengan crop anti-bocor
python -m script_tools.weather_ml slice --dataset kordon

# 2. Pembagian blok temporal (blok 3 dan 8 sebagai uji, 5-frame purge gap)
python -m script_tools.weather_ml split --dataset kordon

# 3. Pelatihan 10 model ML/DL secara berurutan
python -m script_tools.weather_ml train --model all --dataset kordon

# 4. Evaluasi, pembuatan visualisasi, dan ekspor lembar kerja Excel
python -m script_tools.weather_ml evaluate --dataset kordon
```
