"""Susun laporan_eda.html (satu file, gambar tertanam) dari eda/data/statistik.json dan eda/gambar/.

    ..\\.venv\\Scripts\\python.exe eda\\buat_laporan.py

Angka di tabel dan kalimat dibaca dari statistik.json. Dua hal berasal dari pemeriksaan manual dan
ditulis tetap di sini: selisih jam kamera (24 video contoh) dan cek decode ffmpeg pada satu video.
"""
import base64
import json
from pathlib import Path

E = Path(__file__).resolve().parent
S = json.loads((E / "data" / "statistik.json").read_text(encoding="utf-8"))
A, B = S["menjauhi"], S["mendekati"]


def num(x, d=0):
    return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def pct(part, whole, d=1):
    return num(100 * part / whole, d) + "%"


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tabel"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(name, caption, narrow=False):
    data = base64.b64encode((E / "gambar" / f"{name}.png").read_bytes()).decode()
    cls = ' class="kecil"' if narrow else ""
    return f'<figure{cls}><img src="data:image/png;base64,{data}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def peak(d, lo=0, hi=23):
    hours = {int(k): v for k, v in d["arus_per_jam_median"].items() if lo <= int(k) <= hi}
    h = max(hours, key=hours.get)
    return h, hours[h]


def desc(d, col, q, dec=0):
    return num(d["deskriptif"][col][q], dec)


pa, pb = peak(A), peak(B)
na, nb = S["parkir_malam"]["dengan_parkir"], S["parkir_malam"]["tanpa_parkir"]
w = A["dalam_video"], B["dalam_video"]

summary = table(["", "Menjauhi", "Mendekati"], [
    ["Menit data / layak pakai", f"{num(A['menit'])} / {num(A['layak'])}", f"{num(B['menit'])} / {num(B['layak'])}"],
    ["Median arus (kend/jam)", desc(A, "est_flow_veh_per_hour", "50%"), desc(B, "est_flow_veh_per_hour", "50%")],
    ["Median arus (SMP/jam)", desc(A, "est_q_smp_per_hour", "50%"), desc(B, "est_q_smp_per_hour", "50%")],
    ["DJ efektif: median / persentil 95", f"{desc(A, 'avg_dj_effective', '50%', 2)} / {desc(A, 'avg_dj_effective', '95%', 2)}",
     f"{desc(B, 'avg_dj_effective', '50%', 2)} / {desc(B, 'avg_dj_effective', '95%', 2)}"],
    ["Jam puncak (median arus)", f"{pa[0]:02d}.00 ({num(pa[1])})", f"{pb[0]:02d}.00 ({num(pb[1])})"],
    ["Menit berlabel Padat / Macet",
     f"{A['kondisi'].get('Padat (Slow Traffic)', 0)} / {A['kondisi'].get('Macet (Congestion)', 0)}",
     f"{B['kondisi'].get('Padat (Slow Traffic)', 0)} / {B['kondisi'].get('Macet (Congestion)', 0)}"],
])

completeness = table(["Pemeriksaan", "Hasil (sama untuk kedua arah)"], [
    ["Video di dataset / punya baris hasil", f"{A['video']} / {A['video']}"],
    ["Status video", ", ".join(f"{k} {v}" for k, v in A["status_video"].items()) + ". Dua video FAILED tidak bisa dibuka"],
    ["Rentang waktu", f"{A['awal'][:16]} sampai {A['akhir'][:16]} WIB, {A['hari']} hari"],
    ["Menit terisi dari seluruh rentang", f"{num(A['menit'])} dari {num(A['menit_rentang'])} ({pct(A['menit'], A['menit_rentang'])})"],
    ["Baris menit ganda", "tidak ada"],
    ["Nilai kosong", f"arus, SMP, dan DJ arus pada {A['flag']['tanpa_arus']} menit (30 detik pemanasan di awal sesi)"],
    ["Kolom konstan (tidak berguna untuk model)", "location, target_sample_duration_seconds, c_roi_capacity_smp, road_capacity_c_smp_per_hour, error_message"],
])

flags = table(["Flag", "Arti", "Menjauhi", "Mendekati"], [
    [f"<code>flag_{k}</code>", meaning, num(A["flag"][k]), num(B["flag"][k])] for k, meaning in [
        ("cakupan_rendah", "frame yang ada kurang dari 90% menit"),
        ("frame_rusak", "lebih dari 10% frame contoh rusak berat"),
        ("menit_pendek", "durasi sampel di bawah 45 detik"),
        ("tanpa_arus", "arus belum bisa dihitung"),
        ("ulang", "rekaman lama diputar ulang (jam pada gambar sudah pernah terekam)"),
        ("parkir", "mobil parkir ada di ROI lebih dari 10% waktu (tidak mengurangi status layak)"),
    ]
])

bias = table(["Porsi frame yang ada", "Arus menjauhi", "Di ROI menjauhi", "Arus mendekati", "Di ROI mendekati", "Jumlah video"], [
    [k, num(w[0][k]["arus"], 2), num(w[0][k]["di_roi"], 2), num(w[1][k]["arus"], 2), num(w[1][k]["di_roi"], 2),
     f"{w[0][k]['video']} / {w[1][k]['video']}"] for k in ("<80%", "80-90%", "90-98%")
])

names = {"est_flow_veh_per_hour": "Arus (kend/jam)", "est_q_smp_per_hour": "Arus (SMP/jam)", "avg_dj_effective": "DJ efektif",
         "avg_roi_occupancy_pct": "Okupansi ROI (%)", "average_active_total": "Kendaraan di ROI", "avg_stopped_ratio": "Rasio berhenti"}
dec = {"est_flow_veh_per_hour": 0, "est_q_smp_per_hour": 0, "avg_dj_effective": 2, "avg_roi_occupancy_pct": 1,
       "average_active_total": 2, "avg_stopped_ratio": 3}
describe = table(["Variabel", "Arah", "Rata-rata", "Simpangan", "P5", "P25", "Median", "P75", "P95", "Maks"], [
    [label, arah] + [num(d["deskriptif"][col][q], dec[col]) for q in ("mean", "std", "5%", "25%", "50%", "75%", "95%", "max")]
    for col, label in names.items() for arah, d in (("menjauhi", A), ("mendekati", B))
])

composition = table(["Kelas", "Menjauhi", "Mendekati"], [
    [c, num(A["komposisi_pct"][c], 1) + "%", num(B["komposisi_pct"][c], 1) + "%"] for c in ("motor", "mobil", "bus", "truk")
])

hourly = table(["Jam"] + [f"{h:02d}" for h in range(0, 12)],
               [[label] + [num(d["arus_per_jam_median"][str(h)]) for h in range(0, 12)] for label, d in (("menjauhi", A), ("mendekati", B))]
               ) + table(["Jam"] + [f"{h:02d}" for h in range(12, 24)],
               [[label] + [num(d["arus_per_jam_median"][str(h)]) for h in range(12, 24)] for label, d in (("menjauhi", A), ("mendekati", B))])

html = f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laporan EDA Terusan Buahbatu</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; background:#f9f9f7; color:#0b0b0b; font:15px/1.55 system-ui,"Segoe UI",sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:40px 0 8px; border-top:1px solid #e1e0d9; padding-top:20px; }}
h3 {{ font-size:16px; margin:24px 0 6px; }} .sub {{ color:#52514e; margin:0 0 16px; }}
figure {{ margin:14px 0; background:#fcfcfb; border:1px solid #e1e0d9; border-radius:8px; padding:10px; }}
figure img {{ width:100%; height:auto; display:block; }} figure.kecil {{ max-width:520px; }}
figcaption {{ color:#52514e; font-size:13px; margin-top:6px; }}
.tabel {{ overflow-x:auto; margin:12px 0; }}
table {{ border-collapse:collapse; width:100%; background:#fcfcfb; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ border:1px solid #e1e0d9; padding:6px 9px; text-align:left; vertical-align:top; }} th {{ background:#f0efec; }}
code {{ background:#f0efec; padding:1px 4px; border-radius:3px; font-size:12.5px; }}
li {{ margin-bottom:6px; }}
.kotak {{ background:#fcfcfb; border:1px solid #e1e0d9; border-left:4px solid #2a78d6; border-radius:6px; padding:4px 18px; }}
</style></head><body><main>
<h1>Analisis eksploratif dataset lalu lintas Terusan Buahbatu</h1>
<p class="sub">Hasil ekstraksi PKJI per menit, dua arah, {A['awal'][:10]} sampai {A['akhir'][:10]}. Interval sampel 0,04 detik, model v-1-1.</p>

<h2>1. Kesimpulan</h2>
<div class="kotak"><ul>
<li><b>File hasil lengkap dan konsisten.</b> Semua {A['video']} video punya baris di kedua arah, tanpa menit ganda.</li>
<li><b>{pct(A['layak'], A['menit'])} menit layak dipakai.</b> Penyebab utama sisanya adalah frame yang hilang dari video sumber, terutama sejak 21 November.</li>
<li><b>Frame hilang tampaknya membuat arus kurang hitung.</b> Pada video yang sama, menit dengan frame di bawah 80% mencatat arus {num(100 * (1 - w[0]['<80%']['arus']))}% (menjauhi) dan {num(100 * (1 - w[1]['<80%']['arus']))}% (mendekati) lebih rendah daripada menit utuh, sedangkan kendaraan yang terlihat di ROI hanya {num(100 * (1 - w[0]['<80%']['di_roi']))}% dan {num(100 * (1 - w[1]['<80%']['di_roi']))}% lebih sedikit.</li>
<li><b>{num(A['flag']['ulang'])} menit berisi rekaman lama yang diputar ulang</b> dan dikeluarkan dari menit layak.</li>
<li><b>ROI arah mendekati menangkap mobil parkir</b> di tepi kanan. Pada {num(na['menit'])} menit malam yang terdampak, rasio berhenti {num(na['rasio_berhenti'], 2)} dibanding {num(nb['rasio_berhenti'], 2)} pada menit malam lainnya.</li>
<li><b>Angka absolut belum bisa dianggap tervalidasi.</b> Jumlah lajur dan panjang ROI masih nilai bawaan, dan arus baru dibandingkan dengan satu klip manual 60 detik.</li>
<li><b>Label kondisi lalu lintas sangat timpang</b>: Macet hanya {A['kondisi'].get('Macet (Congestion)', 0)} menit di arah menjauhi dan tidak ada di arah mendekati.</li>
</ul></div>
{summary}
<p class="sub">Kecuali disebut lain, statistik dan grafik di bagian 3 sampai 5 hanya memakai menit layak pakai.</p>

<h2>2. Kelengkapan dan mutu data</h2>
{completeness}
{fig('01_cakupan', 'Kiri: menit yang punya data per tanggal dan jam. Kanan: porsi menit layak pakai. Sel kosong berarti tidak ada video.')}
<p>Perekaman tidak menerus. Tiap jam dalam sehari terwakili {min(A['hari_per_jam'].values())} sampai {max(A['hari_per_jam'].values())} hari dari {A['hari']} hari, jadi profil per jam masih bisa dibandingkan, tetapi deret waktu penuh tidak tersedia.</p>

<h3>2.1 Aturan mutu</h3>
{flags}
<p>Menit <b>layak</b> adalah menit tanpa lima flag pertama. Ambangnya pilihan analisis ini dan bisa diubah di <code>eda/analisis.py</code>.</p>
{fig('02_mutu_per_tanggal', 'Mutu per tanggal. Sampai 20 November sebagian besar menit layak; sejak 21 November turun tajam.')}

<h3>2.2 Frame hilang dan dampaknya</h3>
<p>Status <code>PARTIAL</code> pada {A['status_video']['PARTIAL']} video berarti frame yang terbaca kurang dari 98% jumlah frame di file. Pada satu video yang diperiksa (22 November), ffmpeg membaca jumlah frame yang persis sama dengan program (69.227), jadi frame itu memang tidak bisa didapat dari file. Penyebab di sisi perekaman belum diketahui.</p>
{fig('10_arus_vs_mutu', 'Rasio menit bercakupan rendah terhadap menit utuh pada video yang sama (median antar video). 1,0 berarti tidak berbeda.')}
{bias}
<p>Arus turun lebih banyak daripada jumlah kendaraan yang terlihat di ROI. Selisih itu konsisten dengan lintasan garis hitung yang terlewat saat ada celah. Perbandingan ini asosiasi, bukan eksperimen terkontrol. Arus tidak saya koreksi.</p>

<h3>2.3 Gambar rusak dan rekaman ulangan</h3>
<p>Sebagian video memutar ulang cuplikan yang sudah pernah terekam. Jam pada gambar dibaca tiap 10 detik; menit yang mayoritas jamnya sudah pernah muncul ditandai <code>flag_ulang</code> ({num(A['flag']['ulang'])} menit, termasuk yang juga kena flag lain).</p>
<p>Satu frame tiap 5 detik diperiksa pada semua video. Kerusakan berat jarang: hanya {A['flag']['frame_rusak']} menit melewati ambang. Detektornya heuristik dan hanya menangkap kerusakan yang jelas.</p>
{fig('13_contoh_rusak', 'Contoh frame rusak berat (24 November, dini hari).', True)}

<p>Streaming yang rusak sebagian (galat dekode H.264) tidak tertangkap detektor ini dan belum difilter. Arus pada periode tersebut dapat lebih rendah dari kenyataan.</p>

<h3>2.4 Mobil parkir di ROI mendekati</h3>
{fig('12_contoh_parkir', 'Mobil di tepi kanan (kotak kuning) berada di luar jalur, tetapi titik bawah kotaknya masuk ROI ungu.', True)}
{table(['Menit malam (22.00-04.59), arah mendekati', 'Menit', 'Rasio berhenti', 'Kendaraan di ROI', 'Okupansi (%)'],
       [['Dengan mobil parkir', num(na['menit']), num(na['rasio_berhenti'], 2), num(na['kendaraan_di_roi'], 2), num(na['okupansi'], 1)],
        ['Tanpa mobil parkir', num(nb['menit']), num(nb['rasio_berhenti'], 2), num(nb['kendaraan_di_roi'], 2), num(nb['okupansi'], 1)]])}
<p>Arus dari garis hitung tidak terpengaruh langsung oleh mobil diam. Yang terpengaruh: okupansi, DJ ruang, status gerak, komposisi, dan label kondisi. Penanda hanya memantau satu tempat parkir; tempat lain belum diperiksa.</p>

<h3>2.5 Waktu</h3>
<p>Kolom waktu berasal dari nama file. Pada 24 video contoh, jam yang tercetak di gambar tertinggal sekitar 13 menit pada 20 video, sekitar 16 menit pada 1 video, dan hampir sama (di bawah 35 detik) pada 3 video. Belum diketahui jam mana yang benar. Ini penting bila data digabung dengan sumber lain, misalnya cuaca.</p>

<h2>3. Distribusi</h2>
{describe}
{fig('07_distribusi', 'Distribusi per menit. Garis putus-putus adalah median; batang paling kanan menampung nilai di atas batas sumbu.')}
<p>Arus dan DJ berpuncak dua: malam sepi dan siang-sore ramai. DJ efektif melebihi 1 hanya pada {A['dj_lebih_1']} menit (menjauhi) dan {B['dj_lebih_1']} menit (mendekati). Nilai arus maksimum ({desc(A, 'est_flow_veh_per_hour', 'max')} kend/jam) adalah puncak satu menit dan belum diverifikasi manual.</p>

<h2>4. Pola waktu</h2>
{fig('03_profil_jam', 'Profil harian kedua arah.')}
{hourly}
<p>Arah mendekati memuncak pagi (pukul {pb[0]:02d}.00), arah menjauhi sore sampai malam (pukul {pa[0]:02d}.00). Pola ini sesuai dengan arus komuter satu arah pagi dan arah balik sore, bila arah kamera memang mewakili arah perjalanan itu.</p>
{fig('04_hari_kerja', 'Median arus hari kerja dan akhir pekan.')}
<p>Median keseluruhan hari kerja dan akhir pekan: {num(A['weekday_weekend_arus']['hari_kerja'])} dan {num(A['weekday_weekend_arus']['akhir_pekan'])} (menjauhi), {num(B['weekday_weekend_arus']['hari_kerja'])} dan {num(B['weekday_weekend_arus']['akhir_pekan'])} (mendekati). Puncak pagi dan sore lebih tajam pada hari kerja. Akhir pekan 21-22 November banyak menit tidak layak, jadi perbandingan ini bertumpu pada sedikit hari.</p>
{fig('05_arus_tanggal_jam', 'Median arus per tanggal dan jam.')}
{fig('11_kondisi', 'Rata-rata porsi waktu Padat dan Macet per jam.')}

<h2>5. Komposisi dan hubungan antarvariabel</h2>
{composition}
{fig('06_komposisi', 'Rata-rata kendaraan yang terlihat di ROI per jam. Ini kepadatan di ROI, bukan jumlah yang melintas.')}
<p>Porsi mobil yang tinggi pada dini hari di arah mendekati sebagian dipengaruhi mobil parkir. Pemisahan motor dan mobil pada malam hari belum divalidasi.</p>
{fig('08_arus_okupansi', 'Arus terhadap okupansi per menit.')}
<p>Di arah menjauhi, arus naik bersama okupansi sampai sekitar 10%, lalu mendatar atau turun pada okupansi tinggi. Itu bentuk yang diharapkan ketika lalu lintas mulai tersendat. Arah mendekati hampir tidak pernah mencapai wilayah tersebut.</p>
{fig('09_korelasi', 'Korelasi Spearman antarvariabel.')}
<ul>
<li>DJ efektif ditentukan oleh DJ arus pada {num(A['dj_dari_arus_pct'], 1)}% menit (menjauhi) dan {num(B['dj_dari_arus_pct'], 1)}% (mendekati).</li>
<li><code>est_q_smp_per_hour</code> dan <code>avg_dj_flow</code> adalah variabel yang sama dengan skala berbeda (DJ = SMP / 3.264).</li>
<li>Korelasi antar arah pada menit yang sama: arus {num(S['korelasi_antar_arah']['est_flow_veh_per_hour'], 2)}, DJ {num(S['korelasi_antar_arah']['avg_dj_effective'], 2)}.</li>
</ul>

<h2>6. Catatan untuk pemodelan</h2>
<ul>
<li>Saring <code>layak == True</code>. Untuk arah mendekati, pertimbangkan juga <code>flag_parkir == False</code> bila memakai okupansi atau status gerak.</li>
<li><b>Autokorelasi tinggi</b>: arus berkorelasi {num(A['autokorelasi_arus']['1'], 2)} / {num(B['autokorelasi_arus']['1'], 2)} dengan menit sebelumnya dan {num(A['autokorelasi_arus']['60'], 2)} / {num(B['autokorelasi_arus']['60'], 2)} dengan satu jam sebelumnya. Bagi data latih dan uji per hari, bukan acak per menit.</li>
<li><b>Label kondisi dibuat dengan aturan</b> dari okupansi, DJ, dan status gerak. Model yang memakai variabel itu untuk menebak label hanya mempelajari ulang aturannya.</li>
<li><b>Kelas timpang</b>: dengan {A['kondisi'].get('Macet (Congestion)', 0)} menit Macet, klasifikasi tiga kelas tidak akan andal.</li>
<li>Kapasitas, okupansi, pilihan EMP, dan label bergantung pada jumlah lajur dan panjang ROI yang belum diukur. Mengubahnya akan menggeser nilai DJ dan label.</li>
<li>EMP berganti menurut arus ({num(A['emp_high_pct'], 1)}% menit menjauhi memakai EMP arus tinggi), sehingga SMP bisa melompat di sekitar ambang.</li>
</ul>

<h2>7. Langkah yang disarankan</h2>
<ol>
<li>Perbaiki batas ROI mendekati agar tempat parkir tidak masuk, lalu hitung ulang fitur (tanpa deteksi ulang).</li>
<li>Periksa proses perekaman sejak 21 November (koneksi, beban perekam, ruang disk).</li>
<li>Ukur jumlah lajur dan panjang ROI di lapangan.</li>
<li>Hitung manual beberapa klip lagi, terutama malam hari dan jam puncak.</li>
<li>Pastikan jam acuan (nama file atau jam kamera).</li>
</ol>

<h2>8. File</h2>
<ul>
<li><code>eda/data/dataset_menit_berflag.csv</code>: gabungan dua arah, satu baris per menit per arah, dengan flag mutu.</li>
<li><code>eda/data/statistik.json</code>: semua angka ringkasan.</li>
<li><code>eda/pindai_mutu.py</code>, <code>eda/pindai_ulang.py</code>, <code>eda/analisis.py</code>, <code>eda/buat_laporan.py</code>: skrip untuk mengulang.</li>
</ul>
</main></body></html>"""

(E / "laporan_eda.html").write_text(html, encoding="utf-8")
print("ditulis:", E / "laporan_eda.html", f"{len(html) / 1e6:.1f} MB")
