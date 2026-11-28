#!/usr/bin/env python3
"""Split working CCTV samples and annotator program into 4 standalone packages."""

import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
REPORT_JSON = os.path.join(SAMPLES_DIR, "rtsp_check_report.json")
PACKAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "roi_packages")

NUM_PACKAGES = 4

README_TEMPLATE = """# {package_name} — CCTV ROI Annotator

Paket ini berisi **{count} titik CCTV** untuk di-annotate ROI-nya.

## 1. Instalasi
Pastikan Python 3.10+ sudah terpasang, lalu jalankan:
```bash
pip install -r requirements.txt
```

## 2. Menjalankan Program
```bash
python annotator.py
```

## 3. Cara Annotate ROI
1. Isi **Nama Annotator** di panel kanan.
2. Pilih CCTV di daftar sebelah kiri.
3. Gunakan tombol **Play/Pause** (`Space`), slider, atau `-10 / -1 / +1 / +10` untuk mencari frame video yang paling jernih.
4. **Klik kiri** pada gambar untuk menaruh titik sudut ROI (default **Mode 4 Titik**, otomatis tertutup di titik ke-4; atau matikan centang untuk poligon bebas lalu klik kanan / tekan `Enter`).
5. Titik yang sudah jadi bisa **di-drag** untuk merapikan posisi.
6. Tekan `Ctrl+S` atau tombol **Simpan & Export Preview** untuk menyimpan.
7. Setelah semua CCTV selesai, kirimkan file `outputs/roi_annotations.json` ke koordinator.

## Daftar CCTV Paket Ini
{camera_list}
"""


def main():
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)

    ok_cams = [r for r in report if r.get("status") == "OK"]
    ok_cams.sort(key=lambda r: int(r["channel"]))

    # Round-robin / balanced chunk split into 4 packages
    chunks = [[] for _ in range(NUM_PACKAGES)]
    base_size, rem = divmod(len(ok_cams), NUM_PACKAGES)
    idx = 0
    for p in range(NUM_PACKAGES):
        size = base_size + (1 if p < rem else 0)
        chunks[p] = ok_cams[idx : idx + size]
        idx += size

    os.makedirs(PACKAGES_DIR, exist_ok=True)

    for p_idx, cams in enumerate(chunks, start=1):
        pkg_name = f"paket_{p_idx}"
        pkg_dir = os.path.join(PACKAGES_DIR, pkg_name)
        pkg_samples = os.path.join(pkg_dir, "samples")
        pkg_outputs = os.path.join(pkg_dir, "outputs", "previews")
        os.makedirs(pkg_samples, exist_ok=True)
        os.makedirs(pkg_outputs, exist_ok=True)

        # Clear old sample files in this package
        for old_f in os.listdir(pkg_samples):
            old_p = os.path.join(pkg_samples, old_f)
            if os.path.isfile(old_p):
                os.remove(old_p)

        # Copy annotator.py and requirements.txt
        shutil.copy2(
            os.path.join(BASE_DIR, "annotator.py"),
            os.path.join(pkg_dir, "annotator.py"),
        )
        shutil.copy2(
            os.path.join(BASE_DIR, "requirements.txt"),
            os.path.join(pkg_dir, "requirements.txt"),
        )

        # Copy assigned sample images
        for cam in cams:
            src_img = os.path.join(SAMPLES_DIR, cam["filename"])
            dst_img = os.path.join(pkg_samples, cam["filename"])
            shutil.copy2(src_img, dst_img)

        # Write package_info.json
        info = {
            "package_name": pkg_name,
            "annotator": f"Anggota_{p_idx}",
            "cameras": [
                {
                    "channel": c["channel"],
                    "name": c["name"],
                    "filename": c["filename"],
                    "resolution": f"{c['width']}x{c['height']}",
                }
                for c in cams
            ],
        }
        with open(os.path.join(pkg_dir, "package_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        # Write README.md
        cam_lines = "\n".join(
            f"- `{c['channel']}` — {c['name']} (`{c['filename']}`, {c['width']}x{c['height']})"
            for c in cams
        )
        readme = README_TEMPLATE.format(
            package_name=pkg_name.upper(),
            count=len(cams),
            camera_list=cam_lines,
        )
        with open(os.path.join(pkg_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)

        print(f"Created {pkg_dir} with {len(cams)} cameras.")


if __name__ == "__main__":
    main()
