import csv
import time
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path

CSV_PATH = Path("../csv/cctv_bandung_data_validated.csv")
TIMEOUT_SECONDS = 5

def fetch_stream(url: str):
    start = time.time()
    result = {"url": url, "status": "UNKNOWN", "latency": 0.0, "error": None}
    try:
        # Kita gunakan GET dengan timeout singkat untuk mensimulasikan permintaan playlist HLS (m3u8)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as res:
            # Baca sedikit byte saja sekadar membuktikan file bisa diakses
            res.read(1024) 
            result["status"] = res.status
    except urllib.error.HTTPError as e:
        result["status"] = e.code
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        
    result["latency"] = time.time() - start
    return result

def main():
    if not CSV_PATH.exists():
        print(f"Error: File {CSV_PATH} tidak ditemukan.")
        return

    # 1. Parsing CSV untuk ekstrak URL
    urls = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Gunakan kolom m3u8_URL
            url = row.get("m3u8_URL")
            if url and url.startswith("http"):
                urls.append(url)
                
    if not urls:
        print("Tidak ada URL m3u8 yang valid di CSV.")
        return

    print(f"Total CCTV URL yang siap diuji: {len(urls)}")
    print("=" * 50)
    
    # Kelipatan 5 sampai 50
    concurrency_levels = list(range(5, 55, 5)) 
    
    for c in concurrency_levels:
        if c > len(urls):
            print(f"Melewati uji {c} karena jumlah CCTV hanya {len(urls)}")
            break
            
        print(f"[*] Menjalankan Gelombang Konkurensi: {c} koneksi simultan...")
        # Ambil sampel sebanyak c URL
        test_urls = urls[:c] 
        
        success_count = 0
        error_count = 0
        latencies = []
        
        start_batch = time.time()
        
        # Mengeksekusi permintaan secara paralel
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            results = list(executor.map(fetch_stream, test_urls))
            
        total_time = time.time() - start_batch
        
        # Rekap metrik
        for res in results:
            if res["status"] == 200:
                success_count += 1
                latencies.append(res["latency"])
            else:
                error_count += 1
                
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        print(f"    - Berhasil (HTTP 200) : {success_count}/{c}")
        print(f"    - Gagal / Timeout     : {error_count}/{c}")
        print(f"    - Rata-rata Latensi   : {avg_latency:.3f} detik")
        print(f"    - Total Waktu Batch   : {total_time:.3f} detik")
        print("-" * 50)
        
        # Beri jeda 3 detik antar gelombang agar server tidak langsung terhajar
        time.sleep(3)
        
    print("Benchmark selesai!")

if __name__ == "__main__":
    main()
