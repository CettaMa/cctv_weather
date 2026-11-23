import csv
import time
import urllib.request
import urllib.error
import urllib.parse
import concurrent.futures
from pathlib import Path

CSV_PATH = Path("../csv/cctv_bandung_data_validated.csv")
TIMEOUT_SECONDS = 15 # Because we are downloading a full video chunk

def fetch_first_ts_url(m3u8_url):
    req = urllib.request.Request(m3u8_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=5) as res:
        lines = res.read().decode('utf-8').splitlines()
        
    for line in lines:
        if line.strip() and not line.startswith('#'):
            # This is a URL. It might be a .ts file or another .m3u8 (master playlist)
            full_url = urllib.parse.urljoin(m3u8_url, line.strip())
            if line.endswith('.m3u8') or '.m3u8' in line:
                return fetch_first_ts_url(full_url)
            return full_url
    return None

def scrape_video_chunk(url: str):
    start = time.time()
    result = {"url": url, "status": "UNKNOWN", "latency": 0.0, "size_bytes": 0, "error": None}
    
    try:
        ts_url = fetch_first_ts_url(url)
        if not ts_url:
            raise Exception("Tidak menemukan segmen .ts di dalam playlist.")
            
        req = urllib.request.Request(ts_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as res:
            content = res.read()
            result["size_bytes"] = len(content)
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
        print(f"Error: {CSV_PATH} tidak ditemukan.")
        return

    urls = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("m3u8_URL")
            if url and url.startswith("http") and row.get("Sample_Downloaded") == "YES":
                urls.append(url)
                
    if not urls:
        print("Tidak ada URL m3u8.")
        return
        
    concurrency_levels = [30]
    print(f"Mulai Benchmark Scraping Video (Total CCTV siap diuji: {len(urls)})")
    print("=" * 50)
    
    for c in concurrency_levels:
        if c > len(urls): break
            
        print(f"[*] Menjalankan Gelombang: {c} koneksi simultan (unduh .ts)...")
        test_urls = urls[:c] 
        
        success_count = 0
        error_count = 0
        latencies = []
        total_size = 0
        
        start_batch = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            results = list(executor.map(scrape_video_chunk, test_urls))
            
        total_time = time.time() - start_batch
        
        for res in results:
            if res["status"] == 200:
                success_count += 1
                latencies.append(res["latency"])
                total_size += res["size_bytes"]
            else:
                error_count += 1
                
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_mb = total_size / (1024 * 1024)
        throughput_mbps = (total_mb / total_time) if total_time > 0 else 0
        
        print(f"    - Berhasil Unduh    : {success_count}/{c} chunk")
        print(f"    - Gagal / Timeout   : {error_count}/{c}")
        print(f"    - Total Data (MB)   : {total_mb:.2f} MB")
        print(f"    - Rata-rata Latensi : {avg_latency:.3f} detik per video")
        print(f"    - Throughput Global : {throughput_mbps:.2f} MB/detik")
        print(f"    - Total Waktu Batch : {total_time:.3f} detik")
        print("-" * 50)
        
        time.sleep(3)

if __name__ == "__main__":
    main()
