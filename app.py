import os
import json
import threading
import time
import urllib.request
import csv
from pathlib import Path
from flask import Flask, render_template, jsonify

app = Flask(__name__)

CSV_PATH = Path("csv/cctv_bandung_data_validated.csv")
GLOBAL_CCTV_STATUS = {}
markers_global = []

def parse_location(value: str):
    parts = [part.strip() for part in (value or "").split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lng = float(parts[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return lat, lng

def load_markers():
    markers = []
    if not CSV_PATH.exists():
        return markers
        
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            point = parse_location(row.get("Location", ""))
            if point is None:
                continue
            lat, lng = point
            markers.append(
                {
                    "name": row.get("CCTV_Name", "").strip(),
                    "lat": lat,
                    "lng": lng,
                    "provider": row.get("Provider", "").strip(),
                    "stream": row.get("m3u8_URL", "").strip(),
                    "sampleDownloaded": row.get("Sample_Downloaded", "").strip(),
                    "sampleFile": row.get("Sample_File", "").strip(),
                    "sampleError": row.get("Sample_Error", "").strip(),
                }
            )
    return markers

def ping_cctv_worker():
    global GLOBAL_CCTV_STATUS, markers_global
    
    # Initialize statuses based on initial CSV read
    for m in markers_global:
        GLOBAL_CCTV_STATUS[m['name']] = m['sampleDownloaded']
        
    idx = 0
    batch_size = 10
    while True:
        if not markers_global:
            time.sleep(10)
            continue
            
        batch = markers_global[idx:idx+batch_size]
        for m in batch:
            url = m.get('stream')
            if not url:
                continue
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=5) as res:
                    if res.status == 200:
                        GLOBAL_CCTV_STATUS[m['name']] = 'YES'
                    else:
                        GLOBAL_CCTV_STATUS[m['name']] = 'NO'
            except Exception:
                GLOBAL_CCTV_STATUS[m['name']] = 'NO'
                
        idx += batch_size
        if idx >= len(markers_global):
            idx = 0
            
        time.sleep(60)

# Load markers initially
markers_global = load_markers()

# Start background ping thread
t = threading.Thread(target=ping_cctv_worker, daemon=True)
t.start()

@app.route("/")
def index():
    marker_json = json.dumps(markers_global, ensure_ascii=False)
    return render_template("index.html", marker_json=marker_json, marker_count=len(markers_global))

@app.route("/api/status")
def api_status():
    return jsonify(GLOBAL_CCTV_STATUS)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
