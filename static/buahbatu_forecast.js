/*
 * Panel ramalan traffic + cuaca (mode replay) untuk CCTV TERUSAN BUAH BATU-JEMBATAN TOL 02.
 *
 * Data: /api/buahbatu/forecast/meta dan /api/buahbatu/forecast?asal=<ISO>, dibuat oleh
 * extractions/forecast/ekspor_app.py. Panel muncul saat marker kamera ini diklik
 * (index.html mengirim event "cctv:select"). Dua slider horizon (0..+360 menit, langkah 5):
 * - Terkunci: traffic dan cuaca selalu di horizon yang sama.
 * - Bebas: masing-masing slider punya horizon sendiri.
 * Hasil ditampilkan sebagai metrik, grafik kecil, dan dua layer peta: heatmap kepadatan yang hanya
 * menutupi badan Jalan Buah Batu (geometri OpenStreetMap, warna menurut arus per arah pada rentang yang
 * bisa diatur) dan lingkaran hujan berjari-jari 1 km di sekitar kamera.
 */
(function () {
  "use strict";

  const CAMERA_NAME = "CCTV TERUSAN BUAH BATU-JEMBATAN TOL 02";
  const RAIN_RADIUS_M = 1000;
  const ROAD_URL = "/static/buahbatu_jalan.json";
  const RAIN_MAX = 5;            // mm/jam untuk cincin paling pekat
  const DIRS = ["menjauhi", "mendekati"];
  const DIR_LABEL = { menjauhi: "Menjauhi", mendekati: "Mendekati" };
  const DIR_COLOR = { menjauhi: "#2a78d6", mendekati: "#eb6834" };
  // warna heatmap dari batas bawah ke batas atas rentang: hijau, kuning, jingga, merah, merah tua
  const HEAT_COLORS = [[16, 185, 129], [250, 204, 21], [249, 115, 22], [220, 38, 38], [112, 16, 48]];
  const HEAT_STEP = 100;         // kend/jam, pembulatan rentang otomatis
  const HEAT_MIN_SPAN = 100;

  const state = {
    meta: null,
    data: null,
    road: null,
    mode: "terkunci",
    hTraffic: 0,
    hWeather: 0,
    showTraffic: true,
    showRain: true,
    heat: { auto: true, min: 0, max: 6000 },   // rentang warna heatmap, kend/jam
    control: null,
    layer: null,
    charts: { traffic: null, rain: null },
  };

  // ---------------------------------------------------------------- util
  const $ = (root, sel) => root.querySelector(sel);
  const fmtInt = (v) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Math.round(v).toLocaleString("id-ID"));
  const fmt1 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "-" : v.toLocaleString("id-ID", { maximumFractionDigits: 1, minimumFractionDigits: 1 }));
  const cssVar = (name, fallback) => (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fallback;

  function timeAt(h) {
    const t0 = new Date(state.data.asal);
    return new Date(t0.getTime() + h * state.meta.langkah_menit * 60000);
  }

  function hhmm(d) {
    // waktu WIB (UTC+7) tanpa bergantung pada zona waktu browser
    const w = new Date(d.getTime() + 7 * 3600000);
    return `${String(w.getUTCHours()).padStart(2, "0")}.${String(w.getUTCMinutes()).padStart(2, "0")}`;
  }

  function dayLabel(iso) {
    const w = new Date(new Date(iso).getTime() + 7 * 3600000);
    const hari = ["Min", "Sen", "Sel", "Rab", "Kam", "Jum", "Sab"][w.getUTCDay()];
    return `${hari} ${w.getUTCDate()} Nov`;
  }

  function seriAt(key, h) {
    const arr = state.data.seri[key] || [];
    const v = arr[state.data.offset_asal + h];
    return v === undefined ? null : v;
  }

  function trafficAt(dir, key, h) {
    if (h === 0) return seriAt(`${key}_${dir}`, 0);
    const p = state.data.pred[dir][key];
    return p ? p[h - 1] : null;
  }

  function condition(dj) {
    if (dj === null || Number.isNaN(dj)) return { cls: "", text: "-" };
    if (dj >= state.meta.dj_macet) return { cls: "macet", text: "Macet" };
    if (dj >= state.meta.dj_padat) return { cls: "padat", text: "Padat" };
    return { cls: "lancar", text: "Lancar" };
  }

  function djAt(dir, h) {
    const smp = trafficAt(dir, "smp", h);
    return smp === null ? null : smp / state.meta.kapasitas_smp_jam;
  }

  function rainClass(mm) {
    if (mm === null) return "-";
    if (mm < 0.1) return "Tidak hujan";
    if (mm < 2.5) return "Hujan ringan";
    if (mm < 7.6) return "Hujan sedang";
    return "Hujan lebat";
  }

  // posisi arus di dalam rentang heatmap, 0..1 (di luar rentang dijepit ke ujungnya)
  function heatPos(flow) {
    const { min, max } = state.heat;
    return Math.max(0, Math.min(1, (flow - min) / (max - min)));
  }

  function heatColor(t) {
    const x = t * (HEAT_COLORS.length - 1);
    const i = Math.min(Math.floor(x), HEAT_COLORS.length - 2);
    return HEAT_COLORS[i].map((c, k) => Math.round(c + (HEAT_COLORS[i + 1][k] - c) * (x - i)));
  }

  // rentang otomatis: arus terkecil dan terbesar dua arah sepanjang horizon dari waktu acuan ini
  function autoRange() {
    const values = DIRS.flatMap((d) => Array.from({ length: state.meta.h_max + 1 }, (_, h) => trafficAt(d, "flow", h)))
      .filter((v) => v !== null && !Number.isNaN(v));
    if (!values.length) return;
    const min = Math.floor(Math.min(...values) / HEAT_STEP) * HEAT_STEP;
    const max = Math.ceil(Math.max(...values) / HEAT_STEP) * HEAT_STEP;
    state.heat.min = min;
    state.heat.max = Math.max(max, min + HEAT_MIN_SPAN);
  }

  // isi kotak angka dan legenda; kotak yang sedang diketik tidak ditimpa
  function renderHeatControls() {
    const root = state.control && state.control.getContainer();
    if (!root || !state.data) return;
    ["min", "max"].forEach((k) => {
      const input = $(root, `#bbfHeat_${k}`);
      if (document.activeElement !== input) input.value = state.heat[k];
    });
    $(root, "#bbfHeatAuto").setAttribute("aria-pressed", String(state.heat.auto));
    const { min, max } = state.heat;
    $(root, "#bbfLegendTicks").innerHTML = [0, 0.5, 1].map((t) => `<span>${fmtInt(min + (max - min) * t)}</span>`).join("");
    $(root, "#bbfLegendMarks").innerHTML = DIRS.map((d) => {
      const flow = trafficAt(d, "flow", state.hTraffic);
      if (flow === null) return "";
      const out = flow < min ? " (di bawah rentang)" : flow > max ? " (di atas rentang)" : "";
      return `<span class="bbf-mark" style="left:${(heatPos(flow) * 100).toFixed(1)}%" ` +
        `title="${DIR_LABEL[d]}: ${fmtInt(flow)} kend/jam${out}">${d === "menjauhi" ? "▼" : "▽"}</span>`;
    }).join("");
  }

  function setHeatRange(root) {
    const min = parseFloat($(root, "#bbfHeat_min").value);
    const max = parseFloat($(root, "#bbfHeat_max").value);
    const valid = Number.isFinite(min) && Number.isFinite(max) && min >= 0 && max - min >= HEAT_MIN_SPAN;
    $(root, "#bbfHeatError").textContent = valid ? "" : `Batas atas harus minimal ${HEAT_MIN_SPAN} di atas batas bawah.`;
    if (!valid) return;
    state.heat = { auto: false, min, max };
    renderHeatControls();
    renderLayers();
  }

  // ---------------------------------------------------------------- layer peta
  function setGradient(svg, id, stops) {
    let defs = svg.querySelector("defs#bbf-defs");
    if (!defs) {
      defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.setAttribute("id", "bbf-defs");
      svg.insertBefore(defs, svg.firstChild);
    }
    let g = defs.querySelector(`#${id}`);
    if (!g) {
      g = document.createElementNS("http://www.w3.org/2000/svg", "radialGradient");
      g.setAttribute("id", id);
      defs.appendChild(g);
    }
    g.innerHTML = stops.map(([off, color, op]) => `<stop offset="${off}" stop-color="${color}" stop-opacity="${op}"/>`).join("");
  }

  // poligon selebar `width` meter di sekeliling garis tengah jalan, supaya warna hanya menutupi badan jalan
  function roadPolygon(coords, width) {
    const mLat = 110574, mLon = 111320 * Math.cos((coords[0][0] * Math.PI) / 180);
    const pts = coords.map(([lat, lon]) => [lon * mLon, lat * mLat]);
    const normals = pts.slice(1).map((p, i) => {
      const dx = p[0] - pts[i][0], dy = p[1] - pts[i][1], len = Math.hypot(dx, dy) || 1;
      return [-dy / len, dx / len];
    });
    const side = (sign) => pts.map((p, i) => {
      const a = normals[Math.max(i - 1, 0)], b = normals[Math.min(i, normals.length - 1)];
      let nx = a[0] + b[0], ny = a[1] + b[1];
      const len = Math.hypot(nx, ny) || 1;
      nx /= len;
      ny /= len;
      return [(p[1] + (sign * ny * width) / 2) / mLat, (p[0] + (sign * nx * width) / 2) / mLon];
    });
    return side(1).concat(side(-1).reverse());
  }

  function renderRain() {
    const rain = seriAt("hujan", state.hWeather);
    if (!state.showRain || rain === null) return;
    const wet = rain >= 0.1;
    const disk = L.circle([state.meta.kamera.lat, state.meta.kamera.lon], {
      radius: RAIN_RADIUS_M, interactive: false, renderer: state.renderer,
      color: "#60a5fa", weight: 1.5, opacity: 0.7, dashArray: wet ? null : "6 6", fillOpacity: wet ? 1 : 0,
    }).addTo(state.layer);
    const svg = disk._path && disk._path.ownerSVGElement;
    if (!svg || !wet) return;
    const a = 0.14 + 0.36 * Math.min(1, rain / RAIN_MAX);
    setGradient(svg, "bbf-rain-grad", [["0%", "#2563eb", a], ["75%", "#2563eb", a * 0.85], ["100%", "#60a5fa", a * 0.25]]);
    disk._path.setAttribute("fill", "url(#bbf-rain-grad)");
  }

  function renderRoad() {
    if (!state.showTraffic || !state.road) return;
    state.road.ruas.forEach((r) => {
      // ruas dua arah di simpang memakai arah yang lebih padat
      const flows = r.arah.map((d) => trafficAt(d, "flow", state.hTraffic)).filter((v) => v !== null);
      if (!flows.length) return;
      const t = heatPos(Math.max(...flows));
      const color = `rgb(${heatColor(t).join(",")})`;
      const heat = 0.75 + 0.25 * t;
      const rows = r.arah.map((d) => `${DIR_LABEL[d]}: ${fmtInt(trafficAt(d, "flow", state.hTraffic))} kend/jam · ${condition(djAt(d, state.hTraffic)).text}`);
      // tiga pita bertumpuk: tepi tipis, tengah pekat
      [[1, 0.45], [0.68, 0.4], [0.34, 0.5]].forEach(([part, alpha], i) => {
        const poly = L.polygon(roadPolygon(r.koordinat, r.lebar_m * part), {
          stroke: false, fillColor: color, fillOpacity: alpha * heat, interactive: i === 0, renderer: state.renderer,
        }).addTo(state.layer);
        if (i === 0) poly.bindTooltip(`+${state.hTraffic * state.meta.langkah_menit} mnt<br>${rows.join("<br>")}`, { sticky: true });
      });
    });
  }

  function renderLayers() {
    if (!state.data) return;
    if (!state.layer) {
      state.layer = L.layerGroup().addTo(map);
      // peta memakai renderer canvas; gradien radial butuh SVG, jadi layer ini punya renderer sendiri
      state.renderer = L.svg({ padding: 0.2 });
    }
    state.layer.clearLayers();
    renderRain();
    renderRoad();
    L.circleMarker([state.meta.kamera.lat, state.meta.kamera.lon], {
      radius: 4, color: cssVar("--text-strong", "#fff"), weight: 1.5, fillOpacity: 1,
      fillColor: cssVar("--accent", "#38bdf8"), interactive: false, renderer: state.renderer,
    }).addTo(state.layer);
  }

  // geser peta ke badan jalan atau ke seluruh area cuaca, tanpa tertutup panel
  function focusMap(target) {
    const center = L.latLng(state.meta.kamera.lat, state.meta.kamera.lon);
    const bounds = target === "road" && state.road
      ? L.latLngBounds(state.road.ruas.flatMap((r) => r.koordinat))
      : center.toBounds(RAIN_RADIUS_M * 2);
    const panel = window.innerWidth > 600 ? 370 : 0;
    map.fitBounds(bounds, { paddingTopLeft: [panel + 20, 20], paddingBottomRight: [20, 20] });
  }

  // ---------------------------------------------------------------- grafik
  function chartColors() {
    return { grid: cssVar("--chart-grid", "rgba(30,41,59,0.5)"), text: cssVar("--muted", "#64748b") };
  }

  function buildCharts(root) {
    const { grid, text } = chartColors();
    const H = state.meta.h_max;
    const labels = Array.from({ length: H + 1 }, (_, h) => hhmm(timeAt(h)));
    const common = {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: grid }, ticks: { color: text, maxTicksLimit: 7, font: { size: 9 } } },
        y: { grid: { color: grid }, ticks: { color: text, maxTicksLimit: 4, font: { size: 9 } }, beginAtZero: true },
      },
    };
    Object.values(state.charts).forEach((c) => c && c.destroy());

    const datasets = [];
    DIRS.forEach((d) => {
      datasets.push({
        label: `Ramalan ${DIR_LABEL[d]}`, data: Array.from({ length: H + 1 }, (_, h) => trafficAt(d, "flow", h)),
        borderColor: DIR_COLOR[d], backgroundColor: DIR_COLOR[d], borderWidth: 2, pointRadius: 0, tension: 0.25, spanGaps: true,
      });
      datasets.push({
        label: `Aktual ${DIR_LABEL[d]}`, data: Array.from({ length: H + 1 }, (_, h) => seriAt(`flow_${d}`, h)),
        borderColor: DIR_COLOR[d], backgroundColor: DIR_COLOR[d], showLine: false, pointRadius: 1.8, pointStyle: "circle",
      });
    });
    state.charts.traffic = new Chart($(root, "#bbfTrafficChart"), {
      type: "line", data: { labels, datasets },
      options: {
        ...common,
        plugins: {
          legend: { display: true, position: "bottom", labels: { color: text, boxWidth: 8, boxHeight: 8, font: { size: 9 }, filter: (i) => i.text.startsWith("Ramalan") } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtInt(c.parsed.y)} kend/jam` } },
        },
      },
      plugins: [markerPlugin(() => state.hTraffic)],
    });

    state.charts.rain = new Chart($(root, "#bbfRainChart"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Hujan (mm/jam)", data: Array.from({ length: H + 1 }, (_, h) => seriAt("hujan", h)), backgroundColor: "#2563eb", borderRadius: 1, barPercentage: 1, categoryPercentage: 0.9 }] },
      options: {
        ...common,
        scales: { ...common.scales, y: { ...common.scales.y, suggestedMax: 1 } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `Hujan: ${fmt1(c.parsed.y)} mm/jam` } } },
      },
      plugins: [markerPlugin(() => state.hWeather)],
    });
  }

  // garis vertikal di posisi slider
  function markerPlugin(getH) {
    return {
      id: "bbfMarker",
      afterDatasetsDraw(chart) {
        const x = chart.scales.x.getPixelForValue(getH());
        const { top, bottom } = chart.chartArea;
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = cssVar("--text-strong", "#f8fafc");
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 2]);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
        ctx.restore();
      },
    };
  }

  // ---------------------------------------------------------------- panel
  const HEAT_GRADIENT = `linear-gradient(90deg,${HEAT_COLORS.map((c) => `rgb(${c.join(",")})`).join(",")})`;

  function panelHtml() {
    const origins = state.meta.asal;
    const byDay = {};
    origins.forEach((o) => { (byDay[dayLabel(o.t)] = byDay[dayLabel(o.t)] || []).push(o); });
    const options = Object.entries(byDay).map(([day, list]) =>
      `<optgroup label="${day}">${list.map((o) => `<option value="${o.t}">${day} ${hhmm(new Date(o.t))}</option>`).join("")}</optgroup>`
    ).join("");
    const H = state.meta.h_max;
    return `
      <div class="bbf-head">
        <div>
          <div class="bbf-title">Ramalan traffic &amp; cuaca</div>
          <div class="bbf-sub">Terusan Buah Batu – Jembatan Tol 02 · mode replay</div>
        </div>
        <button type="button" class="bbf-close" aria-label="Tutup panel ramalan">✕</button>
      </div>
      <div class="bbf-body">
        <div>
          <div class="bbf-label"><label for="bbfOrigin">Waktu acuan (0 menit)</label></div>
          <div class="bbf-row">
            <button type="button" class="bbf-btn" data-step="-1" aria-label="Waktu acuan sebelumnya">‹</button>
            <select id="bbfOrigin">${options}</select>
            <button type="button" class="bbf-btn" data-step="1" aria-label="Waktu acuan berikutnya">›</button>
          </div>
        </div>
        <div>
          <div class="bbf-label">Mode slider</div>
          <div class="bbf-seg" role="group" aria-label="Mode slider">
            <button type="button" data-mode="terkunci" aria-pressed="true">Terkunci</button>
            <button type="button" data-mode="bebas" aria-pressed="false">Bebas</button>
          </div>
          <div class="bbf-seg-hint" id="bbfModeHint">Traffic dan cuaca selalu di horizon yang sama.</div>
        </div>
        <div class="bbf-slider">
          <div class="bbf-slider-top"><label class="bbf-slider-name" for="bbfTraffic">Traffic</label><span class="bbf-slider-val" id="bbfTrafficVal"></span></div>
          <input id="bbfTraffic" type="range" min="0" max="${H}" step="1" value="0">
        </div>
        <div class="bbf-slider">
          <div class="bbf-slider-top"><label class="bbf-slider-name" for="bbfWeather">Cuaca</label><span class="bbf-slider-val" id="bbfWeatherVal"></span></div>
          <input id="bbfWeather" type="range" min="0" max="${H}" step="1" value="0">
        </div>
        <div>
          <div class="bbf-label" id="bbfTrafficHead">Traffic</div>
          <div class="bbf-grid" id="bbfTrafficCards"></div>
        </div>
        <div class="bbf-chart"><canvas id="bbfTrafficChart" aria-label="Grafik ramalan arus per arah"></canvas></div>
        <div>
          <div class="bbf-label" id="bbfWeatherHead">Cuaca</div>
          <div class="bbf-grid" id="bbfWeatherCards"></div>
        </div>
        <div class="bbf-chart" style="height: 84px;"><canvas id="bbfRainChart" aria-label="Grafik hujan per 5 menit"></canvas></div>
        <div>
          <div class="bbf-label">Layer peta</div>
          <div class="bbf-toggles">
            <label><input type="checkbox" id="bbfShowTraffic" checked> <span class="bbf-swatch" style="background: ${HEAT_GRADIENT}"></span> Kepadatan jalan</label>
            <label><input type="checkbox" id="bbfShowRain" checked> <span class="bbf-swatch" style="background:#2563eb"></span> Hujan (radius ${RAIN_RADIUS_M / 1000} km)</label>
          </div>
          <div class="bbf-heat">
            <div class="bbf-heat-top">
              <span>Rentang warna jalan (kend/jam)</span>
              <button type="button" class="bbf-btn bbf-toggle" id="bbfHeatAuto" aria-pressed="true" title="Ikuti nilai terkecil dan terbesar ramalan dari waktu acuan">Otomatis</button>
            </div>
            <div class="bbf-row">
              <label class="bbf-num">Bawah <input type="number" id="bbfHeat_min" min="0" step="${HEAT_STEP}" inputmode="numeric"></label>
              <label class="bbf-num">Atas <input type="number" id="bbfHeat_max" min="0" step="${HEAT_STEP}" inputmode="numeric"></label>
            </div>
            <div class="bbf-error" id="bbfHeatError" role="alert"></div>
            <div class="bbf-legend">
              <div class="bbf-legend-marks" id="bbfLegendMarks"></div>
              <div class="bbf-legend-bar" style="background: ${HEAT_GRADIENT}"></div>
              <div class="bbf-legend-ticks" id="bbfLegendTicks"></div>
            </div>
            <div class="bbf-seg-hint">▼ menjauhi · ▽ mendekati (horizon slider traffic)</div>
          </div>
          <div class="bbf-row" style="margin-top: 6px;">
            <button type="button" class="bbf-btn" data-focus="road">Lihat jalan</button>
            <button type="button" class="bbf-btn" data-focus="rain">Lihat area cuaca</button>
          </div>
          <div class="bbf-seg-hint">Seluruh ruas memakai ramalan di titik kamera; hujan adalah nilai satu titik, bukan peta sebaran.</div>
        </div>
        <div class="bbf-note" id="bbfNote"></div>
      </div>`;
  }

  function cardsTraffic(h) {
    const parts = DIRS.map((d) => {
      const pred = trafficAt(d, "flow", h);
      const act = h > 0 ? seriAt(`flow_${d}`, h) : null;
      const dj = djAt(d, h);
      const c = condition(dj);
      let sub = "teramati";
      if (h > 0 && act !== null && pred !== null) {
        const diff = Math.round(pred - act);
        sub = `aktual ${fmtInt(act)} · selisih ${diff > 0 ? "+" : ""}${fmtInt(diff)}`;
      } else if (h > 0) {
        sub = seriAt("ulang", h) ? "rekaman ulangan, diabaikan" : "aktual tidak terekam";
      }
      return `<div class="bbf-card"><div class="k">Arus ${DIR_LABEL[d].toLowerCase()}</div>
        <div class="v">${fmtInt(pred)}<span class="u">kend/jam</span></div>
        <div class="s">${sub}</div>
        <div class="bbf-cond ${c.cls}" title="Derajat kejenuhan dari SMP/jam ÷ kapasitas">${c.text} · DJ ${dj === null ? "-" : dj.toFixed(2)}</div></div>`;
    });
    const smp = DIRS.map((d) => trafficAt(d, "smp", h));
    const mae = h > 0 ? state.meta.mae_flow_per_h[h - 1] : null;
    parts.push(`<div class="bbf-card"><div class="k">SMP total</div><div class="v">${fmtInt(smp.every((v) => v !== null) ? smp[0] + smp[1] : null)}<span class="u">smp/jam</span></div><div class="s">kedua arah</div></div>`);
    parts.push(`<div class="bbf-card"><div class="k">Galat tipikal</div><div class="v">${h === 0 ? "-" : "±" + fmtInt(mae)}<span class="u">kend/jam</span></div><div class="s">MAE uji pada horizon ini</div></div>`);
    return parts.join("");
  }

  function cardsWeather(h) {
    const rain = seriAt("hujan", h);
    return [
      `<div class="bbf-card"><div class="k">Hujan</div><div class="v">${fmt1(rain)}<span class="u">mm/jam</span></div><div class="s">${rainClass(rain)}</div></div>`,
      `<div class="bbf-card"><div class="k">Suhu</div><div class="v">${fmt1(seriAt("suhu", h))}<span class="u">°C</span></div><div class="s">kelembapan ${fmtInt(seriAt("rh", h))}%</div></div>`,
      `<div class="bbf-card"><div class="k">Awan</div><div class="v">${fmtInt(seriAt("awan", h))}<span class="u">%</span></div></div>`,
      `<div class="bbf-card"><div class="k">Angin</div><div class="v">${fmt1(seriAt("angin", h))}<span class="u">km/jam</span></div></div>`,
    ].join("");
  }

  function sliderText(h) {
    return `+${h * state.meta.langkah_menit} mnt · ${hhmm(timeAt(h))} WIB`;
  }

  function updateView() {
    const root = state.control && state.control.getContainer();
    if (!root || !state.data) return;
    $(root, "#bbfTraffic").value = state.hTraffic;
    $(root, "#bbfWeather").value = state.hWeather;
    $(root, "#bbfTrafficVal").textContent = sliderText(state.hTraffic);
    $(root, "#bbfWeatherVal").textContent = sliderText(state.hWeather);
    $(root, "#bbfTraffic").setAttribute("aria-valuetext", sliderText(state.hTraffic));
    $(root, "#bbfWeather").setAttribute("aria-valuetext", sliderText(state.hWeather));
    $(root, "#bbfTrafficHead").textContent = `Traffic ${state.hTraffic === 0 ? "saat ini (teramati)" : "ramalan " + sliderText(state.hTraffic)}`;
    $(root, "#bbfWeatherHead").textContent = `Cuaca ${sliderText(state.hWeather)}`;
    $(root, "#bbfTrafficCards").innerHTML = cardsTraffic(state.hTraffic);
    $(root, "#bbfWeatherCards").innerHTML = cardsWeather(state.hWeather);
    Object.values(state.charts).forEach((c) => c && c.update("none"));
    renderHeatControls();
    renderLayers();
  }

  function setMode(mode) {
    state.mode = mode;
    const root = state.control.getContainer();
    root.querySelectorAll(".bbf-seg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    $(root, "#bbfModeHint").textContent = mode === "terkunci"
      ? "Traffic dan cuaca selalu di horizon yang sama."
      : "Traffic dan cuaca diatur sendiri-sendiri.";
    if (mode === "terkunci") state.hWeather = state.hTraffic;
    updateView();
  }

  async function loadOrigin(iso) {
    const root = state.control.getContainer();
    try {
      const res = await fetch(`/api/buahbatu/forecast?asal=${encodeURIComponent(iso)}`);
      const data = await res.json();
      if (data.status !== "success") throw new Error(data.reason || "gagal");
      state.data = data;
      if (state.heat.auto) autoRange();
    } catch (err) {
      console.error("Ramalan Buah Batu gagal dimuat:", err);
      if (typeof showToast === "function") showToast("Gagal memuat ramalan Buah Batu", "⚠️");
      return;
    }
    $(root, "#bbfOrigin").value = state.data.asal;
    $(root, "#bbfNote").innerHTML =
      `Ramalan arus: ${state.meta.model}; ${state.data.model} (out-of-sample). Cuaca: ${state.meta.cuaca_sumber}. ` +
      `${state.meta.catatan} Titik kamera ${state.meta.kamera.lat.toFixed(5)}, ${state.meta.kamera.lon.toFixed(5)}.`;
    buildCharts(root);
    updateView();
  }

  function wirePanel(root) {
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);
    $(root, ".bbf-close").addEventListener("click", closePanel);
    $(root, "#bbfOrigin").addEventListener("change", (e) => loadOrigin(e.target.value));
    root.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => {
      const sel = $(root, "#bbfOrigin");
      const i = Math.max(0, Math.min(sel.options.length - 1, sel.selectedIndex + parseInt(b.dataset.step, 10)));
      if (i !== sel.selectedIndex) loadOrigin(sel.options[i].value);
    }));
    root.querySelectorAll(".bbf-seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    $(root, "#bbfTraffic").addEventListener("input", (e) => {
      state.hTraffic = parseInt(e.target.value, 10) || 0;
      if (state.mode === "terkunci") state.hWeather = state.hTraffic;
      updateView();
    });
    $(root, "#bbfWeather").addEventListener("input", (e) => {
      state.hWeather = parseInt(e.target.value, 10) || 0;
      if (state.mode === "terkunci") state.hTraffic = state.hWeather;
      updateView();
    });
    $(root, "#bbfShowTraffic").addEventListener("change", (e) => { state.showTraffic = e.target.checked; renderLayers(); });
    $(root, "#bbfShowRain").addEventListener("change", (e) => { state.showRain = e.target.checked; renderLayers(); });
    ["min", "max"].forEach((k) => {
      const input = $(root, `#bbfHeat_${k}`);
      input.addEventListener("input", () => setHeatRange(root));
      input.addEventListener("blur", () => { $(root, "#bbfHeatError").textContent = ""; renderHeatControls(); });
    });
    $(root, "#bbfHeatAuto").addEventListener("click", () => {
      state.heat.auto = true;
      autoRange();
      $(root, "#bbfHeatError").textContent = "";
      renderHeatControls();
      renderLayers();
    });
    root.querySelectorAll("[data-focus]").forEach((b) => b.addEventListener("click", () => focusMap(b.dataset.focus)));
  }

  async function openPanel() {
    if (!state.meta) {
      try {
        const res = await fetch("/api/buahbatu/forecast/meta");
        const meta = await res.json();
        if (meta.status !== "success") throw new Error(meta.reason || "tidak tersedia");
        state.meta = meta;
      } catch (err) {
        console.error("Meta ramalan Buah Batu gagal dimuat:", err);
        if (typeof showToast === "function") showToast("Data ramalan Buah Batu belum tersedia", "⚠️");
        return;
      }
    }
    if (!state.road) {
      try {
        const res = await fetch(ROAD_URL);
        if (!res.ok) throw new Error(res.status);
        state.road = await res.json();
        if (map.attributionControl) map.attributionControl.addAttribution("Geometri jalan © OpenStreetMap");
      } catch (err) {
        console.error("Geometri Jalan Buah Batu gagal dimuat:", err);
      }
    }
    if (!state.control) {
      state.control = L.control({ position: "bottomleft" });
      state.control.onAdd = function () {
        const div = L.DomUtil.create("div", "bbf-panel leaflet-control");
        div.setAttribute("role", "region");
        div.setAttribute("aria-label", "Ramalan traffic dan cuaca Terusan Buah Batu");
        div.innerHTML = panelHtml();
        wirePanel(div);
        return div;
      };
      state.control.addTo(map);
      await loadOrigin(state.meta.asal_default || state.meta.asal[state.meta.asal.length - 1].t);
      focusMap("road");
    } else {
      renderLayers();
    }
  }

  function closePanel() {
    if (state.control) { map.removeControl(state.control); state.control = null; }
    if (state.layer) { map.removeLayer(state.layer); state.layer = null; }
    if (state.renderer) { map.removeLayer(state.renderer); state.renderer = null; }
    Object.values(state.charts).forEach((c) => c && c.destroy());
    state.charts = { traffic: null, rain: null };
    state.data = null;
    state.hTraffic = 0;
    state.hWeather = 0;
    state.mode = "terkunci";
  }

  window.addEventListener("cctv:select", (e) => {
    const item = e.detail || {};
    if ((item.name || "").trim().toUpperCase() === CAMERA_NAME) openPanel();
  });

  // TOL 01 dan TOL 02 berkoordinat sama di katalog sehingga markernya bertumpuk;
  // popup keduanya diberi tombol untuk membuka panel ramalan TOL 02.
  map.on("popupopen", (e) => {
    const node = e.popup && e.popup._contentNode;
    const title = node && node.querySelector(".popup-title");
    if (!title || !/TERUSAN BUAH BATU-JEMBATAN TOL 0[12]/i.test(title.textContent) || node.querySelector(".bbf-open-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "inline-action-btn outline block bbf-open-btn";
    btn.textContent = "Buka ramalan traffic & cuaca (TOL 02)";
    btn.addEventListener("click", (ev) => { ev.stopPropagation(); openPanel(); });
    node.appendChild(btn);
  });

  // untuk pengujian / tautan langsung: /?ramalan=buahbatu
  window.BuahBatuForecast = { open: openPanel, close: closePanel, state };
  if (new URLSearchParams(location.search).get("ramalan") === "buahbatu") {
    window.addEventListener("load", () => setTimeout(openPanel, 300));
  }
})();
