/*
 * Halaman ramalan traffic + cuaca (mode replay) untuk CCTV TERUSAN BUAH BATU-JEMBATAN TOL 02
 * (/ramalan/buahbatu, templates/buahbatu.html).
 *
 * Data: /api/buahbatu/forecast/meta dan /api/buahbatu/forecast?asal=<ISO>, dibuat oleh
 * extractions/forecast/ekspor_app.py. Dua slider horizon (0..+360 menit, langkah 5):
 * - Terkunci: traffic dan cuaca selalu di horizon yang sama.
 * - Bebas: masing-masing slider punya horizon sendiri.
 * Dua minimap: kepadatan yang hanya menutupi badan Jalan Buah Batu (geometri OpenStreetMap, warna
 * menurut arus per arah pada rentang yang bisa diatur) dan lingkaran hujan 1 km di sekitar kamera.
 */
(function () {
  "use strict";

  const RAIN_RADIUS_M = 1000;
  const ROAD_URL = "/static/buahbatu_jalan.json";
  const RAIN_MAX = 5;            // mm/jam untuk cincin paling pekat
  const DIRS = ["menjauhi", "mendekati"];
  const DIR_KEY = { menjauhi: "away", mendekati: "towards" };
  const t = I18n.t;
  const dirLabel = (d) => t(DIR_KEY[d]);
  const DIR_COLOR = { menjauhi: "#2a78d6", mendekati: "#eb6834" };
  // warna heatmap dari batas bawah ke batas atas rentang: hijau, kuning, jingga, merah, merah tua
  const HEAT_COLORS = [[16, 185, 129], [250, 204, 21], [249, 115, 22], [220, 38, 38], [112, 16, 48]];
  const HEAT_STEP = 100;         // kend/jam, pembulatan rentang otomatis
  const HEAT_MIN_SPAN = 100;
  // kunci publik CARTO yang sama dengan peta utama
  const CARTO_KEY = "cb1_2ew3_1_dbbdad4a42a2c9c38df8da88";
  const TILE_STYLE = {
    dark: `https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json?key=${CARTO_KEY}`,
    light: `https://basemaps.cartocdn.com/gl/positron-gl-style/style.json?key=${CARTO_KEY}`,
  };
  const HEAT_GRADIENT = `linear-gradient(90deg,${HEAT_COLORS.map((c) => `rgb(${c.join(",")})`).join(",")})`;

  const state = {
    meta: null,
    data: null,
    road: null,
    mode: "terkunci",
    hTraffic: 0,
    hWeather: 0,
    heat: { auto: true, min: 0, max: 6000 },   // rentang warna heatmap, kend/jam
    maps: { road: null, rain: null },   // { map, tiles, layer }
    charts: { traffic: null, rain: null },
  };

  // ---------------------------------------------------------------- util
  const $ = (sel) => document.querySelector(sel);
  const fmtInt = (v) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Math.round(v).toLocaleString(I18n.locale));
  const fmt1 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "-" : v.toLocaleString(I18n.locale, { maximumFractionDigits: 1, minimumFractionDigits: 1 }));
  const cssVar = (name, fallback) => (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fallback;

  function timeAt(h) {
    const t0 = new Date(state.data.asal);
    return new Date(t0.getTime() + h * state.meta.langkah_menit * 60000);
  }

  function hhmm(d) {
    // waktu WIB (UTC+7) tanpa bergantung pada zona waktu browser
    const w = new Date(d.getTime() + 7 * 3600000);
    const sep = I18n.lang === "id" ? "." : ":";
    return `${String(w.getUTCHours()).padStart(2, "0")}${sep}${String(w.getUTCMinutes()).padStart(2, "0")}`;
  }

  function dayLabel(iso) {
    return new Date(iso).toLocaleDateString(I18n.locale, { weekday: "short", day: "numeric", month: "short", timeZone: "Asia/Jakarta" });
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
    if (dj >= state.meta.dj_macet) return { cls: "macet", text: t("jammed") };
    if (dj >= state.meta.dj_padat) return { cls: "padat", text: t("dense") };
    return { cls: "lancar", text: t("smooth") };
  }

  function djAt(dir, h) {
    const smp = trafficAt(dir, "smp", h);
    return smp === null ? null : smp / state.meta.kapasitas_smp_jam;
  }

  function rainClass(mm) {
    if (mm === null) return "-";
    if (mm < 0.1) return t("noRain");
    if (mm < 2.5) return t("lightRain");
    if (mm < 7.6) return t("moderateRain");
    return t("heavyRain");
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
    if (!state.data) return;
    ["min", "max"].forEach((k) => {
      const input = $(`#bbfHeat_${k}`);
      if (document.activeElement !== input) input.value = state.heat[k];
    });
    $("#bbfHeatAuto").setAttribute("aria-pressed", String(state.heat.auto));
    const { min, max } = state.heat;
    $("#bbfLegendTicks").innerHTML = [0, 0.5, 1].map((t) => `<span>${fmtInt(min + (max - min) * t)}</span>`).join("");
    $("#bbfLegendMarks").innerHTML = DIRS.map((d) => {
      const flow = trafficAt(d, "flow", state.hTraffic);
      if (flow === null) return "";
      const out = flow < min ? t("belowRange") : flow > max ? t("aboveRange") : "";
      return `<span class="bbf-mark" style="left:${(heatPos(flow) * 100).toFixed(1)}%" ` +
        `title="${dirLabel(d)}: ${fmtInt(flow)} ${t("vehUnit")}${out}">${d === "menjauhi" ? "▼" : "▽"}</span>`;
    }).join("");
  }

  function setHeatRange() {
    const min = parseFloat($("#bbfHeat_min").value);
    const max = parseFloat($("#bbfHeat_max").value);
    const valid = Number.isFinite(min) && Number.isFinite(max) && min >= 0 && max - min >= HEAT_MIN_SPAN;
    $("#bbfHeatError").textContent = valid ? "" : t("heatError", { n: HEAT_MIN_SPAN });
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

  const cameraPoint = () => L.latLng(state.meta.kamera.lat, state.meta.kamera.lon);
  const theme = () => (document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark");

  function makeTiles(map) {
    return L.maplibreGL({
      style: TILE_STYLE[theme()],
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map);
  }

  function makeMap(id) {
    // roda gulir dimatikan agar halaman tetap bisa digulir; zoom lewat tombol + / -
    const map = L.map(id, { scrollWheelZoom: false, zoomSnap: 0.25 }).setView(cameraPoint(), 16);
    return { map, tiles: makeTiles(map), layer: L.layerGroup().addTo(map) };
  }

  function addCamera(target) {
    L.circleMarker(cameraPoint(), {
      radius: 5, color: cssVar("--text-strong", "#fff"), weight: 1.5, fillOpacity: 1,
      fillColor: cssVar("--accent", "#38bdf8"),
    }).bindTooltip(t("camera")).addTo(target.layer);
  }

  function renderRain() {
    const target = state.maps.rain;
    target.layer.clearLayers();
    const rain = seriAt("hujan", state.hWeather);
    if (rain !== null) {
      const wet = rain >= 0.1;
      const disk = L.circle(cameraPoint(), {
        radius: RAIN_RADIUS_M, interactive: false,
        color: "#60a5fa", weight: 1.5, opacity: 0.8, dashArray: wet ? null : "6 6", fillOpacity: wet ? 1 : 0,
      }).addTo(target.layer);
      const svg = disk._path && disk._path.ownerSVGElement;
      if (svg && wet) {
        const a = 0.14 + 0.36 * Math.min(1, rain / RAIN_MAX);
        setGradient(svg, "bbf-rain-grad", [["0%", "#2563eb", a], ["75%", "#2563eb", a * 0.85], ["100%", "#60a5fa", a * 0.25]]);
        disk._path.setAttribute("fill", "url(#bbf-rain-grad)");
      }
    }
    addCamera(target);
  }

  function renderRoad() {
    const target = state.maps.road;
    target.layer.clearLayers();
    (state.road ? state.road.ruas : []).forEach((r) => {
      // ruas dua arah di simpang memakai arah yang lebih padat
      const flows = r.arah.map((d) => trafficAt(d, "flow", state.hTraffic)).filter((v) => v !== null);
      if (!flows.length) return;
      const pos = heatPos(Math.max(...flows));
      const color = `rgb(${heatColor(pos).join(",")})`;
      const heat = 0.75 + 0.25 * pos;
      const rows = r.arah.map((d) => `${dirLabel(d)}: ${fmtInt(trafficAt(d, "flow", state.hTraffic))} ${t("vehUnit")} · ${condition(djAt(d, state.hTraffic)).text}`);
      // tiga pita bertumpuk: tepi tipis, tengah pekat
      [[1, 0.45], [0.68, 0.4], [0.34, 0.5]].forEach(([part, alpha], i) => {
        const poly = L.polygon(roadPolygon(r.koordinat, r.lebar_m * part), {
          stroke: false, fillColor: color, fillOpacity: alpha * heat, interactive: i === 0,
        }).addTo(target.layer);
        if (i === 0) poly.bindTooltip(`+${state.hTraffic * state.meta.langkah_menit} ${t("minutes")}<br>${rows.join("<br>")}`, { sticky: true });
      });
    });
    addCamera(target);
  }

  function renderLayers() {
    if (!state.data) return;
    renderRoad();
    renderRain();
  }

  // pas-kan minimap ke badan jalan atau ke seluruh area cuaca
  function focusMap(name) {
    const { map } = state.maps[name];
    const bounds = name === "road" && state.road
      ? L.latLngBounds(state.road.ruas.flatMap((r) => r.koordinat))
      : cameraPoint().toBounds(RAIN_RADIUS_M * 2);
    map.invalidateSize();
    map.fitBounds(bounds, { padding: [16, 16], animate: false });
  }

  // ---------------------------------------------------------------- grafik
  function chartColors() {
    return { grid: cssVar("--chart-grid", "rgba(30,41,59,0.5)"), text: cssVar("--muted", "#64748b") };
  }

  function buildCharts() {
    const { grid, text } = chartColors();
    const H = state.meta.h_max;
    const labels = Array.from({ length: H + 1 }, (_, h) => hhmm(timeAt(h)));
    const common = {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: grid }, ticks: { color: text, maxTicksLimit: 7, font: { size: 10 } } },
        y: { grid: { color: grid }, ticks: { color: text, maxTicksLimit: 4, font: { size: 10 } }, beginAtZero: true },
      },
    };
    Object.values(state.charts).forEach((c) => c && c.destroy());

    const datasets = [];
    DIRS.forEach((d) => {
      datasets.push({
        label: t("seriesForecast", { dir: dirLabel(d) }), data: Array.from({ length: H + 1 }, (_, h) => trafficAt(d, "flow", h)),
        borderColor: DIR_COLOR[d], backgroundColor: DIR_COLOR[d], borderWidth: 2, pointRadius: 0, tension: 0.25, spanGaps: true,
      });
      datasets.push({
        label: t("seriesActual", { dir: dirLabel(d) }), data: Array.from({ length: H + 1 }, (_, h) => seriAt(`flow_${d}`, h)),
        borderColor: DIR_COLOR[d], backgroundColor: DIR_COLOR[d], showLine: false, pointRadius: 1.8, pointStyle: "circle",
      });
    });
    state.charts.traffic = new Chart($("#bbfTrafficChart"), {
      type: "line", data: { labels, datasets },
      options: {
        ...common,
        plugins: {
          legend: { display: true, position: "bottom", labels: { color: text, boxWidth: 8, boxHeight: 8, font: { size: 10 }, filter: (i) => i.datasetIndex % 2 === 0 } },   // hanya garis ramalan
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtInt(c.parsed.y)} ${t("vehUnit")}` } },
        },
      },
      plugins: [markerPlugin(() => state.hTraffic)],
    });

    state.charts.rain = new Chart($("#bbfRainChart"), {
      type: "bar",
      data: { labels, datasets: [{ label: t("rainSeries"), data: Array.from({ length: H + 1 }, (_, h) => seriAt("hujan", h)), backgroundColor: "#2563eb", borderRadius: 1, barPercentage: 1, categoryPercentage: 0.9 }] },
      options: {
        ...common,
        scales: { ...common.scales, y: { ...common.scales.y, suggestedMax: 1 } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${t("rain")}: ${fmt1(c.parsed.y)} ${t("rainUnit")}` } } },
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

  // ---------------------------------------------------------------- halaman
  function originOptions() {
    const byDay = {};
    state.meta.asal.forEach((o) => { (byDay[dayLabel(o.t)] = byDay[dayLabel(o.t)] || []).push(o); });
    return Object.entries(byDay).map(([day, list]) =>
      `<optgroup label="${day}">${list.map((o) => `<option value="${o.t}">${day} ${hhmm(new Date(o.t))}</option>`).join("")}</optgroup>`
    ).join("");
  }

  function cardsTraffic(h) {
    const parts = DIRS.map((d) => {
      const pred = trafficAt(d, "flow", h);
      const act = h > 0 ? seriAt(`flow_${d}`, h) : null;
      const dj = djAt(d, h);
      const c = condition(dj);
      let sub = t("observed");
      if (h > 0 && act !== null && pred !== null) {
        const diff = Math.round(pred - act);
        sub = t("actualDiff", { actual: fmtInt(act), diff: `${diff > 0 ? "+" : ""}${fmtInt(diff)}` });
      } else if (h > 0) {
        sub = t(seriAt("ulang", h) ? "replayed" : "notRecorded");
      }
      return `<div class="bbf-card"><div class="k">${t(d === "menjauhi" ? "flowAway" : "flowTowards")}</div>
        <div class="v">${fmtInt(pred)}<span class="u">${t("vehUnit")}</span></div>
        <div class="s">${sub}</div>
        <div class="bbf-cond ${c.cls}" title="${t("djTitle")}">${c.text} · ${t("dj")} ${dj === null ? "-" : dj.toFixed(2)}</div></div>`;
    });
    const smp = DIRS.map((d) => trafficAt(d, "smp", h));
    const mae = h > 0 ? state.meta.mae_flow_per_h[h - 1] : null;
    parts.push(`<div class="bbf-card"><div class="k">${t("pcuTotal")}</div><div class="v">${fmtInt(smp.every((v) => v !== null) ? smp[0] + smp[1] : null)}<span class="u">${t("pcuUnit")}</span></div><div class="s">${t("bothDirections")}</div></div>`);
    parts.push(`<div class="bbf-card"><div class="k">${t("typicalError")}</div><div class="v">${h === 0 ? "-" : "±" + fmtInt(mae)}<span class="u">${t("vehUnit")}</span></div><div class="s">${t("maeNote")}</div></div>`);
    return parts.join("");
  }

  function cardsWeather(h) {
    const rain = seriAt("hujan", h);
    return [
      `<div class="bbf-card"><div class="k">${t("rain")}</div><div class="v">${fmt1(rain)}<span class="u">${t("rainUnit")}</span></div><div class="s">${rainClass(rain)}</div></div>`,
      `<div class="bbf-card"><div class="k">${t("temperature")}</div><div class="v">${fmt1(seriAt("suhu", h))}<span class="u">°C</span></div><div class="s">${t("humidity", { n: fmtInt(seriAt("rh", h)) })}</div></div>`,
      `<div class="bbf-card"><div class="k">${t("cloud")}</div><div class="v">${fmtInt(seriAt("awan", h))}<span class="u">%</span></div></div>`,
      `<div class="bbf-card"><div class="k">${t("wind")}</div><div class="v">${fmt1(seriAt("angin", h))}<span class="u">${t("speedUnit")}</span></div></div>`,
    ].join("");
  }

  function sliderText(h) {
    return `+${h * state.meta.langkah_menit} ${t("minutes")} · ${hhmm(timeAt(h))} WIB`;
  }

  function updateView() {
    if (!state.data) return;
    [["#bbfTraffic", state.hTraffic], ["#bbfWeather", state.hWeather]].forEach(([id, h]) => {
      $(id).value = h;
      $(`${id}Val`).textContent = sliderText(h);
      $(id).setAttribute("aria-valuetext", sliderText(h));
    });
    $("#bbfTrafficHead").textContent = state.hTraffic === 0 ? t("headNow") : t("headTraffic", { time: sliderText(state.hTraffic) });
    $("#bbfWeatherHead").textContent = t("headWeather", { time: sliderText(state.hWeather) });
    $("#bbfTrafficCards").innerHTML = cardsTraffic(state.hTraffic);
    $("#bbfWeatherCards").innerHTML = cardsWeather(state.hWeather);
    Object.values(state.charts).forEach((c) => c && c.update("none"));
    renderHeatControls();
    renderLayers();
  }

  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll(".bbf-seg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    const hint = $("#bbfModeHint");
    hint.setAttribute("data-i18n", mode === "terkunci" ? "hintLocked" : "hintFree");
    hint.textContent = t(hint.getAttribute("data-i18n"));
    if (mode === "terkunci") state.hWeather = state.hTraffic;
    updateView();
  }

  function fail(key, err) {
    console.error(key, err);
    state.failKey = key;
    $("#bbfFail").textContent = t(key);
    $("#bbfFail").hidden = false;
  }

  async function getJson(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.reason || res.status);
    return data;
  }

  // "latih s.d. 22 Nov" dari server
  function splitLabel(text) {
    const match = /^latih s\.d\. (\d+) Nov$/.exec(text || "");
    if (!match) return text;
    const date = new Date(Date.UTC(2026, 10, Number(match[1])));
    return t("trainedUntil", { date: date.toLocaleDateString(I18n.locale, { day: "numeric", month: "short", timeZone: "UTC" }) });
  }

  function renderNote() {
    const { meta, data } = state;
    $("#bbfNote").textContent = t("note", {
      model: I18n.v(meta.model), split: splitLabel(data.model), source: I18n.v(meta.cuaca_sumber),
      remark: I18n.v(meta.catatan), point: `${meta.kamera.lat.toFixed(5)}, ${meta.kamera.lon.toFixed(5)}`,
    });
  }

  function renderRainTicks() {
    $("#bbfRainTicks").innerHTML = [(0.1).toLocaleString(I18n.locale), (2.5).toLocaleString(I18n.locale), `≥ ${RAIN_MAX} ${t("rainUnit")}`]
      .map((s) => `<span>${s}</span>`).join("");
  }

  function setRoadCredit() {
    const control = state.maps.road && state.maps.road.map.attributionControl;
    if (!control || !state.road) return;
    if (state.roadCredit) control.removeAttribution(state.roadCredit);
    state.roadCredit = t("roadCredit");
    control.addAttribution(state.roadCredit);
  }

  // teks yang dibuat skrip ini; teks statis sudah diganti oleh I18n
  function relocalize() {
    renderRainTicks();
    $("#bbfHeatError").textContent = "";
    if (!$("#bbfFail").hidden) $("#bbfFail").textContent = t(state.failKey);
    setRoadCredit();
    if (!state.meta) return;
    const origin = $("#bbfOrigin");
    const selected = origin.value;
    origin.innerHTML = originOptions();
    origin.value = selected;
    if (!state.data) return;
    renderNote();
    buildCharts();
    updateView();
  }

  async function loadOrigin(iso) {
    try {
      state.data = await getJson(`/api/buahbatu/forecast?asal=${encodeURIComponent(iso)}`);
    } catch (err) {
      fail("failLoad", err);
      return;
    }
    $("#bbfFail").hidden = true;
    if (state.heat.auto) autoRange();
    $("#bbfOrigin").value = state.data.asal;
    renderNote();
    buildCharts();
    updateView();
  }

  function wire() {
    $("#bbfOrigin").addEventListener("change", (e) => loadOrigin(e.target.value));
    document.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => {
      const sel = $("#bbfOrigin");
      const i = Math.max(0, Math.min(sel.options.length - 1, sel.selectedIndex + parseInt(b.dataset.step, 10)));
      if (i !== sel.selectedIndex) loadOrigin(sel.options[i].value);
    }));
    document.querySelectorAll(".bbf-seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    $("#bbfTraffic").addEventListener("input", (e) => {
      state.hTraffic = parseInt(e.target.value, 10) || 0;
      if (state.mode === "terkunci") state.hWeather = state.hTraffic;
      updateView();
    });
    $("#bbfWeather").addEventListener("input", (e) => {
      state.hWeather = parseInt(e.target.value, 10) || 0;
      if (state.mode === "terkunci") state.hTraffic = state.hWeather;
      updateView();
    });
    ["min", "max"].forEach((k) => {
      const input = $(`#bbfHeat_${k}`);
      input.addEventListener("input", setHeatRange);
      input.addEventListener("blur", () => { $("#bbfHeatError").textContent = ""; renderHeatControls(); });
    });
    $("#bbfHeatAuto").addEventListener("click", () => {
      state.heat.auto = true;
      autoRange();
      $("#bbfHeatError").textContent = "";
      renderHeatControls();
      renderLayers();
    });
    document.querySelectorAll("[data-focus]").forEach((b) => b.addEventListener("click", () => focusMap(b.dataset.focus)));
  }

  function wireTheme() {
    $("#bbfTheme").addEventListener("click", () => {
      const next = theme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("cctv-theme", next); } catch (e) { /* penyimpanan diblokir */ }
      Object.values(state.maps).forEach((m) => {
        if (!m) return;
        m.map.removeLayer(m.tiles);
        m.tiles = makeTiles(m.map);
      });
      if (state.data) { buildCharts(); updateView(); }
    });
  }

  async function start() {
    wireTheme();
    renderRainTicks();
    I18n.onChange(relocalize);
    try {
      state.meta = await getJson("/api/buahbatu/forecast/meta");
    } catch (err) {
      fail("failMeta", err);
      return;
    }
    $("#bbfOrigin").innerHTML = originOptions();
    $("#bbfTraffic").max = $("#bbfWeather").max = state.meta.h_max;
    $("#bbfLegendBar").style.background = HEAT_GRADIENT;
    state.maps = { road: makeMap("bbfRoadMap"), rain: makeMap("bbfRainMap") };
    try {
      state.road = await getJson(ROAD_URL);
      setRoadCredit();
    } catch (err) {
      fail("failRoad", err);
    }
    wire();
    focusMap("road");
    focusMap("rain");
    await loadOrigin(state.meta.asal_default || state.meta.asal[state.meta.asal.length - 1].t);
  }

  // untuk pengujian
  window.BuahBatuForecast = { state };
  start();
})();
