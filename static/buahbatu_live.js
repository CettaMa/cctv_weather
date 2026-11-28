/*
 * Siaran langsung CCTV Terusan Buah Batu - Jembatan Tol 02 pada halaman ramalan.
 * Berdiri sendiri: video tetap jalan walau data ramalan gagal dimuat.
 * Stream HLS lewat proxy aplikasi; tersambung ulang otomatis bila putus atau macet.
 */
(function () {
  "use strict";

  const video = document.getElementById("bbfVideo");
  const status = document.getElementById("bbfVideoStatus");
  const message = document.getElementById("bbfVideoMessage");
  const retry = document.getElementById("bbfVideoRetry");
  const url = video && video.dataset.stream;
  if (!video) return;

  const RETRY_MS = 10000;     // jeda sebelum mencoba lagi setelah gagal
  const STALL_MS = 45000;     // segmen kamera sekitar 14 detik; selama ini tanpa kemajuan dianggap macet
  const CONFIG = {
    lowLatencyMode: false,
    manifestLoadingTimeOut: 20000, manifestLoadingMaxRetry: 2,
    levelLoadingTimeOut: 20000, levelLoadingMaxRetry: 4,
    fragLoadingTimeOut: 40000, fragLoadingMaxRetry: 4,
    maxBufferLength: 30, backBufferLength: 30,
  };

  let hls = null;
  let retryTimer = null;
  let lastTime = -1;
  let lastProgress = Date.now();
  let mediaRecovered = false;

  let shown = { key: "videoConnecting", reason: null };   // untuk diterjemahkan ulang saat ganti bahasa

  function renderMessage() {
    message.textContent = I18n.t(shown.key, shown.reason ? { reason: I18n.t(shown.reason) } : null);
  }

  function show(key, canRetry, reason) {
    shown = { key, reason: reason || null };
    status.hidden = false;
    renderMessage();
    retry.hidden = !canRetry;
  }

  function stop() {
    clearTimeout(retryTimer);
    retryTimer = null;
    if (hls) { hls.destroy(); hls = null; }
  }

  function fail(reason) {
    stop();
    show("videoRetrying", true, reason);
    retryTimer = setTimeout(start, RETRY_MS);
  }

  function play() {
    video.play().catch(() => { /* autoplay ditolak: tombol putar bawaan video tetap tersedia */ });
  }

  function start() {
    stop();
    lastTime = -1;
    lastProgress = Date.now();
    mediaRecovered = false;
    show("videoConnecting", false);
    if (window.Hls && Hls.isSupported()) {
      hls = new Hls(CONFIG);
      hls.on(Hls.Events.MANIFEST_PARSED, play);
      hls.on(Hls.Events.ERROR, (event, data) => {
        if (!data.fatal) return;
        if (data.type === Hls.ErrorTypes.MEDIA_ERROR && !mediaRecovered) {
          mediaRecovered = true;
          hls.recoverMediaError();
          return;
        }
        fail(data.type === Hls.ErrorTypes.NETWORK_ERROR ? "videoNoResponse" : "videoFailed");
      });
      hls.loadSource(url);
      hls.attachMedia(video);
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {   // Safari tanpa MSE
      video.src = url;
      video.load();
      play();
    } else {
      show("videoUnsupported", false);
    }
  }

  I18n.onChange(renderMessage);
  renderMessage();
  if (!url) {
    show("videoMissing", false);
    return;
  }
  video.addEventListener("playing", () => { status.hidden = true; });
  video.addEventListener("error", () => { if (!hls) fail("videoFailed"); });
  retry.addEventListener("click", start);
  setInterval(() => {
    if (retryTimer || document.hidden || (video.paused && lastTime > 0)) { lastProgress = Date.now(); return; }
    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      lastProgress = Date.now();
    } else if (Date.now() - lastProgress > STALL_MS) {
      fail("videoStalled");
    }
  }, 5000);
  start();
})();
