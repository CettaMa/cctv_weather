/*
 * Halaman About: pengalih bahasa (Inggris, Indonesia, Jepang) dan tema.
 * Teks Inggris juga tertulis di templates/about.html sebagai tampilan awal. Setiap elemen
 * ber-atribut data-i18n harus punya kunci yang sama di ketiga bahasa di bawah.
 */
(function () {
  "use strict";

  const TEXT = {
    en: {
      title: "About · TENKOU-TK",
      back: "‹ Main map",
      heading: "About TENKOU-TK",
      theme: "Switch theme",
      projectHead: "The project",
      lead: "TENKOU-TK is a web map that brings together public CCTV streams, weather information and road intersection data to observe traffic and environmental conditions in Bandung, Indonesia.",
      mapTitle: "Live camera map",
      mapText: "Public CCTV streams across the city, with their online status and a multi-camera wall per road corridor.",
      weatherTitle: "Weather and flood watch",
      weatherText: "Rainfall, wind and river discharge layers, together with monitoring of flood-prone locations.",
      spatialTitle: "Spatial analysis",
      spatialText: "Road intersections, camera clusters, density heatmaps and terrain elevation.",
      forecastTitle: "Traffic and weather forecast",
      forecastText: "An experimental page for Terusan Buah Batu with live video. It replays recorded data from 20–26 November 2026 with forecasts up to six hours ahead.",
      partnersHead: "Participating institutions",
      tk: "Computer Engineering",
      dpri: "Disaster Prevention Research Institute, Kyoto University",
      brin: "National Research and Innovation Agency of Indonesia",
      atcs: "Area Traffic Control System",
      dataHead: "Data sources",
      cctvLabel: "CCTV streams: ",
      cctvAnd: " and ATCS cameras",
      weatherLabel: "Weather: ",
      mapsLabel: "Maps: ",
      and: " and ",
    },
    id: {
      title: "Tentang · TENKOU-TK",
      back: "‹ Peta utama",
      heading: "Tentang TENKOU-TK",
      theme: "Ganti tema",
      projectHead: "Proyek",
      lead: "TENKOU-TK adalah peta web yang menyatukan siaran CCTV publik, informasi cuaca, dan data persimpangan jalan untuk memantau kondisi lalu lintas dan lingkungan di Bandung, Indonesia.",
      mapTitle: "Peta kamera langsung",
      mapText: "Siaran CCTV publik di seluruh kota, lengkap dengan status online dan dinding multi-kamera per koridor jalan.",
      weatherTitle: "Pemantauan cuaca dan banjir",
      weatherText: "Lapisan curah hujan, angin, dan debit sungai, disertai pemantauan titik rawan banjir.",
      spatialTitle: "Analisis spasial",
      spatialText: "Persimpangan jalan, klaster kamera, peta kepadatan, dan elevasi medan.",
      forecastTitle: "Ramalan lalu lintas dan cuaca",
      forecastText: "Halaman eksperimen untuk Terusan Buah Batu dengan video langsung. Halaman ini memutar ulang data rekaman 20–26 November 2026 beserta ramalan hingga enam jam ke depan.",
      partnersHead: "Lembaga yang terlibat",
      tk: "Teknik Komputer",
      dpri: "Lembaga Riset Pencegahan Bencana, Universitas Kyoto",
      brin: "Badan Riset dan Inovasi Nasional",
      atcs: "Sistem Kendali Lalu Lintas Kawasan",
      dataHead: "Sumber data",
      cctvLabel: "Siaran CCTV: ",
      cctvAnd: " dan kamera ATCS",
      weatherLabel: "Cuaca: ",
      mapsLabel: "Peta: ",
      and: " dan ",
    },
    ja: {
      title: "概要 · TENKOU-TK",
      back: "‹ メインマップ",
      heading: "TENKOU-TKについて",
      theme: "テーマ切替",
      projectHead: "プロジェクト概要",
      lead: "TENKOU-TKは、公開CCTV映像、気象情報、道路交差点データを統合し、インドネシア・バンドン市の交通状況と環境を観測するためのウェブ地図です。",
      mapTitle: "ライブカメラマップ",
      mapText: "市内各地の公開CCTV映像を、稼働状況や道路区間ごとのマルチカメラ表示とともに確認できます。",
      weatherTitle: "気象・洪水モニタリング",
      weatherText: "降雨、風、河川流量のレイヤーと、浸水しやすい地点の監視機能を備えています。",
      spatialTitle: "空間分析",
      spatialText: "道路交差点、カメラのクラスター、密度ヒートマップ、地形の標高を表示します。",
      forecastTitle: "交通・気象予測",
      forecastText: "トゥルサン・ブア・バトゥ通りを対象とした実験的なページで、ライブ映像も表示します。2026年11月20日から26日の記録データを再生し、6時間先までの予測を示します。",
      partnersHead: "参加機関",
      tk: "コンピュータ工学",
      dpri: "京都大学防災研究所",
      brin: "インドネシア国家研究イノベーション庁",
      atcs: "エリア交通管制システム",
      dataHead: "データの出典",
      cctvLabel: "CCTV映像：",
      cctvAnd: " および ATCSカメラ",
      weatherLabel: "気象：",
      mapsLabel: "地図：",
      and: "、",
    },
  };

  const DEFAULT_LANG = "en";
  const read = (key) => { try { return localStorage.getItem(key); } catch (e) { return null; } };
  const write = (key, value) => { try { localStorage.setItem(key, value); } catch (e) { /* penyimpanan diblokir */ } };

  function setLanguage(lang) {
    const text = TEXT[lang];
    document.documentElement.lang = lang;
    document.title = text.title;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = text[el.dataset.i18n]; });
    document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  }

  document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => {
    write("cctv-lang", button.dataset.lang);
    setLanguage(button.dataset.lang);
  }));

  document.getElementById("aboutTheme").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    write("cctv-theme", next);
  });

  // urutan: ?lang= pada tautan, pilihan tersimpan, lalu Inggris
  const wanted = [new URLSearchParams(location.search).get("lang"), read("cctv-lang")]
    .find((lang) => Object.prototype.hasOwnProperty.call(TEXT, lang || ""));
  setLanguage(wanted || DEFAULT_LANG);

  window.AboutPage = { TEXT };   // untuk pengujian
})();
