# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Flask (Python 3.11), Gunicorn, HTML5 / CSS3 / Vanilla JS, Leaflet.js, Turf.js, Chart.js, Hls.js, OpenWeatherMap API, OpenStreetMap Overpass API, Docker.

## Users

Bandung citizens, commuters, daily motorists, and urban observers who need real-time situational awareness of road traffic, intersection congestion, weather / precipitation radar, and public CCTV feed availability.

## Product Purpose

Provide an accessible, performant, and intuitive real-time GIS dashboard that unifies live streaming public CCTV surveillance with meteorological radar and road intersection spatial clustering across Bandung City.

## Positioning

A seamless, real-time spatial intelligence interface that integrates Bandung municipal CCTV live streaming feeds (HLS / m3u8) with OpenWeatherMap weather radar layers, OpenStreetMap topological intersection clustering, and automated stream health tracking in a single lightweight web application.

## Operating Context

- Web browsers on desktop, tablet, and mobile devices.
- Live traffic checks before and during commuting hours across Bandung's road network.
- Adverse weather situations (heavy rainfall, flood-prone areas, storm tracking) where visual confirmation from cameras and precipitation radar layers are critical.
- Urban infrastructure inspection, spatial analytics, and stream connectivity health monitoring.

## Capabilities and Constraints

- **Capabilities:**
  - Real-time in-map HLS video streaming preview for 375+ Bandung CCTV cameras.
  - Interactive precipitation radar overlay via OpenWeatherMap tile API.
  - Spatial analytics: Mean center calculation, dispersion radius, bounding area coverage, and camera density metrics.
  - Density visualization via heatmaps (`leaflet-heat`) and hexagonal binning (1 km² / 2 km² grids).
  - Coverage buffer simulations (50m, 100m, 150m radius).
  - Background asynchronous camera stream ping worker (`/api/status`) verifying live stream availability.
  - Road intersection clustering and nearest junction calculation using OpenStreetMap road network graphs.
  - Dynamic multi-parameter filtering (by camera name, provider, area, and download/active status).
  - Summary analytics charts (Active status breakdown, Top 5 providers, Top 5 camera areas).
  - Stream latency and video segment throughput benchmarking scripts.
- **Constraints:**
  - Stream availability relies on uptime and CORS/TLS accessibility of external public surveillance endpoints (`pelindung.bandung.go.id`).
  - OpenWeatherMap radar requires a valid API key (provided via environment variable `OWM_API_KEY` or frontend input).

## Brand Commitments

- Project Name: Bandung CCTV & Weather Platform (*Pelindung* CCTV Map).
- Tone & Voice: Clear, pragmatic, informative, high-utility, civic-tech oriented.
- Clean, responsive UI with accessible map controls and high-contrast status indicators (green for active/downloaded, red for missing/unreachable).

## Evidence on Hand

- [`csv/cctv_bandung_data_validated.csv`](file:///C:/Users/Ichi/Documents/pelindung/csv/cctv_bandung_data_validated.csv): Master dataset containing ~375 camera records with GPS coordinates, providers, stream URLs, and verification history.
- [`csv/cctv_perempatan_clusters.csv`](file:///C:/Users/Ichi/Documents/pelindung/csv/cctv_perempatan_clusters.csv) & [`csv/cctv_di_sekitar_perempatan_100m.csv`](file:///C:/Users/Ichi/Documents/pelindung/csv/cctv_di_sekitar_perempatan_100m.csv): Intersection clustering datasets mapped from OpenStreetMap.
- [`cctv_video_samples/`](file:///C:/Users/Ichi/Documents/pelindung/cctv_video_samples): Sample video captures (.mp4) verifying stream functionality.
- Scraping, clustering, and benchmarking scripts in [`script_tools/`](file:///C:/Users/Ichi/Documents/pelindung/script_tools).

## Product Principles

1. **Real-Time Utility First:** Provide instant visual and spatial confirmation of on-the-ground road and weather conditions without UI clutter or unnecessary loading overhead.
2. **Robust Spatial Transparency:** Clearly differentiate verified active streams from unreachable ones with accurate geographical placement and telemetry data.
3. **Responsive and Lightweight GIS:** Deliver smooth interactive mapping, heatmaps, and live video streaming that works reliably across mobile and desktop browsers with minimal resource consumption.
4. **Data Integrity & Open Architecture:** Maintain clean data structures, repeatable scrapers, and open standard GIS/video protocols (HLS, GeoJSON/OSM, CSV).

## Accessibility & Inclusion

- High visual contrast for map markers, indicators, chart elements, and overlay panels.
- Responsive layout adapting to small screens with collapsible stats drawers and touch-friendly controls.
- Keyboard-navigable inputs and clear semantic labels for filter controls.
