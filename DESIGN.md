---
name: Bandung CCTV & Weather Platform
description: Real-time GIS dashboard for public surveillance streams and spatial weather analytics
colors:
  primary: "#3b82f6"
  primary-hover: "#2563eb"
  status-active: "#1f8f4d"
  status-missing: "#c74343"
  status-coverage: "#10b981"
  metric-gold: "#f1c40f"
  metric-border: "#b45309"
  chart-purple: "#8b5cf6"
  neutral-bg: "#f8fafc"
  neutral-surface: "#ffffff"
  neutral-border: "#cbd5e1"
  neutral-border-subtle: "#e2e8f0"
  neutral-border-strong: "#94a3b8"
  text-headline: "#0f172a"
  text-body: "#1e293b"
  text-secondary: "#475569"
  text-muted: "#64748b"
  black: "#000000"
typography:
  large:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    lineHeight: 1.2
  display:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1.25
  headline:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "14px"
    fontWeight: 700
    lineHeight: 1.3
  title:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.4
  body:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
  small:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1.4
  label:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    letterSpacing: "0.05em"
  micro:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "10px"
    fontWeight: 600
    lineHeight: 1.2
  caption:
    fontFamily: "'Public Sans', system-ui, -apple-system, sans-serif"
    fontSize: "9px"
    fontWeight: 600
    letterSpacing: "0.03em"
rounded:
  sm: "4px"
  chip: "6px"
  md: "8px"
  lg: "12px"
  full: "50%"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "14px"
  xl: "16px"
components:
  panel-hud:
    backgroundColor: "rgba(255, 255, 255, 0.95)"
    rounded: "{rounded.lg}"
    padding: "12px"
  control-input:
    backgroundColor: "{colors.neutral-surface}"
    textColor: "{colors.text-body}"
    rounded: "{rounded.md}"
    padding: "8px 10px"
  button-toggle:
    backgroundColor: "{colors.neutral-surface}"
    textColor: "{colors.text-headline}"
    rounded: "{rounded.md}"
    padding: "8px 14px"
  stat-card:
    backgroundColor: "{colors.neutral-bg}"
    rounded: "{rounded.md}"
    padding: "8px"
---

# Design System: Bandung CCTV & Weather Platform

## Overview

**Creative North Star: "The Urban Telemetry Glasshouse"**

The visual language of the Bandung CCTV & Weather Platform is a precise, utilitarian civic-telemetry dashboard. It treats the dynamic OpenStreetMap and CARTO base canvas as the living ground truth, floating high-density frosted-glass control HUDs and analytical telemetry panels directly above the map viewport. The aesthetic balances civic reliability, rapid spatial comprehension, and data density without visual clutter.

Interactions are immediate and responsive: live streaming previews load on-demand directly inside lightweight map popups, spatial statistical cards recalculate dynamically on viewport and filter adjustments, and high-contrast color pins immediately broadcast infrastructure health across the city.

**Key Characteristics:**
- **Floating Frosted HUDs:** Semi-translucent glassmorphic control containers (`rgba(255, 255, 255, 0.95)` with 8px backdrop blur) providing depth while keeping map context visible.
- **High-Contrast Telemetry Signals:** Unambiguous status indicators (emerald green for verified live streams, crimson red for unreachable/missing feeds).
- **Dense, Compact Hierarchy:** Strict information density optimized for operational monitoring, with compact 8px radius form controls and clean uppercase metric badges.

## Colors

The palette employs a cool slate neutral foundation punctuated by vivid functional signals for stream health, weather layers, and spatial dispersion.

### Primary
- **Telemetry Blue** (`#3b82f6`): Primary active focus rings, dispersion boundary outlines, and interactive map accents.
- **Deep Electric Blue** (`#2563eb`): Popup hyperlink text and high-intent hover states.

### Secondary
- **Operational Emerald** (`#1f8f4d`): Verified active camera markers, downloaded status badge, and healthy stream signal.
- **Outage Crimson** (`#c74343`): Missing / unreachable camera markers, error states, and high-density hexbin heat indicators.
- **Coverage Green** (`#10b981`): Simulated surveillance coverage rings (50m, 100m, 150m buffers).

### Tertiary
- **Mean Center Gold** (`#f1c40f`): Geometric mean center star icon and focal coordinate highlight.
- **Analytics Violet** (`#8b5cf6`): Area distribution bar charts.

### Neutral
- **Slate Canvas** (`#f8fafc`): Page canvas background, panel header backgrounds, and metric stat card fill.
- **Pure Surface** (`#ffffff`): Input backgrounds, popup bubbles, and control pill buttons.
- **Slate Border Subtle** (`#e2e8f0`): Panel headers, section dividers, and stat card outlines.
- **Slate Border Control** (`#cbd5e1`): Input borders, select dropdowns, and inactive button borders.
- **Deep Slate Text** (`#0f172a`): High-contrast titles, header typography, and primary numerical readouts.
- **Body Slate** (`#1e293b`): Primary reading text and dropdown value text.
- **Secondary Slate** (`#475569`): Section subtitles, legend labels, and uppercase category headers.
- **Muted Slate** (`#64748b`): Popup metadata labels, placeholder text, and secondary stats keys.

### Named Rules
**The Telemetry Contrast Rule.** Semantic status colors (green `#1f8f4d`, red `#c74343`) are reserved strictly for operational stream health and density alerts. They are never used decoratively on standard navigation or generic buttons.

## Typography

**Display & Body Font:** Inter (`'Inter', system-ui, -apple-system, sans-serif`)
**Monospace / Numerical Readouts:** Inter with tabular numerical styling (`precision: 0`)

**Character:** Utilitarian, sharp, and highly legible across compact telemetry cards and micro coordinate displays.

### Hierarchy
- **Display / Panel Header** (Weight: 700, Size: 15px, Line-Height: 1.25): Main application title on the control HUD.
- **Headline / Section Header** (Weight: 700, Size: 14px, Line-Height: 1.3): Analytics drawer header and popup title.
- **Title / Form Label** (Weight: 600, Size: 13px, Line-Height: 1.4): Input labels, filter options, and button text.
- **Body** (Weight: 400, Size: 13px, Line-Height: 1.4): Standard body copy, dropdown options, and popup rows.
- **Label / Micro Header** (Weight: 700, Size: 11px, Letter-Spacing: 0.05em, Uppercase): Category headings (`ANALISIS DISTRIBUSI SPASIAL`, `STATUS KEAKTIFAN CCTV`).
- **Telemetry Key** (Weight: 600, Size: 9px, Uppercase): Stat card titles (`PUSAT RATA-RATA`, `RADIUS DISPERSI`, `CAKUPAN AREA`).

### Named Rules
**The Micro-Clarity Rule.** Uppercase micro-labels (9px–11px) must always use 600–700 font weight and `0.05em` letter-spacing to ensure instant scannability during map navigation.

## Layout

The interface is structured as a full-viewport spatial map ($100\text{vw} \times 100\text{vh}$) overlaid with floating glass HUD panels:
- **Left HUD (Controls & Filtering):** Positioned at `top: 12px`, `left: 12px`, with max width `min(340px, calc(100vw - 24px))`. Contains camera search, multi-attribute dropdowns, weather radar toggle, and density mode selectors.
- **Right HUD (Telemetry Drawer):** Positioned at `top: 12px`, `right: 12px`, with max width `min(360px, calc(100vw - 24px))`. Features a collapsible slide animation (`translateX(400px)`), containing spatial metric grids and Chart.js distribution breakdowns.
- **Map Bottom Left Telemetry:** Non-intrusive live coordinate readout box tracking cursor hover position (`Lat: -6.91750 | Lng: 107.61910`).

## Elevation & Depth

Depth is established through translucent frosted-glass layering and tailored directional drop shadows that visually elevate HUD panels above dynamic base map tiles.

### Shadow Vocabulary
- **HUD Panel Float** (`box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)`): Floating control and analytics panels.
- **Map Popup Depth** (`box-shadow: 0 3px 14px rgba(0, 0, 0, 0.25)`): Active camera popup cards.
- **Button Float** (`box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12)`): Floating stats toggle button.
- **Marker Pin Shadow** (`box-shadow: 0 1px 7px rgba(0, 0, 0, 0.35)`): Circular map pins to ensure pin visibility over diverse map imagery.
- **Focal Glow** (`box-shadow: 0 0 12px rgba(241, 196, 15, 0.9)`): Mean center focal point.

### Named Rules
**The Frosted Glass Rule.** Overlay panels must maintain `background: rgba(255, 255, 255, 0.95)` paired with `backdrop-filter: blur(8px)` and a subtle `1px solid #e2e8f0` stroke to preserve legibility against moving map tiles.

## Shapes

- **HUD Panels:** Smooth curved corners (`border-radius: 12px`).
- **Interactive Controls & Inputs:** Ergonomic 8px corners (`border-radius: 8px`).
- **Telemetry Stat Cards & Video Players:** Compact 4px–8px corners (`border-radius: 4px` for video embed, `8px` for stat tiles).
- **Marker Pins & Status Dots:** Pure geometric circles (`border-radius: 50%`) with crisp 2px solid white halos.

## Components

### Floating HUD Panel
- **Shape:** `12px` border radius, `1px solid #e2e8f0` border.
- **Background:** `rgba(255, 255, 255, 0.95)` with `backdrop-filter: blur(8px)`.
- **Header:** Distinct `14px 16px` padding with `#f8fafc` background and bottom divider `#e2e8f0`.

### Form Controls (Search & Selects)
- **Shape:** `8px` border radius, `8px 10px` internal padding.
- **Border:** `1px solid #cbd5e1`, transitioning to `#3b82f6` on focus.
- **Background:** `#ffffff`, with `13px` font size.

### Telemetry Stat Card
- **Shape:** `8px` border radius, `8px` padding.
- **Background:** `#f8fafc` with `1px solid #e2e8f0` perimeter stroke.
- **Typography:** 9px uppercase label over 11px bold value.

### Camera Map Pin
- **Shape:** 14px circle (`border-radius: 50%`) with `2px solid #ffffff` border.
- **Fill:** `#1f8f4d` (downloaded/active) or `#c74343` (missing).
- **Shadow:** `0 1px 7px rgba(0, 0, 0, 0.35)`.

### Live HLS Video Popup Player
- **Container:** `background: #000000`, `border-radius: 4px`, `overflow: hidden`, max-height `180px`.
- **Controls:** Native HLS video playback with inline audio mute and autoplay.

## Do's and Don'ts

### Do:
- **Do** keep map overlay panels semi-translucent (`rgba(255, 255, 255, 0.95)` + `blur(8px)`) with crisp `1px solid #e2e8f0` border outlines.
- **Do** use 2px white halo borders on all circular map markers to ensure contrast against light and dark base maps.
- **Do** format all spatial coordinate readouts to 5 decimal places with tabular font alignments.
- **Do** maintain collapsible drawer transitions (`transform: translateX(...)` in 300ms) for mobile viewport responsiveness.

### Don't:
- **Don't** use saturated bright colors for generic UI cards; keep accents reserved for operational status and radar data.
- **Don't** remove the `2px solid #ffffff` border from map marker pins, which prevents pins from bleeding into basemap lines.
- **Don't** allow map popups to exceed 320px width on mobile viewports.
