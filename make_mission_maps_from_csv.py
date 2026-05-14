#!/usr/bin/env python3
"""
Generate mission_map.png images directly from recorded mission_travel_path.csv
files, without reading the (multi-GB) ROS bag files.

This is a fast complement to extract_georeferenced_images.py. It walks a root
directory containing one folder per mission (folder name format:
``YYYYMMDDHHMMSS-<MissionName>``) and produces one PNG per mission in the
chosen output directory.

Each mission folder is expected to contain:
  - mission_travel_path.csv  (recorded pose log)
  - mission_summary.json     (dataset_id, mission_name, start times)
  - <MissionName>.json       (optional: planned waypoints overlay)

The plotting style matches ``create_mission_map`` in
``extract_georeferenced_images.py``: actual path scatter coloured by speed,
optional planned-path overlay, and a right-hand statistics panel.

Output filenames are ``<DDMonYYYY>_<MissionName>_missionmap_<N>.png`` where N
restarts at 1 for each distinct ``<MissionName>`` and is assigned in
chronological order. The on-figure label (title + stats panel) is the shorter
form ``<MissionName> <N>``.

Usage:
    python make_mission_maps_from_csv.py --root /path/to/missions --out /path/to/mission_maps
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_map(mission_dir: Path, out_path: Path, label: str) -> None:
    df = pd.read_csv(mission_dir / "mission_travel_path.csv")
    with open(mission_dir / "mission_summary.json") as f:
        summary = json.load(f)
    mission_name = summary.get("mission_name", mission_dir.name)
    dataset_id = summary.get("dataset_id", mission_dir.name)

    wp_json_path = mission_dir / f"{mission_name}.json"
    mission_json = None
    if wp_json_path.exists():
        with open(wp_json_path) as f:
            mission_json = json.load(f)

    lon = df["longitude"].values
    lat = df["latitude"].values
    ts = df["timestamp_ros"].values.astype(float)

    dt = np.diff(ts)
    cos_lat = np.cos(np.radians(np.mean(lat)))
    dx = np.diff(lon) * 111000 * cos_lat
    dy = np.diff(lat) * 111000
    dist = np.sqrt(dx**2 + dy**2)
    speed = np.where(dt > 0, dist / dt, 0.0)
    speed = np.append(speed, speed[-1] if len(speed) else 0.0)
    speed = pd.Series(speed).rolling(window=15, center=True, min_periods=1).median().values

    start_ts = ts.min()
    end_ts = ts.max()
    duration = end_ts - start_ts
    alt_series = df["altitudeUsed"]
    stats = {
        "start_time": datetime.fromtimestamp(start_ts).strftime("%d %B %Y %H:%M:%S"),
        "end_time": datetime.fromtimestamp(end_ts).strftime("%d %B %Y %H:%M:%S"),
        "duration": duration,
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_span_m": (lon.max() - lon.min()) * 111000 * cos_lat,
        "lat_span_m": (lat.max() - lat.min()) * 111000,
        "alt_min": float(alt_series.min()),
        "alt_max": float(alt_series.max()),
        "alt_mean": float(alt_series.mean()),
        "samples": len(df),
        "depth_min": float(df["depth"].min()),
        "depth_max": float(df["depth"].max()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [3, 1]})

    ax1 = axes[0]
    scatter = ax1.scatter(lon, lat, c=speed, cmap="plasma", s=1, alpha=0.7, vmin=0, vmax=1.0)
    ax1.plot(lon[0], lat[0], "go", markersize=10, label="Start", zorder=5)
    ax1.plot(lon[-1], lat[-1], "ro", markersize=10, label="End", zorder=5)

    set_alt_str = "  Set Altitude: N/A"
    if mission_json is not None:
        waypoints = mission_json.get("waypoints", [])
        if waypoints:
            wp_lons = [wp["longitude"] for wp in waypoints]
            wp_lats = [wp["latitude"] for wp in waypoints]
            ax1.plot(wp_lons, wp_lats, "--", color="white", linewidth=3.0, zorder=3)
            ax1.plot(wp_lons, wp_lats, "--", color="limegreen", linewidth=1.8, zorder=3,
                     label="Planned path")
            for wp in waypoints:
                ax1.plot(wp["longitude"], wp["latitude"], "D", color="white", markersize=8,
                         markeredgecolor="limegreen", markeredgewidth=1.5, zorder=4)
                ax1.annotate(
                    f'WP{wp["waypoint_number"]}\n{wp["speed"]:.2f} m/s',
                    (wp["longitude"], wp["latitude"]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="limegreen", alpha=0.8),
                    zorder=7,
                )
            for wp in waypoints:
                alt = wp.get("additional_data", {}).get("Altitude")
                if alt is not None:
                    set_alt_str = f"  Set Altitude: {alt:.1f}m"
                    break

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title(f"Mission Path: {label}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    ax1.ticklabel_format(useOffset=False, style="plain")
    plt.colorbar(scatter, ax=ax1, label="Speed (m/s)")

    ax2 = axes[1]
    ax2.axis("off")
    duration_min = int(stats["duration"] // 60)
    duration_sec = stats["duration"] % 60
    stats_text = f"""
MISSION STATISTICS
{'='*40}

Dataset: {dataset_id}
Mission: {label}

TIME
  Start: {stats['start_time']}
  End:   {stats['end_time']}
  Duration: {duration_min}m {duration_sec:.1f}s

LOCATION (WGS84)
  Longitude: {stats['lon_min']:.6f}° to {stats['lon_max']:.6f}°
  Latitude:  {stats['lat_min']:.6f}° to {stats['lat_max']:.6f}°
  Span: {stats['lon_span_m']:.1f}m x {stats['lat_span_m']:.1f}m

DEPTH / ALTITUDE
  Depth:        {stats['depth_min']:.2f}m to {stats['depth_max']:.2f}m
  DVL Altitude: {stats['alt_min']:.2f}m to {stats['alt_max']:.2f}m
  Mean Altitude: {stats['alt_mean']:.2f}m
{set_alt_str}

Path samples: {stats['samples']}
"""
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mission maps from mission_travel_path.csv files.",
    )
    parser.add_argument("--root", required=True, type=Path,
                        help="Directory containing YYYYMMDDHHMMSS-<MissionName>/ mission folders")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for mission_map PNGs (created if missing)")
    parser.add_argument("--glob", default="*-*",
                        help="Glob pattern for mission folders (default: '*-*')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mission_dirs = sorted(p for p in args.root.glob(args.glob) if p.is_dir())
    if not mission_dirs:
        raise SystemExit(f"No mission folders matching '{args.glob}' found in {args.root}")

    counters: dict[str, int] = {}
    for d in mission_dirs:
        if "-" not in d.name:
            continue
        timestamp_prefix, mission_suffix = d.name.split("-", 1)
        try:
            dt = datetime.strptime(timestamp_prefix, "%Y%m%d%H%M%S")
        except ValueError:
            print(f"Skipping {d.name}: timestamp prefix not in YYYYMMDDHHMMSS form")
            continue

        counters[mission_suffix] = counters.get(mission_suffix, 0) + 1
        n = counters[mission_suffix]
        label = f"{mission_suffix} {n}"
        date_prefix = dt.strftime("%d%b%Y")
        out_path = args.out / f"{date_prefix}_{mission_suffix}_missionmap_{n}.png"
        print(f"Building {out_path.name} ...")
        build_map(d, out_path, label)

    print(f"Done. Wrote {sum(counters.values())} maps to {args.out}")


if __name__ == "__main__":
    main()
