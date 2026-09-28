#!/usr/bin/env python3
"""Generate a per-mission ``*_mission_map.png`` for every ``.bag`` that has a
matching exported mission folder.

This is a CSV-driven companion to ``extract_georeferenced_images.py``. It
reuses the same `create_mission_map` visual style (speed-colored path,
start/end markers, planned-waypoint overlay, stats panel) but reads pose
data from the per-mission ``mission_travel_path.csv`` exported by the
RangerBot mission runner rather than from the bag itself, so it does not
require ``rosbags`` or Python >= 3.10.

Expected mission-root layout (each mission produces both a bag in a
``<mission-name>/`` directory and an exported folder named
``YYYYMMDDHHMMSS-<mission-name>/``)::

    <mission-root>/
      DocSsur/                                  # bag dir
        DocSsur_0_UTC_2026-06-24_16-31-35.bag
      20260624123134-DocSsur/                   # exported mission folder
        DocSsur.json
        mission_travel_path.csv
        mission_summary.json

The script walks every ``<bag-dir>/*.bag``, shifts the bag's UTC timestamp
to local time, and finds the timestamped mission folder whose name suffix
matches the bag's mission name and whose timestamp is within
``--match-tolerance`` seconds. For matched bags it writes
``<bag-stem>_mission_map.png`` into ``--out``.

Example::

    python generate_mission_maps.py \\
        /Users/vicar/Desktop/Rangerbot\\ Mission\\ \\(Keep\\)/24Jun2026_LemonDock \\
        --site-label "Rangerbot: Lemon"
"""

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

DEFAULT_BAG_DIRS = ("DocSsub", "DocSsubm", "DocSsur", "GOTO")

BAG_TS_RE = re.compile(
    r"_UTC_(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})_(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})\.bag$"
)
FOLDER_TS_RE = re.compile(r"^(?P<ts>\d{14})-(?P<name>.+)$")


def bag_utc_dt(bag_path: Path) -> Optional[datetime]:
    """Parse the UTC datetime encoded in a bag filename, or None if absent."""
    m = BAG_TS_RE.search(bag_path.name)
    if not m:
        return None
    return datetime(
        int(m["y"]), int(m["mo"]), int(m["d"]),
        int(m["h"]), int(m["mi"]), int(m["s"]),
        tzinfo=timezone.utc,
    )


def find_mission_folder(bag_path: Path, root: Path,
                        local_offset: timedelta,
                        tolerance_s: float) -> Optional[Path]:
    """Return the timestamped mission folder matching this bag, or None.

    Matches by mission-name suffix + closest folder timestamp within
    ``tolerance_s`` seconds of the bag's local-time timestamp.
    """
    m = BAG_TS_RE.search(bag_path.name)
    if not m:
        return None
    utc = datetime(
        int(m["y"]), int(m["mo"]), int(m["d"]),
        int(m["h"]), int(m["mi"]), int(m["s"]),
        tzinfo=timezone.utc,
    )
    local = (utc + local_offset).replace(tzinfo=None)

    # Bag stem like 'DocSsubm_1_UTC_2026-06-24_16-44-04' -> mission name 'DocSsubm'.
    mission_name = bag_path.stem.split("_")[0]

    best: Optional[Path] = None
    best_dt = None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        fm = FOLDER_TS_RE.match(d.name)
        if not fm or fm["name"] != mission_name:
            continue
        try:
            folder_dt = datetime.strptime(fm["ts"], "%Y%m%d%H%M%S")
        except ValueError:
            continue
        diff = abs((folder_dt - local).total_seconds())
        if diff <= tolerance_s and (best_dt is None or diff < best_dt):
            best, best_dt = d, diff
    return best


def find_mission_json(mission_folder: Path) -> Optional[Path]:
    """Find the per-mission plan JSON (e.g. ``DocSsur.json``) in the folder.

    These have a ``waypoints`` list. ``mission_summary.json`` does not and
    is skipped.
    """
    for p in mission_folder.glob("*.json"):
        if p.name == "mission_summary.json":
            continue
        try:
            with p.open() as f:
                data = json.load(f)
            if isinstance(data, dict) and "waypoints" in data:
                return p
        except Exception:
            continue
    return None


def load_pose_df(csv_path: Path) -> pd.DataFrame:
    """Load the travel-path CSV into the column shape ``create_mission_map``
    expects: ``timestamp_ns, x (lon), y (lat), depth, altitude_dvl, heading,
    pitch, roll``.
    """
    df = pd.read_csv(csv_path)
    return pd.DataFrame({
        "timestamp_ns": (df["timestamp_ros"].astype(float) * 1e9).astype("int64"),
        "x": df["longitude"].astype(float),
        "y": df["latitude"].astype(float),
        "depth": df["depth"].astype(float),
        "altitude_dvl": df["altitudeUsed"].astype(float),
        "heading": df["yaw"].astype(float),
        "pitch": df["pitch"].astype(float),
        "roll": df["roll"].astype(float),
    })


def compute_stats(pose_df: pd.DataFrame) -> dict:
    """Mirror the stats dict assembled in ``main()`` of the upstream script."""
    duration = (pose_df["timestamp_ns"].max() - pose_df["timestamp_ns"].min()) / 1e9
    start_ts = pose_df["timestamp_ns"].min() / 1e9
    end_ts = pose_df["timestamp_ns"].max() / 1e9
    return {
        "start_time": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
        "lon_min": pose_df["x"].min(),
        "lon_max": pose_df["x"].max(),
        "lat_min": pose_df["y"].min(),
        "lat_max": pose_df["y"].max(),
        "lon_span_m": (pose_df["x"].max() - pose_df["x"].min())
            * 111000 * np.cos(np.radians(pose_df["y"].mean())),
        "lat_span_m": (pose_df["y"].max() - pose_df["y"].min()) * 111000,
        "alt_min": pose_df["altitude_dvl"].min(),
        "alt_max": pose_df["altitude_dvl"].max(),
        "alt_mean": pose_df["altitude_dvl"].mean(),
    }


def place_labels_without_overlap(fig, ax, labels) -> None:
    """Move each annotation to the nearest offset where its box overlaps no
    other waypoint label or waypoint marker and stays inside the axes."""
    if not labels:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    pad = 3  # px clearance between boxes
    marker_r = 7  # px half-size of the waypoint diamond

    anchors = [ax.transData.transform(a.xy) for a in labels]
    obstacles = [(x - marker_r, y - marker_r, x + marker_r, y + marker_r)
                 for x, y in anchors]

    def box_of(a):
        # Text-only extent (excludes the leader line); recomputed from the
        # current xyann, then grown by the bbox padding + clearance.
        a.update_positions(renderer)
        bb = Text.get_window_extent(a, renderer)
        grow = pad + 4
        return (bb.x0 - grow, bb.y0 - grow, bb.x1 + grow, bb.y1 + grow)

    def hits(b, others):
        return any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1]
                   for o in others)

    def inside(b):
        return (b[0] >= ax_box.x0 and b[2] <= ax_box.x1 and
                b[1] >= ax_box.y0 and b[3] <= ax_box.y1)

    # Candidate offsets (points): 8 directions at increasing radius.
    candidates = []
    for r in (8, 20, 35, 50, 70, 95, 125):
        for ang in (45, 135, 315, 225, 90, 270, 0, 180, 22, 158, 338, 202):
            dx, dy = r * np.cos(np.radians(ang)), r * np.sin(np.radians(ang))
            candidates.append((dx, dy))

    placed = []
    for a in labels:
        best = None
        for dx, dy in candidates:
            # Anchor the box's nearest corner/edge toward the waypoint.
            a.set_ha('left' if dx > 1 else 'right' if dx < -1 else 'center')
            a.set_va('bottom' if dy > 1 else 'top' if dy < -1 else 'center')
            a.xyann = (dx, dy)
            b = box_of(a)
            if inside(b) and not hits(b, placed) and not hits(b, obstacles):
                best = b
                break
        if best is None:  # nothing fully clear; fall back to first candidate
            dx, dy = candidates[0]
            a.set_ha('left'); a.set_va('bottom'); a.xyann = (dx, dy)
            best = box_of(a)
        # Hide the leader line when the box sits right next to its point.
        if np.hypot(*a.xyann) <= 10:
            a.arrow_patch.set_visible(False)
        placed.append(best)


def compute_waypoint_arrivals(pose_df: pd.DataFrame, waypoints: list) -> list:
    """For each planned waypoint, return the actual time the RangerBot reached
    it, defined as the pose sample of closest approach (in metres).

    Waypoints are matched in mission order: the search for waypoint *k* starts
    at the pose sample where waypoint *k-1* was reached, so a later waypoint can
    never bind to a pose sample the vehicle had already passed for an earlier
    one (important when a survey path doubles back near an earlier target).

    Returns a list aligned with ``waypoints``; each entry is a dict with
    ``waypoint_number``, ``timestamp_ns`` (int), ``time_str`` (local HH:MM:SS),
    and ``distance_m`` (closest-approach distance), or ``None`` if no pose
    samples remain to search.
    """
    if pose_df.empty or not waypoints:
        return [None] * len(waypoints)

    lon = pose_df["x"].to_numpy(dtype=float)
    lat = pose_df["y"].to_numpy(dtype=float)
    ts_ns = pose_df["timestamp_ns"].to_numpy(dtype="int64")
    cos_lat = np.cos(np.radians(np.nanmean(lat)))

    arrivals = []
    search_start = 0
    n = len(pose_df)
    for wp in waypoints:
        if search_start >= n:
            arrivals.append(None)
            continue
        dx = (lon[search_start:] - wp["longitude"]) * 111000 * cos_lat
        dy = (lat[search_start:] - wp["latitude"]) * 111000
        d = np.sqrt(dx * dx + dy * dy)
        j = int(np.argmin(d))
        idx = search_start + j
        t_ns = int(ts_ns[idx])
        arrivals.append({
            "waypoint_number": wp.get("waypoint_number"),
            "timestamp_ns": t_ns,
            "time_str": datetime.fromtimestamp(t_ns / 1e9).strftime("%H:%M:%S"),
            "distance_m": float(d[j]),
        })
        search_start = idx  # keep arrivals monotonic in time
    return arrivals


def create_mission_map(pose_df, output_path, bag_name, stats,
                       mission_json=None, site_label: Optional[str] = None,
                       display_name: Optional[str] = None):
    """Port of the upstream ``create_mission_map`` function. If ``site_label``
    is provided, a ``SITE`` header is rendered at the top of the stats panel.

    ``display_name`` is the human-friendly mission label shown as the plot
    title (e.g. ``"BITfW 1"``); it defaults to ``bag_name`` if not given. The
    full bag file name is still recorded verbatim in the stats panel."""
    if pose_df.empty:
        return
    title_name = display_name or bag_name

    fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                             gridspec_kw={'width_ratios': [3, 1]})

    ax1 = axes[0]
    lon = pose_df["x"].values
    lat = pose_df["y"].values
    ts = pose_df["timestamp_ns"].values.astype(float)

    dt = np.diff(ts) / 1e9
    cos_lat = np.cos(np.radians(np.mean(lat)))
    dx = np.diff(lon) * 111000 * cos_lat
    dy = np.diff(lat) * 111000
    dist = np.sqrt(dx**2 + dy**2)
    speed = np.where(dt > 0, dist / dt, 0.0)
    speed = np.append(speed, speed[-1] if len(speed) > 0 else 0.0)
    speed = pd.Series(speed).rolling(window=15, center=True, min_periods=1).median().values

    scatter = ax1.scatter(lon, lat, c=speed, cmap='plasma', s=1, alpha=0.7,
                          vmin=0, vmax=1.0)
    ax1.plot(lon[0], lat[0], 'go', markersize=10, label='Start', zorder=5)
    ax1.plot(lon[-1], lat[-1], 'ro', markersize=10, label='End', zorder=5)

    arrival_block = ""  # filled below; rendered in the stats panel
    wp_labels = []  # waypoint annotations, de-overlapped before saving
    if mission_json is not None:
        try:
            with open(mission_json, 'r') as f:
                mission = json.load(f)
            waypoints = mission.get("waypoints", [])
            arrivals = compute_waypoint_arrivals(pose_df, waypoints)
            wp_lons = [wp["longitude"] for wp in waypoints]
            wp_lats = [wp["latitude"] for wp in waypoints]
            ax1.plot(wp_lons, wp_lats, '--', color='white', linewidth=3.0, zorder=3)
            ax1.plot(wp_lons, wp_lats, '--', color='limegreen', linewidth=1.8, zorder=3,
                     label='Planned path')
            for wp, arr in zip(waypoints, arrivals):
                wp_lon = wp["longitude"]
                wp_lat = wp["latitude"]
                wp_speed = wp["speed"]
                wp_type = wp.get("additional_data", {}).get("transect_type")
                type_str = f"\n{wp_type}" if wp_type else ""
                ax1.plot(wp_lon, wp_lat, 'D', color='white', markersize=8,
                         markeredgecolor='limegreen', markeredgewidth=1.5, zorder=4)
                wp_labels.append(ax1.annotate(
                    f'WP{wp["waypoint_number"]}\n{wp_speed:.2f} m/s{type_str}',
                    (wp_lon, wp_lat), textcoords="offset points",
                    xytext=(8, 8), fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='limegreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='-', color='limegreen', lw=0.8,
                                    shrinkA=0, shrinkB=4),
                    zorder=7))

            # Build the "WAYPOINT ARRIVALS" table for the stats panel.
            arrival_lines = []
            for arr in arrivals:
                if not arr:
                    continue
                flag = "  *" if arr["distance_m"] > 3.0 else ""
                arrival_lines.append(
                    f"  WP{arr['waypoint_number']:<3} {arr['time_str']}"
                    f"  ({arr['distance_m']:.1f}m){flag}"
                )
            if arrival_lines:
                arrival_block = (
                    "WAYPOINT ARRIVALS (local)\n"
                    + "=" * 40 + "\n\n"
                    + "\n".join(arrival_lines)
                    + "\n\n  (time = closest approach on actual track;\n"
                    + "   value in () is closest-approach distance,\n"
                    + "   * = never came within 3 m of the waypoint)\n"
                )
        except Exception as e:
            print(f"  Warning: Could not overlay mission waypoints: {e}")

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Mission Path: {title_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.ticklabel_format(useOffset=False, style='plain')
    plt.colorbar(scatter, ax=ax1, label='Speed (m/s)')

    ax2 = axes[1]
    ax2.axis('off')

    duration_min = int(stats['duration'] // 60)
    duration_sec = stats['duration'] % 60

    set_alt_str = "  Set Altitude: N/A"
    if mission_json is not None:
        try:
            with open(mission_json, 'r') as f:
                mission_data = json.load(f)
            for wp in mission_data.get("waypoints", []):
                alt = wp.get("additional_data", {}).get("Altitude")
                if alt is not None:
                    set_alt_str = f"  Set Altitude: {alt:.1f}m"
                    break
        except Exception:
            pass

    site_block = ""
    if site_label:
        site_block = f"""
SITE
{'='*40}

  {site_label}
"""

    stats_text = f"""{site_block}
MISSION STATISTICS
{'='*40}

Bag File: {bag_name}

TIME
  Start: {stats['start_time']}
  End: {stats['end_time']}
  Duration: {duration_min}m {duration_sec:.1f}s

LOCATION (WGS84)
  Longitude: {stats['lon_min']:.6f}° to {stats['lon_max']:.6f}°
  Latitude: {stats['lat_min']:.6f}° to {stats['lat_max']:.6f}°
  Span: {stats['lon_span_m']:.1f}m x {stats['lat_span_m']:.1f}m

DEPTH / ALTITUDE
  DVL Altitude: {stats['alt_min']:.2f}m to {stats['alt_max']:.2f}m
  Mean Altitude: {stats['alt_mean']:.2f}m
{set_alt_str}

{arrival_block}"""

    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    place_labels_without_overlap(fig, ax1, wp_labels)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("mission_root", type=Path,
                    help="Mission root directory containing the bag dirs and "
                         "exported YYYYMMDDHHMMSS-<name>/ folders.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory for the PNGs "
                         "(default: <mission-root>/mission_maps).")
    ap.add_argument("--bag-dirs", nargs="+", default=list(DEFAULT_BAG_DIRS),
                    help="Subdirectories under <mission-root> that contain "
                         f".bag files (default: {' '.join(DEFAULT_BAG_DIRS)}).")
    ap.add_argument("--site-label", default=None,
                    help="Optional site name rendered as a SITE section atop "
                         "the stats panel (e.g. 'Rangerbot: Lemon').")
    ap.add_argument("--utc-offset-hours", type=float, default=-4.0,
                    help="Local-time offset from UTC used to align bag "
                         "timestamps to folder timestamps. Default -4 (EDT).")
    ap.add_argument("--match-tolerance", type=float, default=5.0,
                    help="Max seconds of skew allowed between a bag's "
                         "local-time timestamp and a candidate folder's "
                         "timestamp (default: 5).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.mission_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"mission_root not found: {root}")
    out = (args.out or (root / "mission_maps")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    local_offset = timedelta(hours=args.utc_offset_hours)

    rows = []
    for sub in args.bag_dirs:
        d = root / sub
        if not d.is_dir():
            continue
        # Order this mission's bags by their actual UTC time so the display
        # sequence number reflects run order (BITfW_0 -> "BITfW 1", etc.).
        # Bags with unparseable names sort last, by filename. The index spans
        # every bag in the dir, so a skipped bag still occupies its ordinal.
        bags = sorted(d.glob("*.bag"),
                      key=lambda b: (bag_utc_dt(b) is None, bag_utc_dt(b) or b.name))
        for idx, bag in enumerate(bags, start=1):
            mission_name = bag.stem.split("_")[0]
            display_name = f"{mission_name} {idx}"
            bdt = bag_utc_dt(bag)
            date_str = ((bdt + local_offset).strftime("%d%b%Y")
                        if bdt is not None else "unknown-date")
            folder = find_mission_folder(bag, root, local_offset,
                                         args.match_tolerance)
            if folder is None:
                rows.append((bag.name, "skipped (no matched mission folder)", ""))
                continue
            csv = folder / "mission_travel_path.csv"
            if not csv.exists():
                rows.append((bag.name, f"skipped (no CSV in {folder.name})", ""))
                continue
            mission_json = find_mission_json(folder)
            pose_df = load_pose_df(csv)
            stats = compute_stats(pose_df)
            out_path = out / f"{date_str}_{mission_name}_{idx}_missionmap.png"
            create_mission_map(pose_df, str(out_path), bag.stem, stats,
                               mission_json=str(mission_json) if mission_json else None,
                               site_label=args.site_label,
                               display_name=display_name)
            rows.append((bag.name, f"-> {out_path.name}",
                         f"[{display_name}] folder={folder.name} "
                         f"wp={'yes' if mission_json else 'no'}"))

    width = max(len(r[0]) for r in rows) if rows else 0
    print(f"{'BAG':<{width}}  RESULT")
    for name, result, note in rows:
        print(f"{name:<{width}}  {result}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
