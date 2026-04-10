#!/usr/bin/env python3
"""
Extract georeferenced images from ROS bag for Agisoft Metashape.

Extracts down-facing and forward-facing camera images with corresponding
pose data to create camera reference CSV files for Metashape import.

Usage:
    python extract_georeferenced_images.py <bag_file> [output_dir]

Example:
    python extract_georeferenced_images.py /path/to/mission.bag /path/to/output

VICARIUS Integration:
    This module logs to the VICARIUS event stream per Commandment VIII.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# VICARIUS logging integration
VICARIUS_ROOT = os.environ.get("VICARIUS_ROOT", "/mnt/vicarius_drive/vicarius")
sys.path.insert(0, os.path.join(VICARIUS_ROOT, "_logging", "src"))
try:
    from vicarius_log import get_log
    VICARIUS_LOGGING = True
except ImportError:
    VICARIUS_LOGGING = False

import numpy as np
import pandas as pd
import cv2
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# Camera topic configuration
CAMERA_CONFIG = {
    "down": {
        "topic": "/science/image_raw",
        "description": "Down-facing science camera (4K)",
        "compressed": False,
    },
    "forward": {
        "topic": "/zed2/zed_node/left/image_rect_color",
        "description": "Forward-facing ZED2 camera",
        "compressed": False,
    },
}

POSE_TOPIC = "/pose"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract georeferenced images from ROS bag for Metashape"
    )
    parser.add_argument("bag_file", help="Path to the ROS bag file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--mission-json",
        default=None,
        help="Path to mission JSON file for waypoint speed overlay",
    )
    return parser.parse_args()


def setup_typestore(reader):
    """Register custom message types from bag file."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    for conn in reader.connections:
        if conn.msgdef:
            typestore.register(get_types_from_msg(conn.msgdef, conn.msgtype))
    return typestore


def extract_poses(reader, typestore):
    """Extract all pose data with timestamps."""
    poses = []
    pose_conns = [c for c in reader.connections if c.topic == POSE_TOPIC]

    if not pose_conns:
        print(f"  Warning: No pose topic found at {POSE_TOPIC}")
        return pd.DataFrame()

    for conn, timestamp, rawdata in reader.messages(connections=pose_conns):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
        poses.append({
            "timestamp_ns": timestamp,
            "x": msg.x,  # longitude
            "y": msg.y,  # latitude
            "depth": msg.depth,
            "altitude_dvl": msg.altitudeUsed,
            "heading": msg.heading,
            "pitch": msg.pitch,
            "roll": msg.roll,
        })

    return pd.DataFrame(poses)


def extract_and_save_images(reader, typestore, topic, output_dir, prefix, compressed=False):
    """Extract images from a topic and save to disk."""
    os.makedirs(output_dir, exist_ok=True)

    images = []
    conns = [c for c in reader.connections if c.topic == topic]

    if not conns:
        print(f"  Warning: No topic found at {topic}")
        return pd.DataFrame()

    for idx, (conn, timestamp, rawdata) in enumerate(reader.messages(connections=conns)):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

        if compressed:
            # Decode compressed image
            img_array = cv2.imdecode(
                np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
            )
        else:
            # Convert raw image to numpy array
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == "mono8":
                img_array = img_array.reshape((msg.height, msg.width))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif msg.encoding in ["bgr8", "rgb8"]:
                img_array = img_array.reshape((msg.height, msg.width, 3))
                if msg.encoding == "rgb8":
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgra8":
                img_array = img_array.reshape((msg.height, msg.width, 4))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            else:
                # Try generic 3-channel
                img_array = img_array.reshape((msg.height, msg.width, -1))[:, :, :3]

        filename = f"{prefix}_{idx:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])

        header_ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        images.append({
            "filename": filename,
            "timestamp_ns": header_ts_ns,
            "bag_timestamp_ns": timestamp,
            "index": idx,
        })

        if (idx + 1) % 20 == 0:
            print(f"    Extracted {idx + 1} images...")

    return pd.DataFrame(images)


def interpolate_poses_to_images(image_df, pose_df):
    """Interpolate pose data to exact image timestamps."""
    if image_df.empty or pose_df.empty:
        return pd.DataFrame()

    pose_ts = pose_df["timestamp_ns"].values.astype(float)

    interp_x = interpolate.interp1d(pose_ts, pose_df["x"].values, kind='linear', fill_value='extrapolate')
    interp_y = interpolate.interp1d(pose_ts, pose_df["y"].values, kind='linear', fill_value='extrapolate')
    interp_altitude = interpolate.interp1d(pose_ts, pose_df["altitude_dvl"].values, kind='linear', fill_value='extrapolate')
    interp_heading = interpolate.interp1d(pose_ts, pose_df["heading"].values, kind='linear', fill_value='extrapolate')
    interp_pitch = interpolate.interp1d(pose_ts, pose_df["pitch"].values, kind='linear', fill_value='extrapolate')
    interp_roll = interpolate.interp1d(pose_ts, pose_df["roll"].values, kind='linear', fill_value='extrapolate')

    matched = []
    for _, img_row in image_df.iterrows():
        img_time = float(img_row["bag_timestamp_ns"])

        matched.append({
            "filename": img_row["filename"],
            "longitude": float(interp_x(img_time)),
            "latitude": float(interp_y(img_time)),
            "altitude": float(interp_altitude(img_time)),
            "yaw": np.degrees(float(interp_heading(img_time))),
            "pitch": np.degrees(float(interp_pitch(img_time))),
            "roll": np.degrees(float(interp_roll(img_time))),
        })

    return pd.DataFrame(matched)


def export_metashape_csv(matched_df, output_path):
    """Export CSV for Metashape camera reference import."""
    if matched_df.empty:
        return matched_df

    export_df = matched_df[["filename", "longitude", "latitude", "altitude", "yaw", "pitch", "roll"]].copy()
    export_df.columns = ["label", "longitude", "latitude", "altitude", "yaw", "pitch", "roll"]
    export_df.to_csv(output_path, index=False)
    return export_df


def create_mission_map(pose_df, output_path, bag_name, stats, mission_json=None):
    """Create a map showing the mission path colored by speed, with statistics."""
    if pose_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                              gridspec_kw={'width_ratios': [3, 1]})

    # Left plot: Path map colored by speed
    ax1 = axes[0]
    lon = pose_df["x"].values
    lat = pose_df["y"].values
    ts = pose_df["timestamp_ns"].values.astype(float)

    # Compute speed from consecutive positions
    dt = np.diff(ts) / 1e9  # seconds
    cos_lat = np.cos(np.radians(np.mean(lat)))
    dx = np.diff(lon) * 111000 * cos_lat  # meters
    dy = np.diff(lat) * 111000  # meters
    dist = np.sqrt(dx**2 + dy**2)
    speed = np.where(dt > 0, dist / dt, 0.0)
    # Append 0 for the last point so array matches lon/lat length
    speed = np.append(speed, speed[-1] if len(speed) > 0 else 0.0)
    # Smooth speed with rolling median to reduce GPS noise
    speed = pd.Series(speed).rolling(window=15, center=True, min_periods=1).median().values

    scatter = ax1.scatter(lon, lat, c=speed, cmap='plasma', s=1, alpha=0.7,
                          vmin=0, vmax=1.0)
    ax1.plot(lon[0], lat[0], 'go', markersize=10, label='Start', zorder=5)
    ax1.plot(lon[-1], lat[-1], 'ro', markersize=10, label='End', zorder=5)

    # Overlay mission waypoints with commanded speeds if JSON provided
    if mission_json is not None:
        try:
            with open(mission_json, 'r') as f:
                mission = json.load(f)
            waypoints = mission.get("waypoints", [])
            # Draw planned mission path as a connected line
            wp_lons = [wp["longitude"] for wp in waypoints]
            wp_lats = [wp["latitude"] for wp in waypoints]
            ax1.plot(wp_lons, wp_lats, '--', color='white', linewidth=3.0, zorder=3)
            ax1.plot(wp_lons, wp_lats, '--', color='limegreen', linewidth=1.8, zorder=3,
                     label='Planned path')
            # Draw waypoint markers and labels
            for wp in waypoints:
                wp_lon = wp["longitude"]
                wp_lat = wp["latitude"]
                wp_speed = wp["speed"]
                ax1.plot(wp_lon, wp_lat, 'D', color='white', markersize=8,
                         markeredgecolor='limegreen', markeredgewidth=1.5, zorder=4)
                ax1.annotate(f'WP{wp["waypoint_number"]}\n{wp_speed:.2f} m/s',
                             (wp_lon, wp_lat), textcoords="offset points",
                             xytext=(8, 8), fontsize=7, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                       edgecolor='limegreen', alpha=0.8),
                             zorder=7)
        except Exception as e:
            print(f"  Warning: Could not overlay mission waypoints: {e}")

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Mission Path: {bag_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.ticklabel_format(useOffset=False, style='plain')
    plt.colorbar(scatter, ax=ax1, label='Speed (m/s)')

    # Right plot: Stats text
    ax2 = axes[1]
    ax2.axis('off')

    # Duration in minutes.seconds
    duration_min = int(stats['duration'] // 60)
    duration_sec = stats['duration'] % 60

    # Get set altitude from mission JSON if available
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

    stats_text = f"""
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

"""

    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    start_time = time.time()
    start_event_id = None

    bag_path = Path(args.bag_file)
    if not bag_path.exists():
        print(f"Error: Bag file not found: {bag_path}")
        sys.exit(1)

    # Create output directory named after bag file
    bag_name = bag_path.stem
    output_base = Path(args.output_dir) / bag_name
    output_base.mkdir(parents=True, exist_ok=True)

    # VICARIUS: Log process start
    if VICARIUS_LOGGING:
        try:
            log = get_log()
            start_event_id = log.process_start(
                module="bag_metashape_export",
                purpose=f"Extract georeferenced images from {bag_name}",
                inputs=[str(bag_path)]
            )
        except Exception as e:
            print(f"  Warning: VICARIUS logging unavailable: {e}")

    print(f"{'='*60}")
    print(f"ROS Bag Georeferenced Image Extractor")
    print(f"{'='*60}")
    print(f"Input: {bag_path}")
    print(f"Output: {output_base}")
    print()

    # Extract pose data
    print("Extracting pose data...")
    with Reader(str(bag_path)) as reader:
        typestore = setup_typestore(reader)
        pose_df = extract_poses(reader, typestore)
    print(f"  Found {len(pose_df)} pose messages")

    if pose_df.empty:
        print("Error: No pose data found. Cannot continue.")
        sys.exit(1)

    # Calculate statistics
    duration = (pose_df["timestamp_ns"].max() - pose_df["timestamp_ns"].min()) / 1e9
    start_ts = pose_df["timestamp_ns"].min() / 1e9
    end_ts = pose_df["timestamp_ns"].max() / 1e9

    stats = {
        "start_time": datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
        "lon_min": pose_df["x"].min(),
        "lon_max": pose_df["x"].max(),
        "lat_min": pose_df["y"].min(),
        "lat_max": pose_df["y"].max(),
        "lon_span_m": (pose_df["x"].max() - pose_df["x"].min()) * 111000 * np.cos(np.radians(pose_df["y"].mean())),
        "lat_span_m": (pose_df["y"].max() - pose_df["y"].min()) * 111000,
        "alt_min": pose_df["altitude_dvl"].min(),
        "alt_max": pose_df["altitude_dvl"].max(),
        "alt_mean": pose_df["altitude_dvl"].mean(),
        "pose_samples": len(pose_df),
        "pose_rate": len(pose_df) / duration if duration > 0 else 0,
        "down_images": 0,
        "forward_images": 0,
    }

    # Process each camera
    for camera_name, config in CAMERA_CONFIG.items():
        print(f"\nProcessing {camera_name} camera ({config['description']})...")

        image_dir = output_base / f"{camera_name}_images"

        with Reader(str(bag_path)) as reader:
            typestore = setup_typestore(reader)
            image_df = extract_and_save_images(
                reader, typestore,
                config["topic"], str(image_dir),
                camera_name, config["compressed"]
            )

        if image_df.empty:
            print(f"  No images found for {camera_name} camera")
            continue

        print(f"  Extracted {len(image_df)} images to {image_dir}")
        stats[f"{camera_name}_images"] = len(image_df)

        # Interpolate poses
        print(f"  Interpolating poses...")
        matched_df = interpolate_poses_to_images(image_df, pose_df)

        # Export CSV
        csv_path = output_base / f"{camera_name}_reference.csv"
        export_metashape_csv(matched_df, str(csv_path))
        print(f"  Exported reference: {csv_path}")

        if not matched_df.empty:
            print(f"    Altitude range: {matched_df['altitude'].min():.2f}m to {matched_df['altitude'].max():.2f}m")

    # Create mission map
    print(f"\nGenerating mission map...")
    map_path = output_base / "mission_map.png"
    create_mission_map(pose_df, str(map_path), bag_name, stats, mission_json=args.mission_json)
    print(f"  Saved: {map_path}")

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_base}")
    print(f"  - down_images/: {stats['down_images']} images")
    print(f"  - down_reference.csv")
    print(f"  - forward_images/: {stats['forward_images']} images")
    print(f"  - forward_reference.csv")
    print(f"  - mission_map.png")
    print(f"  - Duration: {elapsed:.1f} seconds")
    print()
    print("METASHAPE IMPORT:")
    print("  1. Add photos from down_images/ or forward_images/")
    print("  2. Reference pane → Import → Select corresponding _reference.csv")
    print("  3. Settings: WGS84, Columns: Label=1, Lon=2, Lat=3, Alt=4, Yaw=5, Pitch=6, Roll=7")

    # VICARIUS: Log process end
    if VICARIUS_LOGGING:
        try:
            log = get_log()
            total_images = stats['down_images'] + stats['forward_images']
            log.process_end(
                module="bag_metashape_export",
                status="success",
                duration_sec=elapsed,
                outputs=[str(output_base)],
                notes=f"Extracted {total_images} images from {bag_name}"
            )
        except Exception:
            pass  # Logging is best-effort


if __name__ == "__main__":
    main()
