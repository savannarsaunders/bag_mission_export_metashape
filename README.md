# bag_metashape_export

Extract georeferenced images from ROS bag files for Agisoft Metashape photogrammetry.

> **Note:** This module was incorporated into the [VICARIUS](https://github.com/vicar-cmes/vicarius) marine research data platform in February 2026. It includes VICARIUS logging integration and run management.

## Purpose

This module processes ROS bag files from AUV/ROV missions, extracting camera images and synchronizing them with pose data to create georeferenced camera reference files for Metashape import. It supports the photogrammetry workflow for 3D reef structure analysis.

## Quick Start

```bash
# Install dependencies
cd $VICARIUS_ROOT/modules/bag_metashape_export/github_repo
pip install -r requirements.txt

# Initialize a processing run
python src/init_run.py my_run --purpose "Extract images from Jan 2026 mission" --study S2_3D_structure

# Run the extraction
python extract_georeferenced_images.py \
    $VICARIUS_ROOT/raw/auv/mission.bag \
    ../inprocess/my_run/outputs/

# Shelve the run when done
python src/shelve_run.py my_run --disposition keep --notes "Extraction complete"
```

## Inputs

| Input | Type | Format | Source | Required |
|-------|------|--------|--------|----------|
| bag_file | file | .bag | raw/auv/ | yes |

### Expected ROS Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/pose` | `rangerbot_msgs/pose` | Position (x=lon, y=lat), depth, heading, pitch, roll, altitudeUsed |
| `/science/image_raw` | `sensor_msgs/Image` | Down-facing science camera (4K) |
| `/zed2/zed_node/left/image_rect_color` | `sensor_msgs/Image` | Forward-facing ZED2 camera |

### Other Topics (may vary)

| Topic | Description |
|-------|-------------|
| `/pressure` | Pressure sensor depth (true depth below surface) |
| `/dvl_dr` | DVL dead reckoning position |
| `/dvl_fix` | DVL velocity and altitude |
| `/stereo_down/left/image_mono` | Stereo down camera (mono) |
| `/mavros/imu/data` | IMU data |
| `/tf` | Transform tree |

## Outputs

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| down_images/ | directory | .jpg | Down-facing camera images |
| forward_images/ | directory | .jpg | Forward-facing camera images |
| down_reference.csv | file | .csv | Georeferenced positions for down camera |
| forward_reference.csv | file | .csv | Georeferenced positions for forward camera |
| mission_map.png | file | .png | Path visualization with statistics |

Output structure:
```
{output_dir}/{bag_name}/
  ├── down_images/         # Down-facing camera images
  ├── down_reference.csv   # Georeferenced positions for down camera
  ├── forward_images/      # Forward-facing camera images
  ├── forward_reference.csv# Georeferenced positions for forward camera
  └── mission_map.png      # Path visualization with statistics
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| bag_file | string | (required) | Path to the ROS bag file |
| output_dir | string | "." | Output directory |

## Usage Examples

### Example 1: Basic extraction

```bash
python extract_georeferenced_images.py /path/to/mission.bag /path/to/output
```

### Example 2: Within VICARIUS workflow

```bash
# Create a run for processing multiple bags
python src/init_run.py jan2026_missions \
    --purpose "Extract all January 2026 AUV missions" \
    --study S2_3D_structure

# Process each bag
for bag in $VICARIUS_ROOT/raw/auv/*.bag; do
    python extract_georeferenced_images.py "$bag" ../inprocess/jan2026_missions/outputs/
done
```

## Metashape Import

After running this module:

1. Add photos from `down_images/` or `forward_images/`
2. Reference pane → Import → select corresponding `_reference.csv`
3. Settings:
   - Coordinate System: **WGS 84 (EPSG:4326)**
   - Columns: Label=1, Longitude=2, Latitude=3, Altitude=4, Yaw=5, Pitch=6, Roll=7
   - Start row: 2
   - Rotation: Yaw, Pitch, Roll (degrees)

## Logging Integration

This module integrates with the VICARIUS logging system (Commandment VIII):

```python
import sys
sys.path.insert(0, "/mnt/vicarius_drive/vicarius/_logging/src")
from vicarius_log import get_log

log = get_log()

# At start of processing
start_event = log.process_start(
    module="bag_metashape_export",
    purpose="Extract georeferenced images from mission bag",
    study="S2_3D_structure",
    inputs=["/path/to/bag"]
)

# At end of processing
log.process_end(
    module="bag_metashape_export",
    status="success",
    duration_sec=elapsed,
    outputs=["/path/to/output"]
)
```

## Run Management

This module uses standard VICARIUS run management:

```bash
# Create a new run
python src/init_run.py my_run_name --purpose "Description" --study S2_3D_structure

# After processing, archive the run
python src/shelve_run.py my_run_name --disposition keep --notes "Final notes"
```

## Configuration

Edit `CAMERA_CONFIG` in the script to change camera topics:

```python
CAMERA_CONFIG = {
    "down": {
        "topic": "/science/image_raw",
        "compressed": False,
    },
    "forward": {
        "topic": "/zed2/zed_node/left/image_rect_color",
        "compressed": False,
    },
}
```

Set `compressed: True` for `sensor_msgs/CompressedImage` topics.

## Runtime Requirements

- **Python:** >=3.10
- **GPU:** Not required
- **Tested OS:** Ubuntu 22.04, Ubuntu 24.04
- **Dependencies:** See requirements.txt

## Known Limitations

- Requires specific ROS topic structure (rangerbot_msgs/pose)
- Camera topics may need customization for different vehicle configurations
- Large bag files may take several minutes to process

## Troubleshooting

**Problem:** No pose data found
**Solution:** Check that `/pose` topic exists in the bag: `rosbag info mission.bag`

**Problem:** No images extracted
**Solution:** Verify camera topics match your bag file; edit CAMERA_CONFIG if needed

**Problem:** Import errors
**Solution:** Install dependencies: `pip install -r requirements.txt`

## Companion scripts (CSV-driven, no `rosbags` required)

Two helper scripts work off the already-exported `mission_travel_path.csv`
files (one per timestamped mission folder) instead of the raw bag. They
don't depend on `rosbags` or Python ≥ 3.10, so they're useful for quick
batch reporting on a laptop without a ROS toolchain.

Expected mission-root layout (one per dock/site visit):

```
<mission-root>/
  DocSsur/                                   # bag dir (per mission name)
    DocSsur_0_UTC_2026-06-24_16-31-35.bag
  20260624123134-DocSsur/                    # exported mission folder
    DocSsur.json
    mission_travel_path.csv
    mission_summary.json
  ...
```

### `generate_mission_maps.py`

Renders the same speed-colored path + planned-waypoint overlay + stats
panel as `extract_georeferenced_images.py::create_mission_map`, but reads
pose from `mission_travel_path.csv`. For each `.bag` under the
`--bag-dirs` it finds the timestamped mission folder whose name suffix
matches the bag's mission name and whose timestamp is within
`--match-tolerance` seconds (after applying `--utc-offset-hours`), then
writes `<bag-stem>_mission_map.png` into `--out`. Pass `--site-label` to
render a `SITE` header above the `MISSION STATISTICS` block (e.g. so
several dock visits to the same vehicle stay distinguishable in a report).

```bash
python generate_mission_maps.py \
    "/path/to/24Jun2026_LemonDock" \
    --site-label "Rangerbot: Lemon"
# -> /path/to/24Jun2026_LemonDock/mission_maps/*_mission_map.png
```

### `generate_error_summary.py`

Builds `mission_error_state_summary.docx` from every
`mission_travel_path.csv` under the mission root. Finds contiguous runs of
non-zero `errorState`, resolves the name from the RangerBot error-code
table (`ERROR_NAMES` in the script, sourced from `RBerrorcodes.PDF`), and
— for each error block — reconstructs the waypoint the vehicle was driving
toward by replaying the GPS track against the mission plan's waypoint
list, advancing whenever the vehicle enters that waypoint's capture
radius. The "Patterns worth noting" bullets at the top are generated from
the observed data (e.g. "all blocks at near-surface depth", "errorState=9
is the only code observed"), so they describe each dataset rather than
restating a template.

```bash
python generate_error_summary.py \
    "/path/to/24Jun2026_LemonDock" \
    --site-label "Rangerbot: Lemon" \
    --dataset-label "24 June 2026, Lemon Dock"
# -> /path/to/24Jun2026_LemonDock/mission_error_state_summary.docx
```

Both scripts only need `pandas`, `numpy`, `matplotlib`, and (for the
summary doc) `python-docx` — all already listed in `requirements.txt`.

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-06-24 | Add CSV-driven `generate_mission_maps.py` and `generate_error_summary.py` companion scripts |
| 1.0.0 | 2026-02-04 | Initial VICARIUS integration |

## Related Documentation

- [Module Checklist](../../_DOCS/MODULE_CHECKLIST.md)
- [Module Registry](../MODULE_REGISTRY.md)
- [VICARIUS Logging](../../_logging/README.md)
- [Ten Commandments](../../_DOCS/TEN_COMMANDMENTS.md)
