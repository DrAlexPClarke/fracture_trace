# Fracture Trace (Skeletonise & Analyse)

This script is an omnivorous pre-processor for FracPaQ (https://github.com/DaveHealy-github/FracPaQ) intended to take as input any vector file with polygons or lines or any raster image file and produce a FracPaQ-compatible SVG file containing skeletonised traces of polygon features and simplified line traces of line features. This script will trace the centreline of any elongated polygonal features — such as fractures, faults, veins, dykes, or channels — and measuring their width profiles along the centreline. This script was originally developed for geological fracture analysis but is applicable to any dataset of elongated polygons.

---

## What it does

Given a vector or raster file containing polygon features, the script:

1. **Parses** the input and extracts individual polygon features.
2. **Rasterises** each polygon at a configurable resolution.
3. **Skeletonises** each polygon using one of several methods (see [Skeletonisation Methods](#skeletonisation-methods)).
4. **Measures width** at regular intervals along the skeleton by shooting perpendicular rays to the polygon walls.
5. **Exports** results as CSV data, width-profile plots, a FracPaQ-compatible skeleton SVG, and diagnostic overlay plots.

### Outputs

| File | Description |
|------|-------------|
| `skeleton.svg` | FracPaQ-compatible polyline SVG of all skeleton centrelines |
| `skeleton_raw.svg` | Full-density smoothed centrelines before simplification (optional) |
| `summary.csv` | One row per feature: length, mean/median width, tortuosity, etc. |
| `feature_<id>_branch_<n>_profile.csv` | Width at every sample point along each branch (optional) |
| `feature_<id>_branch_<n>_profile.svg` | Width-profile graph (optional) |
| `feature_<id>_tortuosity_fft.svg` | Lateral-deviation FFT spectrum plot (optional) |
| `feature_<id>_skeleton_overlay.svg` | Diagnostic overlay: raster, skeleton, branches, centreline (optional) |

---

## Supported input formats

| Format | Notes |
|--------|-------|
| `.shp` | Shapefile — requires a `.prj` sidecar; polygons read via GeoPandas |
| `.svg` | Scalable Vector Graphics — closed paths parsed directly |
| `.pdf` | PDF — polygon outlines extracted from vector paths via PyMuPDF |
| `.jpg` / `.jpeg` | Raster image — polygons extracted from dark regions via thresholding |
| `.png` | Raster image |
| `.tif` / `.tiff` | Raster image (GeoTIFF supported) |

---

## Installation

### Requirements

Python 3.9 or later is recommended.

```bash
pip install geopandas shapely scikit-image scipy numpy matplotlib pillow PyMuPDF
```

`scipy` is optional but strongly recommended — it provides Gaussian smoothing for the centreline. Without it, smoothing is disabled and a warning is printed.

### Clone

```bash
git clone https://github.com/<your-username>/fracture-width-analysis.git
cd fracture-width-analysis
```

---

## Usage

1. Open `skeletonise_and_analyse.py` in a text editor.
2. Set `INPUT_FILE` and `OUTPUT_DIRECTORY` at the top of the file.
3. Adjust any other configuration variables (see below).
4. Run:

```bash
python skeletonise_and_analyse.py
```

The script prints progress to the terminal as it processes each feature and writes all outputs to `OUTPUT_DIRECTORY`.

---

## Configuration reference

All configuration is done by editing the variables at the top of the script. There is no command-line interface.

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_FILE` | — | Path to the input file. Supported formats: `.shp`, `.svg`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` |
| `OUTPUT_DIRECTORY` | — | Directory where all outputs are written. Created automatically if it does not exist. |
| `SAMPLING_INTERVAL` | `1` | Spacing between successive width measurements, in world units. Smaller values give denser profiles. |
| `SMOOTHING` | `1` | Gaussian sigma for centreline smoothing, in world units. `0` disables smoothing. |
| `OUTPUT_RESOLUTION` | `1` | Minimum vertex spacing in `skeleton.svg`, in world units. `0` keeps all vertices. |
| `RDP_EPSILON` | `0.1` | Ramer–Douglas–Peucker tolerance for SVG simplification, in world units. Larger values produce smoother but less accurate polylines. |
| `MINIMUM_FEATURE_SIZE` | `0.5` | Features whose bounding-box side length is below this value (world units) are dropped before processing. `0` keeps all features. |

### Advanced settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RASTER_RESOLUTION` | `0.01` | World units per pixel when rasterising polygons. For shapefiles (metres), values around `0.000001`–`0.001` are typical; for PDFs and SVGs (points), `0.01`–`0.1` is typical. Ignored for raster inputs. |
| `IMAGE_THRESHOLD` | `128` | Greyscale threshold (0–255) used when parsing raster images. Pixels darker than this value are treated as polygon interior. |
| `MIN_BRANCH_PIXELS` | `5` | Minimum branch length in pixels. Branches shorter than this are treated as stubs and pruned. |
| `MIN_BRANCH_PERCENT` | `1.0` | Minimum branch length as a percentage of total skeleton pixels. Branches below this fraction are pruned. Both `MIN_BRANCH_PIXELS` and `MIN_BRANCH_PERCENT` must be satisfied to keep a branch. |
| `MAX_RAY_DISTANCE` | `None` | Maximum perpendicular ray length for width measurement, in world units. `None` sets this automatically to twice the polygon diagonal. |
| `MAX_RASTER_PIXELS` | `25_000_000` | Pixel budget per polygon. If rasterisation would exceed this, `RASTER_RESOLUTION` is automatically coarsened. |
| `MIN_RASTER_PIXELS` | `2_500_000` | Minimum pixel count per polygon. If rasterisation falls below this, `RASTER_RESOLUTION` is automatically refined. `0` disables this check. |
| `N_WORKERS` | `None` | Number of parallel worker threads for the compute stage. `None` uses all available CPU cores. |
| `TOPOLOGY_CONNECT` | `True` | If `True`, branch endpoints that abut a neighbouring feature are extended to meet that feature's skeleton, restoring T-junction and end-to-end connectivity after independent skeletonisation. |
| `SEPARATE_MULTIPOLYGONS` | `True` | If `True`, each component of a MultiPolygon feature is analysed independently (classic behaviour). If `False`, components are kept together and a half-pixel outward buffer is applied before rasterisation to bridge zero-width connections (pinch points, coincident edges, gaps). |

### Export options

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPORT_SKELETON_OVERLAY` | `True` | Save `feature_<id>_skeleton_overlay.svg` for every feature — a diagnostic plot showing the raster footprint, raw skeleton, branch pixels, and smoothed centreline. |
| `EXPORT_PROFILE_PLOT` | `True` | Save `feature_<id>_branch_<n>_profile.svg` width-profile graphs. |
| `EXPORT_PROFILE_DATA` | `True` | Save `feature_<id>_branch_<n>_profile.csv` per-branch width-profile CSVs. |
| `EXPORT_PROFILE_FFT` | `True` | Save `feature_<id>_tortuosity_fft.svg` lateral-deviation FFT spectrum plots. |
| `EXPORT_RAW_TRACES` | `False` | Save `skeleton_raw.svg` using the full-density smoothed centreline instead of the simplified output coordinates. Useful for inspecting the pre-simplification path. |

### Skeletonisation method

```python
SKELETONISATION_METHOD = "auto"
```

| Value | Description |
|-------|-------------|
| `"auto"` | Full automatic decision tree: geometry gate → directional attempt → Lee single-branch or multi-branch. **Recommended for most datasets.** |
| `"directional_and_single_branch"` | Same geometry gate and directional attempt as `"auto"`, but the Lee fallback always produces a single-branch result. Multi-branch is never the end-point. |
| `"directional"` | Directional cross-sections only. **No fallback**: if validation fails the feature produces no skeleton. Use `"directional_and_single_branch"` for a robust alternative. |
| `"single_or_multi_branch"` | Lee thinning with stub pruning; decides single- or multi-branch via `BRANCHING_THRESHOLD`. No directional stage. |
| `"single_branch"` | Lee thinning + graph diameter; always a single path. May stop short of rounded ends or fork at flat ends. |
| `"multi_branch"` | Full medial-axis skeleton with stub pruning. Use when branching is intentional (e.g., fracture networks with genuine junctions). |

### Decision-tree thresholds

These parameters govern the `"auto"` and `"directional_and_single_branch"` decision tree. The skeleton overlay plots display all threshold values alongside the measured values to help with tuning.

| Variable | Default | Description |
|----------|---------|-------------|
| `CURVATURE_THRESHOLD` | `1.3` | Maximum sinuosity (arc length / chord length) of the Lee skeleton diameter path before the directional method is skipped. `0` disables the curvature gate. |
| `SOLIDITY_THRESHOLD` | `0.2` | Minimum solidity (polygon area / convex-hull area). Features below this value are too non-convex for directional skeletonisation and use Lee instead. |
| `ASPECT_RATIO_THRESHOLD` | `3.0` | Minimum aspect ratio (long axis / short axis of the minimum rotated bounding rectangle). Features below this value are too compact for directional skeletonisation. |
| `ESCAPE_THRESHOLD` | `0.10` | Maximum fraction of interior directional-skeleton vertices allowed to lie outside the polygon. Exceeding this triggers a Lee fallback. |
| `BRANCHING_THRESHOLD` | `0.20` | In `"auto"` and `"single_or_multi_branch"` modes: fraction of post-pruning skeleton pixels that lie off the main (diameter) branch. Above this value, multi-branch is used; below, single-branch. |

---

## Skeletonisation methods explained

### Directional (cross-section)
Projects cross-sections perpendicular to the feature's long axis (determined by PCA of the boundary) and records the midpoint of each cross-section as a centreline point. Produces a clean, smooth path that extends all the way to the polygon ends, and handles flat terminations correctly. Best for simple, elongated, roughly-straight features.

### Lee single-branch
Applies Lee's morphological thinning algorithm to the rasterised polygon to produce a one-pixel-wide skeleton, then extracts the longest end-to-end path (the "diameter"). Handles curved and complex features well. May stop short of rounded ends or produce a fork at flat terminations.

### Lee multi-branch
Same thinning as single-branch but retains all skeleton branches (with stub pruning). Branch 0 is always the main spine (diameter path), guaranteeing that multi-branch is a strict superset of single-branch. Additional branches represent genuine side-arms, junctions, and secondary paths. Use for intentionally branching features.

### Auto mode decision tree

```
Geometry gate (pre-rasterisation)
  ├─ Interior holes?          → Lee
  ├─ Solidity < threshold?    → Lee
  ├─ Aspect ratio < threshold?→ Lee
  └─ Pass                     → attempt Directional
       ├─ Sinuosity too high?  → Lee
       ├─ Closed loop?         → Lee
       ├─ Escape > threshold?  → Lee
       └─ Pass                 → Directional ✓
                Lee path
                └─ off-main fraction > BRANCHING_THRESHOLD?
                     ├─ Yes → multi_branch
                     └─ No  → single_branch
```

---

## Output files in detail

### `skeleton.svg`
A FracPaQ-compatible polyline SVG. Coordinates are in the same world-unit space as the input geometry. This file can be imported directly into [FracPaQ](https://www.fracpaq.com/) for fracture network analysis. Branch endpoints that abut neighbouring features are topologically connected when `TOPOLOGY_CONNECT = True`.

### `summary.csv`
One row per feature with columns including:
- Feature ID, branch count
- Total centreline length
- Mean, median, minimum, and maximum width
- Width standard deviation
- Sinuosity (arc length / chord length)
- Dominant tortuosity wavelengths (from FFT)

### Profile CSVs (`feature_<id>_branch_<n>_profile.csv`)
One row per sample point with columns: sample distance along centreline, x/y world coordinates, total width, left-wall distance, right-wall distance.

### Skeleton overlay plots (`feature_<id>_skeleton_overlay.svg`)
Diagnostic multi-layer plots showing:
- Rasterised polygon footprint (faint grey)
- Source polygon with holes (blue fill)
- Discarded stub pixels (black dots)
- Kept branch pixels (colour-coded per branch)
- Smoothed centreline(s)

In auto/directional_and_single_branch modes the title encodes all decision-tree parameters and thresholds, making it easy to understand why a particular method was chosen and to tune the thresholds.

---

## Tips for different input types

### Shapefiles
- `RASTER_RESOLUTION` should be in the same units as the CRS. For a projected CRS in metres, try `0.001`–`0.01`.
- Ensure a `.prj` sidecar file is present alongside the `.shp` for correct coordinate handling.

### PDF / SVG
- World units are typographic points (1 pt ≈ 0.353 mm). `RASTER_RESOLUTION = 0.01` is a good starting point.
- For PDFs, only vector polygon paths are extracted — raster elements in the PDF are ignored.

### Raster images (JPG / PNG / TIFF)
- No rasterisation step is needed; pixels are used directly.
- Adjust `IMAGE_THRESHOLD` to correctly separate dark features from the background.
- `RASTER_RESOLUTION` is set automatically to 1 pixel = 1 world unit.

### Very large datasets
- Increase `N_WORKERS` for faster parallel processing (or set `None` for all cores).
- Reduce `MAX_RASTER_PIXELS` to cap per-feature memory usage.
- Disable `EXPORT_SKELETON_OVERLAY` and `EXPORT_PROFILE_PLOT` if only the CSV outputs are needed.

---

## Compatibility

The `skeleton.svg` output is designed to satisfy the strict import requirements of **FracPaQ v2**:
- Branches are `<polyline>` elements with a space-separated `points` attribute.
- Stroke styling uses the CSS class `.sk` declared in a `<defs>` block.
- The SVG uses a `viewBox` for geometry; no explicit `width`/`height` attributes.

---

## Licence

GNU General Public License v3
Copyright (c) 2026 Alex Clarke

---

## Citation

If you use this tool in published research, please cite it as:

> Clarke, A. (2026). *Fracture Width Analysis* [Software]. GitHub. https://github.com/<your-username>/fracture-width-analysis
