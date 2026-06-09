"""
config.py — Configuration constants and extent-scaling utility.

Contains all user-facing configuration parameters for the polygon
skeletonisation and width analysis pipeline, plus the _apply_extent_scale()
function that converts %-of-extent values to world units after features are
loaded.
"""

# All distance/size parameters marked "% of extent" are percentages of the
# dataset bounding-box extent. _apply_extent_scale() converts them to world
# units after loading features.

# =============================================================================
# BASIC SETTINGS
# =============================================================================

INPUT_FILE        = r"w:\Dropbox (Personal)\Working\Fracture width analysis\demo.shp"   # .shp | .svg | .pdf | .jpg | .png | .tif | .tiff
OUTPUT_DIRECTORY  = r"w:\Dropbox (Personal)\Working\Fracture width analysis\demo_shp"   # all output is written here

SEPARATE_MULTIPOLYGONS = False    # True  – split MultiPolygon features into one feature per component
                                  #         (classic behaviour; each polygon is analysed independently)
                                  # False – keep all components of a compound feature together and
                                  #         trace a single connected skeleton across them.  A
                                  #         RASTER_BUFFER outward buffer is applied before
                                  #         rasterisation so that zero-width connections — points,
                                  #         finite-length coincident edges, or sections where the left
                                  #         and right walls of the trace lie exactly on top of each
                                  #         other — become rasterisable thin strips rather than
                                  #         invisible zero-area gaps.

PRESERVE_TOPOLOGY  = True         # extend branch endpoints that abut another feature's polygon
                                  # so they snap on to that feature's skeleton

IMAGE_THRESHOLD    = 235          # greyscale threshold for image parsing (0–255)

# =============================================================================
# INPUT PARSING
# =============================================================================

ARTBOARD_MIN_SIZE  = 0.50         # artboard detection: polygon must cover at least this fraction of
                                  # the page/canvas area to be considered an artboard rectangle.
                                  # Default 0.50 (50%). Decrease to catch smaller background frames;
                                  # set to 1.0 to disable size-based filtering entirely.

ARTBOARD_MIN_RECT  = 0.95         # artboard detection: polygon must fill at least this fraction of
                                  # its own bounding box to be considered rectangular.
                                  # Default 0.95. Decrease to also catch non-axis-aligned frames or
                                  # rectangles with heavily rounded corners.

# =============================================================================
# SKELETONISER SETTINGS
# =============================================================================

MINIMUM_FEATURE_SIZE = 0.05       # % of extent — features smaller than this are dropped before
                                  # processing (0 = keep all features).
                                  # For polygons: the bounding-box side length must exceed this value
                                  # (i.e. bbox area ≥ MINIMUM_FEATURE_SIZE²).
                                  # For line features: the arc length must exceed this value.

SKELETONISATION_METHOD = "auto"
                                  # "auto"                        – full automatic decision tree: geometry
                                  #                                 gate → directional attempt → Lee
                                  #                                 single_branch or multi_branch
                                  # "directional"                 – directional cross-sections only; no
                                  #                                 geometry gate and no Lee fallback.
                                  #                                 If validation fails (high curvature,
                                  #                                 closed loop, escape) a warning is
                                  #                                 printed but the directional skeleton is
                                  #                                 still output — the feature is never
                                  #                                 dropped.  Use
                                  #                                 "directional_and_single_branch" for a
                                  #                                 robust version with a Lee fallback.
                                  # "directional_and_single_branch" – same geometry gate + directional attempt
                                  #                                 as "auto", but the Lee fallback is always
                                  #                                 single_branch (multi_branch is never the
                                  #                                 end-point).
                                  # "single_or_multi_branch"      – Lee thinning with stub pruning; decides
                                  #                                 single_branch or multi_branch via
                                  #                                 BRANCHING_THRESHOLD; no directional stage.
                                  # "single_branch"               – Lee thinning + graph diameter; single path.
                                  #                                 May stop short of round ends or fork at
                                  #                                 flat ends.
                                  # "multi_branch"                – full medial-axis skeleton with stub pruning;
                                  #                                 use when branching is intentional.

RASTER_RESOLUTION  = 0.0005       # % of extent — world units per pixel when rasterising (~20 px across
                                  # the image; auto-scaling via MAX/MIN_RASTER_PIXELS then refines this).
                                  # Ignored for raster images.

RASTER_BUFFER      = 0.0005       # % of extent — outward buffer applied to every polygon before
                                  # rasterisation, in world units (after _apply_extent_scale converts it).
                                  # Ensures thin features survive morphological thinning.  The buffer is
                                  # applied to the LOCAL rasterisation polygon only — feature.polygon
                                  # (used for width measurement and directional skeletonisation) is
                                  # never modified.

MAX_RASTER_PIXELS  = 25_000_000   # pixel budget per polygon; resolution is auto-scaled if exceeded

MIN_RASTER_PIXELS  = 2_500_000    # minimum pixel count per polygon; resolution is refined if below this (0 = off)

MIN_BRANCH_PIXELS  = 5            # a branch is a stub if it has fewer than this many pixels …

MIN_BRANCH_PERCENT = 0.5          # … or fewer than this % of total skeleton pixels (0 = off)

# --- Skeletonisation method thresholds (auto / directional modes) -----------

CURVATURE_THRESHOLD = 1.3         # "directional", "directional_and_single_branch", and "auto" modes:
                                  # if the Lee skeleton diameter path has sinuosity (arc / chord) above
                                  # this value the feature is treated as strongly curved and the
                                  # directional method is skipped.
                                  # Set to 0 to always attempt directional with no curvature gate.

SOLIDITY_THRESHOLD     = 0.2      # "auto" and "directional_and_single_branch" modes: polygon area /
                                  # convex-hull area; below this the feature is non-convex enough to
                                  # skip directional and use Lee

ASPECT_RATIO_THRESHOLD = 3.0      # "auto" and "directional_and_single_branch" modes: long / short
                                  # axis of minimum rotated bounding rectangle; below this the feature
                                  # is too compact for directional

ESCAPE_THRESHOLD       = 0.10     # "auto" and "directional_and_single_branch" modes: fraction of
                                  # interior directional-skeleton vertices that may lie outside the
                                  # polygon before falling back to Lee

BRANCHING_THRESHOLD    = 0.15     # "auto" mode: fraction of post-pruning skeleton pixels that lie
                                  # off the main (diameter) branch; above this → multi_branch

# =============================================================================
# POST-PROCESSING
# =============================================================================

SAMPLING_INTERVAL    = 0.05       # % of extent — spacing between width measurements along the centreline

MAX_WIDTH_RAY_DISTANCE = 0.1      # % of extent — maximum length of the perpendicular rays fired from
                                  # the centreline to locate fracture walls (width measurement).
                                  # Set to 0 to use auto (2 × polygon bounding-box diagonal), which
                                  # works well for closed polygons but may give very large values for
                                  # open or branching features.

SMOOTHING            = 0.01       # % of extent — Gaussian sigma for centreline smoothing (0 = none)

RDP_EPSILON          = 0.01       # % of extent — Ramer–Douglas–Peucker epsilon: points within this
                                  # distance of the simplified line are discarded

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

EXPORT_SKELETON_OVERLAY = True    # save a skeleton overlay plot for every feature
EXPORT_PROFILE_PLOT     = False   # save width-profile graphs
EXPORT_PROFILE_DATA     = False   # save per-branch width CSVs
EXPORT_PROFILE_FFT      = False   # save tortuosity FFT (fast Fourier transform) spectrum plot
EXPORT_RAW_TRACES       = False   # save an additional skeleton_raw.svg using the full-density
                                  # smoothed centreline (Branch.centerline) instead of the
                                  # decimated output coords (Branch.output_coords).  Useful for
                                  # inspecting the pre-simplification path geometry.

OUTPUT_SIZE        = 800          # display size (pixels) of the longer SVG dimension in skeleton.svg

OUTPUT_RESOLUTION    = 0.01       # % of extent — minimum vertex spacing in skeleton.svg (0 = keep all)

# =============================================================================
# ADVANCED
# =============================================================================

N_WORKERS          = None         # parallel worker threads; None = all logical CPU cores

# =============================================================================
# EXTENT SCALING
# =============================================================================

def _apply_extent_scale(features):
    """
    Convert all % config parameters to world-unit values by scaling
    against the combined bounding box of all loaded features.
    Must be called after features are loaded, before any processing.
    """
    global SAMPLING_INTERVAL, SMOOTHING, OUTPUT_RESOLUTION, RDP_EPSILON
    global MINIMUM_FEATURE_SIZE, RASTER_RESOLUTION, RASTER_BUFFER, MAX_WIDTH_RAY_DISTANCE

    if not features:
        return

    # Compute combined bounding box
    all_bounds = [f.polygon.bounds for f in features if hasattr(f.polygon, 'bounds')]
    if not all_bounds:
        return
    xs = [b[0] for b in all_bounds] + [b[2] for b in all_bounds]
    ys = [b[1] for b in all_bounds] + [b[3] for b in all_bounds]
    total_W = max(xs) - min(xs)
    total_H = max(ys) - min(ys)
    extent = max(total_W, total_H)
    if extent <= 0:
        return

    scale = extent / 100.0   # 1% of extent

    SAMPLING_INTERVAL      *= scale
    SMOOTHING              *= scale
    OUTPUT_RESOLUTION      *= scale
    RDP_EPSILON            *= scale
    MINIMUM_FEATURE_SIZE   *= scale
    RASTER_RESOLUTION      *= scale
    RASTER_BUFFER          *= scale
    MAX_WIDTH_RAY_DISTANCE *= scale

    print(f"  [config] Dataset extent: {total_W:.6g} × {total_H:.6g} (scale={extent:.6g})")
    print(f"  [config] RASTER_RESOLUTION={RASTER_RESOLUTION:.6g}, SAMPLING_INTERVAL={SAMPLING_INTERVAL:.6g}, "
          f"SMOOTHING={SMOOTHING:.6g}, RDP_EPSILON={RDP_EPSILON:.6g}")
    print(f"  [config] MINIMUM_FEATURE_SIZE={MINIMUM_FEATURE_SIZE:.6g}, RASTER_BUFFER={RASTER_BUFFER:.6g}, "
          f"MAX_WIDTH_RAY_DISTANCE={MAX_WIDTH_RAY_DISTANCE:.6g}")
