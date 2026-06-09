"""
exports.py — File-output functions for the skeletonisation pipeline.

Writes per-branch profile CSVs, Matplotlib SVG plots (width profile,
tortuosity FFT, skeleton overlay), and the FracPaQ-compatible skeleton SVG.
"""

import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt

# --- optional dependencies (warn but don't crash at import time) --------------

try:    # scipy
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# -----------------------------------------------------------------------------

import config
from config import (
    RASTER_RESOLUTION, SMOOTHING, SKELETONISATION_METHOD,
    SOLIDITY_THRESHOLD, ASPECT_RATIO_THRESHOLD, CURVATURE_THRESHOLD,
    ESCAPE_THRESHOLD, BRANCHING_THRESHOLD,
    EXPORT_SKELETON_OVERLAY, EXPORT_PROFILE_PLOT, EXPORT_PROFILE_DATA,
    EXPORT_PROFILE_FFT, OUTPUT_SIZE,
)
from skeletonisation import _is_line_feature, _pixels_to_world


# =============================================================================
# HELPERS
# =============================================================================

def _safe_filename(feature_id):
    """Return *feature_id* sanitised for use as a filename component.

    Characters outside ``[A-Za-z0-9_-]`` are replaced with underscores to
    avoid invalid paths or accidental writes to unexpected locations.

    Args:
        feature_id (str | int): The raw feature identifier.

    Returns:
        str: A filesystem-safe version of the identifier.
    """
    return re.sub(r"[^\w\-]", "_", str(feature_id))


# =============================================================================
# CSV EXPORT
# =============================================================================

def _export_individual_csvs(feature, output_dir):
    """
    Write one CSV file per branch containing its full width-profile data.

    Each row in a profile CSV represents one valid sample point along the
    branch centreline where both perpendicular rays successfully intersected
    the polygon boundary. Sample points stored as ``None`` in the profile
    (where one or both rays missed the boundary) are silently skipped so
    that the output contains only rows with complete measurements.

    The ``sample_index`` column reflects the position of the row within the
    written output, not the original index in ``branch["profile"]``, so
    indices are always contiguous starting from zero even when ``None``
    entries are present.

    File naming follows the pattern::

        feature_{safe_id}_branch_{branch_id}_profile.csv

    The feature ID is sanitised before use in the filename: any character
    that is not alphanumeric, a hyphen, or an underscore is replaced with an
    underscore to ensure the path is valid on all major operating systems.

    Args:
        feature (Feature): Feature dataclass instance containing:

            - ``id`` (*str | int*): Feature identifier, used in the output
              filename.
            - ``branches`` (*list[Branch]*): Branch instances, each with:

              - ``id`` (*int*): Branch identifier, used in the filename.
              - ``profile`` (*list[dict | None]*): Per-sample measurement
                dicts as produced by :func:`_find_partial_thickness`. Each
                non-``None`` entry must contain ``"dist"``, ``"x"``, ``"y"``,
                ``"side_a"``, ``"side_b"``, and ``"width"`` keys.

        output_dir (str | os.PathLike): Directory in which to create the CSV
            files. Must already exist.

    Returns:
        list[str]: Absolute paths of all CSV files written, one per branch,
        in branch order. Returns an empty list if the feature has no branches.

    CSV columns:
        - ``sample_index``: Zero-based index of the measurement row within
          this file (``None`` profile entries are excluded from the count).
        - ``distance``: Cumulative arc-length from the first sample point to
          this one, in world units (4 decimal places).
        - ``x``, ``y``: World coordinates of the sample point (6 d.p.).
        - ``width_side_a``: Partial thickness in the +normal direction (6 d.p.).
        - ``width_side_b``: Partial thickness in the −normal direction (6 d.p.).
        - ``total_width``: Sum of the two partial thicknesses (6 d.p.).
    """
    # Sanitise the feature ID for safe use as a filename component. The regex
    # replaces any character outside [A-Za-z0-9_-] with an underscore so that
    # IDs containing spaces, slashes, or other special characters do not create
    # invalid paths or silently write to unexpected locations.
    # Create the output directory here (not just at program start) so that the
    # directory still exists even if it was deleted during a long processing run.
    os.makedirs(output_dir, exist_ok=True)
    safe_fid = _safe_filename(feature.id)
    paths    = []

    for branch in feature.branches:
        path = os.path.join(
            output_dir,
            f"feature_{safe_fid}_branch_{branch.id}_profile.csv",
        )

        # newline="" is required by the csv module on all platforms: the module
        # handles its own line termination and passing newline="" prevents the
        # file object from applying an additional \r on Windows, which would
        # produce double carriage returns in the output.
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_index", "distance",
                "x", "y", "width_side_a", "width_side_b", "total_width",
                "orientation_deg",
            ])

            # sample_idx counts only the rows actually written, not positions
            # in the profile list. This keeps the index contiguous (0, 1, 2, …)
            # even when None entries interrupt the profile sequence.
            sample_idx = 0
            for pt in branch.profile:
                if pt is None:
                    # Ray casting failed at this sample point; skip it rather
                    # than writing a row with empty or placeholder values that
                    # could be misinterpreted as real measurements.
                    continue

                writer.writerow([
                    sample_idx,
                    f"{pt['dist']:.6f}",              # 6 d.p. to match x, y precision
                    f"{pt['x']:.6f}",                 # 6 d.p. for world coordinates to
                    f"{pt['y']:.6f}",                 # preserve sub-pixel precision
                    f"{pt['side_a']:.6f}",
                    f"{pt['side_b']:.6f}",
                    f"{pt['width']:.6f}",
                    f"{pt['orientation_deg']:.4f}",   # 4 d.p.; [0°, 180°) undirected
                ])
                sample_idx += 1

        paths.append(path)

    return paths

def _export_summary_csv(features, output_dir):
    """
    Write a single summary CSV with one row per feature, containing all
    computed width statistics and geometric properties.

    The column order is fixed and matches the order used by FracPaQ and
    downstream analysis scripts. Missing values (features where width
    measurement failed or geometric properties could not be computed) are
    written as empty strings rather than ``None`` or ``NaN``, which is the
    convention expected by downstream tools.

    ``csv.DictWriter`` is used so that the column-to-value mapping is
    explicit and robust to changes in the order of keys in
    ``feature.stats``. Any key present in ``feature.stats`` but absent
    from ``cols`` is silently ignored; any key present in ``cols`` but absent
    from ``feature.stats`` is written as an empty string via the
    ``s.get(k, "")`` fallback.

    Args:
        features (list[Feature]): Feature dataclass instances, each with a
            ``stats`` field as produced by :func:`_calculate_statistics`.
            Features whose ``stats`` field is ``None`` are written as a row
            of empty strings.
        output_dir (str | os.PathLike): Directory in which to create
            ``summary.csv``. Must already exist.

    Returns:
        str: Absolute path to the written ``summary.csv`` file.

    CSV columns:
        **Width statistics**

        - ``feature_id``: Feature identifier string.
        - ``n_branches``: Number of skeleton branches.
        - ``n_samples``: Number of valid width measurements.
        - ``average_thickness``: Mean total width across all sample points.
        - ``minimum_thickness``: Minimum total width.
        - ``maximum_thickness``: Maximum total width.
        - ``roughness_side_a``: Standard deviation of partial thickness, side A.
        - ``roughness_side_b``: Standard deviation of partial thickness, side B.
        - ``roughness``: Mean of ``roughness_side_a`` and ``roughness_side_b``.

        **Geometric properties**

        - ``orientation_deg``: Long-axis angle from +x, normalised to [0, 180).
        - ``long_axis_length``: PCA extent along the principal axis.
        - ``path_length``: Total arc length of all branch centrelines.
        - ``tortuosity``: ``path_length / chord_length``.
        - ``aspect_ratio_path``: ``path_length / average_thickness``.
        - ``aspect_ratio_long_axis``: ``long_axis_length / average_thickness``.

        **Tortuosity FFT (fast Fourier transform)**

        - ``fft_peak_wavelength_1/2/3``: Three dominant lateral-deviation
          wavelengths in world units, sorted by descending spectral magnitude.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.csv")

    # The column list defines both the header order and the set of keys
    # extracted from each feature's stats dict. Keeping it as an explicit
    # ordered list (rather than inferring columns from the first feature's
    # stats dict) ensures a consistent column order even if stats dicts from
    # different features have different key sets due to partial failures.
    cols = [
        "feature_id", "n_branches", "n_samples",
        "average_thickness", "minimum_thickness", "maximum_thickness",
        "roughness_side_a", "roughness_side_b", "roughness",
        "orientation_deg",
        "long_axis_length", "path_length",
        "tortuosity",
        "aspect_ratio_path", "aspect_ratio_long_axis",
        "fft_peak_wavelength_1", "fft_peak_wavelength_2", "fft_peak_wavelength_3",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        # The dict comprehension {k: s.get(k, "") for k in cols} only produces
        # keys from 'cols', so no extra keys ever reach the writer and internal
        # fields (e.g. fft_data) never appear in the CSV.
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for feat in features:
            # Fall back to an empty dict if stats is None (e.g. the feature
            # failed during skeletonisation and was never processed). The
            # s.get(k, "") comprehension then writes an empty string for every
            # column, producing a clearly-incomplete row that is still parseable
            # by downstream tools rather than omitting the feature entirely.
            s = feat.stats or {}
            writer.writerow({k: s.get(k, "") for k in cols})

    return path

# =============================================================================
# GRAPHS & PLOTS EXPORT
# =============================================================================
def _plot_width_profile(feature, output_dir):
    """
    Save a width-profile SVG graph for each branch of a feature.

    Each plot shows how the polygon width varies along the branch centreline,
    with the two-sided nature of the measurement (side A and side B) made
    visually explicit by mirroring side B below the zero axis.

    **Layout** (bottom to top in z-order):

    - *Zero reference line*: a faint horizontal line at ``y = 0`` so the
      A/B asymmetry reads immediately without needing to judge distances
      from the axis spine.
    - *Side B (negated)*: dotted line, plotted as ``-side_b`` so it appears
      below zero. This convention makes the plot bilaterally symmetric for a
      feature with equal wall distances, and highlights asymmetry when the
      walls are unequal.
    - *Side A*: dashed line, positive values above zero.
    - *Total width*: solid line (``side_a + side_b``), always above zero.
    - *Average width reference*: thin dash-dot black line if the feature has
      a computed ``average_thickness`` statistic.

    All three data lines share a single colour taken from the default
    Matplotlib colour cycle, distinguished only by line style, to avoid
    colour conflicts between the three series on the same axes.

    When no valid profile measurements exist for a branch, a centred
    "No valid measurements" text label is drawn instead so the file is still
    created and the absence of data is explicit.

    File names follow the pattern::

        feature_{safe_id}_branch_{branch_id}_profile.svg

    Args:
        feature (Feature): Feature dataclass instance containing:

            - ``id`` (*str | int*): Used in the output filename and plot
              title.
            - ``branches`` (*list[Branch]*): Branch instances, each with:

              - ``id`` (*int*): Used in the filename and title.
              - ``profile`` (*list[dict | None]*): Per-sample measurement
                dicts as produced by :func:`_find_partial_thickness`.

            - ``stats`` (*dict*, optional): Used to read
              ``"average_thickness"`` for the reference line. If absent,
              the reference line is omitted.

        output_dir (str | os.PathLike): Directory in which to save the SVG
            files. Must already exist.

    Returns:
        list[str]: Paths of all SVG files written, one per branch, in branch
        order. Returns an empty list if the feature has no branches.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Sanitise the feature ID for use in filenames. Characters outside
    # [A-Za-z0-9_-] are replaced with underscores to avoid invalid paths.
    safe_fid = _safe_filename(feature.id)
    paths    = []

    # Compute the feature-level average width directly from the profile data so
    # that the reference line is always derived from this feature's own
    # measurements, independent of feature.stats.
    all_valid_widths = [pt["width"]
                        for b in feature.branches
                        for pt in b.profile
                        if pt is not None]
    feature_avg = float(np.mean(all_valid_widths)) if all_valid_widths else None

    for branch in feature.branches:
        # Filter out None profile entries upfront. Working with a clean list
        # simplifies all the list comprehensions below and avoids repeated
        # None checks inside the plotting block.
        valid = [pt for pt in branch.profile if pt is not None]

        path = os.path.join(
            output_dir,
            f"feature_{safe_fid}_branch_{branch.id}_profile.svg",
        )
        fig, ax = plt.subplots(figsize=(10, 4))

        if valid:
            dists  = [pt["dist"]   for pt in valid]
            widths = [pt["width"]  for pt in valid]
            a_vals = [pt["side_a"] for pt in valid]
            # Side B is negated so it plots below zero. This mirrors the
            # physical geometry: side A is the "above" wall and side B is the
            # "below" wall of the feature cross-section, and plotting them on
            # opposite sides of zero makes any asymmetry immediately legible.
            b_vals = [-pt["side_b"] for pt in valid]

            # Pull a single colour from the default cycle. Using index [0]
            # always gives the first colour, which is intentional — each
            # branch gets its own figure, so there is only ever one set of
            # three lines per axes, and all three should share a colour to
            # emphasise that they come from the same measurement series.
            col = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

            ax.plot(dists, widths, "-",  color=col, linewidth=1.8,
                    label="Total width")
            ax.plot(dists, a_vals, "--", color=col, linewidth=1.0, alpha=0.85,
                    label="Side A")
            ax.plot(dists, b_vals, ":",  color=col, linewidth=1.0, alpha=0.85,
                    label="Side B (negated)")

            # Horizontal zero line. Using axhline rather than a manually
            # computed zero list keeps the line aligned with the axis
            # regardless of the x data range, and zorder=1 places it behind
            # all data lines so it does not obscure the measurements.
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.4, zorder=1)

            # Average width reference line, computed from this feature's own
            # profile data (all branches combined).  Using the raw measurements
            # rather than feature.stats ensures the value is always per-feature
            # and up-to-date even when the export runs concurrently with
            # statistics computation.
            if feature_avg is not None:
                ax.axhline(feature_avg, color="black", linestyle="-.",
                           linewidth=0.9, alpha=0.6,
                           label=f"Feature avg = {feature_avg:.2f}")
        else:
            # No valid measurements: draw a centred placeholder text so the
            # file exists and its emptiness is unambiguous. An empty SVG
            # would silently mislead a viewer into thinking the export failed.
            ax.text(0.5, 0.5, "No valid measurements", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=12)

        ax.set_xlabel("Distance along centreline (world units)")
        ax.set_ylabel("Width (world units)")
        ax.set_title(
            f"Width profile — Feature {feature.id}  Branch {branch.id}"
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.45)
        fig.tight_layout()
        fig.savefig(path, format="svg", bbox_inches="tight")
        # Explicitly close the figure to release its memory. Without this,
        # Matplotlib accumulates all figures in the process heap, which
        # exhausts memory when processing large feature sets in a single run.
        plt.close(fig)
        paths.append(path)

    return paths

def _plot_tortuosity_fft(feature, output_dir):
    """
    Save an FFT (fast Fourier transform) spectrum plot of the lateral-deviation (tortuosity) signal
    for a feature as an SVG file.

    The lateral-deviation signal captures how far the centreline wanders
    sideways relative to its overall chord direction at each sample point
    (see :func:`_tortuosity_fft_peaks` for the construction details). The
    FFT of this signal reveals which spatial wavelengths dominate the
    feature's sinuosity — short wavelengths indicate tight, high-frequency
    meanders while long wavelengths indicate broad, low-frequency curves.

    **Plot layout**:

    - *X-axis*: wavelength in world units on a logarithmic scale. Log scale
      is used because the dominant wavelengths of fracture traces typically
      span several orders of magnitude (from sub-pixel ripple to feature-length
      curves), and a linear scale would compress most of the spectrum into an
      unreadable cluster near zero.
    - *Y-axis*: FFT magnitude (linear scale).
    - *Spectrum line*: plotted as wavelength = 1 / frequency, with the DC
      component (frequency = 0) excluded because it represents the mean
      lateral offset rather than any oscillatory wavelength.
    - *Peak annotations*: vertical dashed lines at the three dominant
      wavelengths from ``feature.fft_data["peaks"]``, each labelled with its
      wavelength value. Three distinct colours are used to make overlapping
      peaks individually identifiable.

    When insufficient data exists for the FFT (fewer than four sample points),
    a centred "Insufficient data" placeholder text is drawn instead.

    The output filename follows the pattern::

        feature_{safe_id}_tortuosity_fft.svg

    Args:
        feature (Feature): Feature dataclass instance containing:

            - ``id`` (*str | int*): Used in the output filename and title.
            - ``fft_data`` (*dict*): FFT data as written by
              :func:`_calculate_statistics`, with keys ``"freqs"``,
              ``"magnitudes"``, and ``"peaks"``.

        output_dir (str | os.PathLike): Directory in which to save the SVG.
            Must already exist.

    Returns:
        str: Path to the written SVG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_fid = _safe_filename(feature.id)
    path     = os.path.join(
        output_dir,
        f"feature_{safe_fid}_tortuosity_fft.svg",
    )

    # Retrieve pre-computed FFT data from the feature. These were
    # written by _calculate_statistics → _tortuosity_fft_peaks and contain
    # the full frequency/magnitude arrays plus the top-n peak wavelengths.
    fft_info   = feature.fft_data or {}
    freqs      = fft_info.get("freqs", [])
    magnitudes = fft_info.get("magnitudes", [])
    peaks      = fft_info.get("peaks", [])   # list[float | None], length == n_peaks

    fig, ax = plt.subplots(figsize=(10, 4))

    if len(freqs) > 1:
        freqs_arr = np.array(freqs)
        mags_arr  = np.array(magnitudes)

        # Exclude the DC component (frequency == 0) by masking non-positive
        # frequencies. Converting to wavelength (1/f) also requires f > 0
        # to avoid a division-by-zero or a negative wavelength.
        valid       = freqs_arr > 0
        wavelengths = 1.0 / freqs_arr[valid]
        ax.plot(wavelengths, mags_arr[valid],
                color="steelblue", linewidth=1.0, zorder=2)

        # Annotate peak wavelengths with vertical dashed lines. Three fixed
        # colours are used rather than the default cycle so that peak 1 is
        # always red, peak 2 always teal, and peak 3 always amber — making
        # the peaks identifiable at a glance when multiple feature plots are
        # compared side-by-side.
        peak_colors = ["#e63946", "#2a9d8f", "#e9c46a"]
        for i, wl in enumerate(peaks):
            if wl is not None:
                ax.axvline(wl,
                           color=peak_colors[i % len(peak_colors)],
                           linewidth=1.2, linestyle="--",
                           label=f"Peak {i + 1}:  λ = {wl:.3f}",
                           zorder=3)

        # Log scale on the x-axis compresses the wide wavelength range into
        # a readable spread. Without it, the high-magnitude low-frequency
        # peaks would be squeezed against the left edge of the plot.
        ax.set_xscale("log")
        ax.set_xlabel("Wavelength (world units)")
    else:
        # Fewer than two frequency bins means the centreline had too few
        # sample points for a meaningful FFT (fast Fourier transform). Draw a placeholder rather than
        # leaving empty axes that might be mistaken for a flat spectrum.
        ax.text(0.5, 0.5, "Insufficient data for FFT",
                transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=12)
        ax.set_xlabel("Wavelength (world units)")

    ax.set_ylabel("FFT (fast Fourier transform) magnitude")
    ax.set_title(f"Tortuosity FFT spectrum — Feature {feature.id}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path

def _plot_individual_feature_skeleton(feature, output_dir):
    """
    Save a skeleton overlay plot for a single feature as an SVG file.

    The plot stacks multiple rendering layers to show the complete
    skeletonisation pipeline — from the raw rasterised footprint through to
    the final smoothed centreline — so that algorithmic choices (method
    selection, stub pruning, curvature fallback) can be visually verified.

    **Rendered layers (bottom to top in z-order)**:

    1. *Rasterised polygon footprint*: the boolean grid produced by
       :func:`_rasterise_polygon`, shown as a very faint (20% opacity) grey
       image. Provides spatial context for the pixel-coordinate branches.
    2. *Source polygon*: the original Shapely geometry as a 50%-opacity blue
       fill with outline. Holes (interior rings) are whited out and outlined
       with a dashed blue line.
    3. *Discarded stub pixels* (single/multi-branch modes only): raw skeleton
       branches that were entirely absent from the merged branch set, drawn
       as black dots and a thin black line. Black was chosen so stubs stand
       out clearly against both the grey raster and the blue polygon fill.
    4. *Kept branch pixels*: the merged/diameter branch pixels in tab10
       colours, at 50% opacity so the centreline on top remains legible.
       A thin line connects the pixels in skeleton-walk order to show
       directionality.
    5. *Smoothed centreline*: Gaussian smoothing applied to the kept branch
       pixels in world space, drawn as a solid 2.0-width line. This is the
       dense, pre-resampling centreline — every pixel position is included,
       so it follows the skeleton exactly before ``SAMPLING_INTERVAL``
       resampling is applied.

    In **directional mode**, layers 3–5 are replaced with:

    - The Lee skeleton pixels as a light grey reference scatter (so the user
      can compare what isotropic thinning would have produced with what the
      PCA cross-section method actually generated).
    - The directional centreline points and their smoothed curve.

    The plot title encodes the skeletonisation method, branch count,
    sinuosity (for directional and curved-fallback modes), number of pruned
    stubs or discarded side-arm pixels, and the effective raster resolution.

    File names follow the pattern::

        feature_{safe_id}_skeleton_overlay.svg

    Args:
        feature (Feature): Feature dataclass instance containing:

            - ``id`` (*str | int*): Used in the filename and title.
            - ``polygon`` (*shapely.geometry.Polygon*): The source geometry.
            - ``skeleton_overlay_data`` (*dict*): Overlay data written by
              :func:`_dispatch_skeletoniser`, including ``"raster_grid"``,
              ``"skeleton_grid"``, ``"raw_branches"``, ``"merged_branches"``,
              ``"directional_world"`` (directional mode only),
              ``"skeleton_method"``, ``"sinuosity"``, ``"x_min"``, ``"y_min"``,
              and ``"resolution"``. Any missing keys fall back to safe
              defaults (empty lists, zero origins, ``RASTER_RESOLUTION``).
            - ``flip_y`` (*bool*): If ``True``, the y-axis is inverted after
              plotting so geographic north faces upward.

        output_dir (str | os.PathLike): Directory in which to save the SVG.
            Must already exist.

    Returns:
        str: Path to the written SVG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_id = re.sub(r"[^\w\-]", "_", str(feature.id))
    path    = os.path.join(output_dir, f"feature_{safe_id}_skeleton_overlay.svg")

    polygon    = feature.polygon
    ovl        = feature.skeleton_overlay_data or {}
    skeleton   = ovl.get("skeleton_grid")
    raster     = ovl.get("raster_grid")
    raw_br     = ovl.get("raw_branches", [])
    merged_br  = ovl.get("merged_branches", [])
    x_min      = ovl.get("x_min", 0.0)
    y_min      = ovl.get("y_min", 0.0)
    resolution = ovl.get("resolution", config.RASTER_RESOLUTION)

    # Pre-compute a set of all pixel coordinates present in any merged branch.
    # This is used to classify raw branches as either "kept" (all their pixels
    # appear in a merged branch) or "stub" (none of their pixels appear), so
    # that stubs can be drawn in black for visual distinction.
    merged_px_sets = [set(map(tuple, px)) for px in merged_br]
    all_merged_px  = set().union(*merged_px_sets) if merged_px_sets else set()

    # tab10 provides 10 perceptually distinct colours; cycling through it with
    # `bid % 10` ensures that multi-branch features with many branches never
    # wrap around to ambiguous colour reuse in the legend.
    palette = [matplotlib.colormaps["tab10"](i) for i in range(10)]

    fig, ax = plt.subplots(figsize=(12, 9))

    # ── Layer 1: rasterised polygon footprint ─────────────────────────────────
    # Shown at 20% opacity so it doesn't obscure the polygon fill or branch
    # pixels above it, but provides a pixel-resolution reference frame that
    # makes the skeleton pixel positions physically interpretable.
    if raster is not None and raster.any():
        H, W   = raster.shape
        # extent maps the array's (row, col) pixel indices to world coordinates
        # using the grid origin and resolution from the rasterisation step.
        # origin="lower" ensures row 0 is at the bottom of the image, matching
        # the mathematical (x increases right, y increases up) convention.
        extent = (
            x_min,
            x_min + W * resolution,
            y_min,
            y_min + H * resolution,
        )
        ax.imshow(raster, origin="lower", extent=extent,
                  cmap="Greys", alpha=0.20, interpolation="nearest", zorder=1)

    skeleton_method = ovl.get("skeleton_method", config.SKELETONISATION_METHOD)

    # ── Layer 2: source polygon — skip if the feature is a line input ─────────
    # Handles both Polygon and MultiPolygon by normalising to a flat part list.
    # Interior rings (holes) are whited out after filling the exterior so that
    # the hole area appears as white rather than inheriting the blue fill.
    if not _is_line_feature(feature):
        parts = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        poly_label_done = False
        for part in parts:
            if part.is_empty or not hasattr(part, "exterior"):
                continue
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, alpha=0.50, color="tab:blue", zorder=2)
            ax.plot(xs, ys, color="tab:blue", linewidth=1.4, zorder=2,
                    # Only add the legend label on the first part to avoid
                    # duplicate "Source polygon" entries in the legend.
                    label=("Source polygon" if not poly_label_done else None))
            poly_label_done = True
            for interior in part.interiors:
                ixs, iys = interior.xy
                # White fill punches the hole out of the blue exterior fill.
                ax.fill(ixs, iys, color="white", zorder=2)
                ax.plot(ixs, iys, color="tab:blue", linewidth=1.0,
                        linestyle="--", zorder=2)

    if skeleton_method.startswith("directional"):
        # ── Directional mode: layers 3–5 ─────────────────────────────────────

        # Layer 3: Lee skeleton as a light grey reference scatter.
        # Plotted at low prominence (silver, s=3, alpha=0.6) so it does not
        # compete visually with the directional centreline above it, but
        # gives the user a point of comparison with what isotropic thinning
        # would have produced.
        n_skel_px = 0
        if skeleton is not None:
            skeleton_rows, skeleton_cols = np.where(skeleton)
            n_skel_px = len(skeleton_rows)
            if n_skel_px:
                skel_wx = x_min + skeleton_cols * resolution
                skel_wy = y_min + skeleton_rows * resolution
                ax.scatter(skel_wx, skel_wy, color="silver", s=3, alpha=0.6,
                           zorder=3, label=f"Lee skeleton ({n_skel_px} px, reference)")

        # Layers 4–5: directional centreline points and smoothed curve.
        dir_world = ovl.get("directional_world", [])
        if len(dir_world) >= 2:
            col = palette[0]
            dwx = [p[0] for p in dir_world]
            dwy = [p[1] for p in dir_world]

            # Scatter the raw midpoint positions (before smoothing) so the
            # user can see exactly where the cross-section intersections fell.
            ax.scatter(dwx, dwy, color=col, s=5, alpha=0.5, zorder=4,
                       label=f"Directional skeleton ({len(dir_world)} pts)")

            # Apply the same resolution-scaled Gaussian smoothing used by _smooth_coords.
            xs_d = np.array(dwx)
            ys_d = np.array(dwy)
            if HAS_SCIPY and config.SMOOTHING > 0:
                # Convert world-unit sigma to pixel-index sigma, consistent
                # with _smooth_coords, so the skeleton overlay matches the actual
                # smoothed centreline stored in branch.centerline.
                sigma_px = config.SMOOTHING / resolution if resolution > 0 else config.SMOOTHING
                xs_d = gaussian_filter1d(xs_d, sigma=sigma_px)
                ys_d = gaussian_filter1d(ys_d, sigma=sigma_px)
            ax.plot(xs_d, ys_d, "-", color=col, linewidth=2.0, alpha=0.9,
                    zorder=5, label="Branch 0 centreline")

    else:
        # ── Single_branch / multi_branch mode: layers 3–5 ────────────────────

        # Layer 3: discarded stub pixels in black.
        # A raw branch is classified as a stub if none of its pixel coordinates
        # appear in any merged branch. Drawing stubs in black (rather than a
        # dim version of a branch colour) ensures they are clearly visible
        # against both the grey raster footprint and the blue polygon fill.
        stub_px_drawn = False
        for raw_pixels in raw_br:
            if not any(tuple(p) in all_merged_px for p in raw_pixels):
                world = _pixels_to_world(raw_pixels, x_min, y_min, resolution)
                wx = [p[0] for p in world]
                wy = [p[1] for p in world]
                ax.scatter(wx, wy, color="black", s=4, alpha=0.9, zorder=3,
                           # Only add the stub label once to keep the legend
                           # compact, even when many stubs are present.
                           label=("Discarded stub pixels" if not stub_px_drawn else None))
                if len(world) >= 2:
                    ax.plot(wx, wy, "-", color="black", linewidth=0.7,
                            alpha=0.9, zorder=3)
                stub_px_drawn = True

        # Layers 4–5: kept branch pixels and smoothed centrelines.
        # Each branch gets a distinct tab10 colour. The scatter (alpha=0.5,
        # zorder=4) sits below the centreline (alpha=0.9, zorder=5) so the
        # smooth curve is readable even in dense pixel regions.
        n_skel_px = 0
        for bid, merged_pixels in enumerate(merged_br):
            col   = palette[bid % len(palette)]
            world = _pixels_to_world(merged_pixels, x_min, y_min, resolution)
            wx    = [p[0] for p in world]
            wy    = [p[1] for p in world]
            n_skel_px += len(merged_pixels)

            ax.scatter(wx, wy, color=col, s=5, alpha=0.5, zorder=4,
                       label=f"Branch {bid} pixels ({len(merged_pixels)} px)")
            if len(world) >= 2:
                # Thin semi-transparent line connecting pixels in walk order.
                # This reveals the traversal direction of the skeleton trace
                # and makes gaps or kinks in the pixel path visible.
                ax.plot(wx, wy, "-", color=col, linewidth=0.9, alpha=0.5, zorder=4)

            if len(world) >= 2:
                xs_d = np.array(wx)
                ys_d = np.array(wy)
                if HAS_SCIPY and config.SMOOTHING > 0:
                    xs_d = gaussian_filter1d(xs_d, sigma=config.SMOOTHING)
                    ys_d = gaussian_filter1d(ys_d, sigma=config.SMOOTHING)
                # Thick, opaque centreline drawn on top of the pixel scatter.
                # This is the pre-resampling centreline: every skeleton pixel
                # contributes a point, so it traces the skeleton more closely
                # than the SAMPLING_INTERVAL-resampled sample_points would.
                ax.plot(xs_d, ys_d, "-", color=col, linewidth=2.0, alpha=0.9,
                        zorder=5, label=f"Branch {bid} centreline")

    # Invert the y-axis after all drawing is complete so that the axis limits
    # are set from the actual data range before the inversion is applied,
    # preventing Matplotlib from re-expanding the limits on the next draw.
    if feature.flip_y:
        ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (world units)")
    ax.set_ylabel("y (world units)")

    # ── Title construction ─────────────────────────────────────────────────────
    # The title uses multiple lines so all decision-tree diagnostics fit without
    # truncation.  Each line covers one logical phase of the decision tree.
    sinuosity = ovl.get("sinuosity")
    auto_dec  = ovl.get("auto_decision") or {}

    # ── Line 1: feature / method summary ──────────────────────────────────────
    _is_directional_result = skeleton_method.startswith("directional")
    if skeleton_method == "line_input":
        n_branches = len(feature.branches)
    elif _is_directional_result:
        dir_world  = ovl.get("directional_world", [])
        n_branches = 1 if dir_world else 0
    else:
        n_branches = len(merged_br)

    n_stubs_removed   = len(raw_br) - len(merged_br)
    n_px_removed      = sum(len(b) for b in raw_br) - sum(len(b) for b in merged_br)

    line1 = (f"Skeleton — Feature {feature.id}  |  {skeleton_method}  |  "
             f"{n_branches} branch(es)  |  res={resolution:.6f}")

    title_lines = [line1]

    if auto_dec:
        # ── Line 2: geometry gate ──────────────────────────────────────────────
        sol     = auto_dec.get("solidity")
        ar      = auto_dec.get("aspect_ratio")
        n_holes = int(auto_dec.get("has_hole", False))   # 0 or 1 from bool

        geo_parts = []
        if sol is not None:
            arrow = "✓" if sol >= config.SOLIDITY_THRESHOLD else "✗"
            geo_parts.append(
                f"solidity={sol:.3f}{arrow}(thr≥{config.SOLIDITY_THRESHOLD})")
        if ar is not None:
            arrow = "✓" if ar >= config.ASPECT_RATIO_THRESHOLD else "✗"
            geo_parts.append(
                f"AR={ar:.2f}{arrow}(thr≥{config.ASPECT_RATIO_THRESHOLD})")
        geo_parts.append(f"holes={n_holes}")
        title_lines.append("Geo: " + "  |  ".join(geo_parts))

        # ── Line 3: directional gate (shown when it was attempted) ────────────
        dir_sin = auto_dec.get("sinuosity")
        is_loop = auto_dec.get("is_closed_loop")   # None if curvature gate fired first
        esc_len = auto_dec.get("escaped_length")
        int_len = auto_dec.get("interior_length")

        dir_parts = []
        if dir_sin is not None:
            arrow = "✗" if (config.CURVATURE_THRESHOLD > 0 and dir_sin > config.CURVATURE_THRESHOLD) else "✓"
            dir_parts.append(
                f"sinuosity={dir_sin:.2f}{arrow}(thr≤{config.CURVATURE_THRESHOLD})")
        if is_loop is not None:
            dir_parts.append(f"closed_loop={'Yes✗' if is_loop else 'No✓'}")
        if esc_len is not None and int_len is not None:
            frac     = esc_len / int_len if int_len > 1e-9 else 0.0
            esc_pct  = f"{frac:.1%}"
            arrow    = "✗" if frac > config.ESCAPE_THRESHOLD else "✓"
            dir_parts.append(
                f"escaped={esc_pct}{arrow}(thr≤{config.ESCAPE_THRESHOLD:.0%})")
        if dir_parts:
            geo_gate = auto_dec.get("geo_method", "")
            prefix   = "Dir ✓" if auto_dec.get("final_method") == "directional" else "Dir ✗"
            if geo_gate == "lee":
                prefix = "Dir skipped (geometry gate)"
            title_lines.append(prefix + ": " + "  |  ".join(dir_parts))

        # ── Line 4: Lee gate (shown when Lee path was taken) ──────────────────
        off_frac = auto_dec.get("off_main_fraction")
        if off_frac is not None:
            n_sp = auto_dec.get("n_stubs_pruned", n_stubs_removed)
            n_pp = auto_dec.get("n_pixels_pruned", n_px_removed)
            arrow = "→ multi" if off_frac > config.BRANCHING_THRESHOLD else "→ single"
            title_lines.append(
                f"Lee: off_main={off_frac:.1%}{arrow}(thr={config.BRANCHING_THRESHOLD:.0%})  |  "
                f"stubs pruned={n_sp} ({n_pp} px)")
    else:
        # Explicit (non-auto) mode — show sinuosity and pruning summary only.
        extra_parts = []
        if sinuosity is not None:
            extra_parts.append(f"sinuosity={sinuosity:.2f}")
        if not _is_directional_result:
            n_sp = n_stubs_removed
            n_pp = n_px_removed
            extra_parts.append(f"stubs pruned={n_sp} ({n_pp} px)")
        if extra_parts:
            title_lines.append("  |  ".join(extra_parts))

    ax.set_title("\n".join(title_lines), fontsize=8, loc="left")

    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    ax.grid(True, linestyle=":", alpha=0.40)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path

# =============================================================================
# SKELETON EXPORT  (FracPaQ-compatible)
# =============================================================================

def _export_skeleton_svg(features, output_dir, flip_y=False,
                         coords_attr="output_coords", filename="skeleton.svg"):
    """
    Write all skeleton centreline branches to a FracPaQ-compatible SVG file.

    The output SVG is formatted to satisfy the strict requirements of
    FracPaQ v2's SVG importer:

    - Branches are represented as ``<polyline>`` elements with a
      space-separated ``x1 y1 x2 y2 …`` coordinate string in the ``points``
      attribute.
    - Stroke styling is applied via a CSS class ``".sk"`` declared in a
      ``<defs>`` block rather than as inline ``stroke`` / ``fill`` attributes.
      FracPaQ's parser recognises the class-based form only.
    - The SVG uses the 1.1 namespace and a ``viewBox`` for sizing. Explicit
      ``width`` and ``height`` attributes are intentionally omitted because
      FracPaQ reads geometry from ``viewBox`` coordinates.
    - The XML declaration omits the ``standalone`` attribute, which some SVG
      parsers require to be absent.

    **Coordinate mapping**

    All centreline coordinates are in world space (the same units as the
    source polygon geometry). They are mapped to SVG pixel space by:

    1. Subtracting the world-space origin (``x_origin``, ``y_origin``), which
       includes a 2% margin on each side so no branch touches the SVG edge.
    2. Multiplying by a uniform scale factor chosen so the longer world-space
       dimension maps to 800 display pixels.
    3. Optionally inverting the y-axis when ``flip_y=True``, to convert from
       a geographic coordinate system (y increases northward) to SVG space
       (y increases downward).

    **Vertex reduction**

    By default (``coords_attr="output_coords"``) each branch's ``output_coords``
    attribute is used; these are pre-computed by :func:`_post_process_skeleton`
    via two decimation steps:

    1. Uniform arc-length resampling at ``OUTPUT_RESOLUTION`` world units,
       which removes overly dense vertices.
    2. Ramer–Douglas–Peucker simplification at ``RDP_EPSILON`` tolerance,
       which further reduces vertex count in low-curvature regions while
       preserving the visual shape of tight bends.

    When ``coords_attr="centerline"`` the full-density smoothed centreline is
    used instead (see ``EXPORT_RAW_TRACES``).

    Consecutive duplicate points that arise from coordinate rounding after
    the SVG-space conversion are removed before writing, because FracPaQ
    crashes on zero-length polyline segments.

    Args:
        features (list[Feature]): Feature dataclass instances as produced by
            :func:`_dispatch_skeletoniser`. The branch attribute named by
            ``coords_attr`` is used for polyline vertices; ``centerline`` is
            always used to compute the bounding box.
        output_dir (str | os.PathLike): Directory in which to create the SVG
            file. The directory must already exist.
        flip_y (bool): If ``True``, invert the y-axis so that world-space
            "up" (increasing y) maps to SVG "up" (decreasing SVG y). Set
            this when the source coordinates use a geographic convention
            (e.g. projected CRS). Defaults to ``False``, which preserves
            the SVG convention of y increasing downward.
        coords_attr (str): Name of the :class:`Branch` attribute to use for
            polyline vertices.  Use ``"output_coords"`` (default) for the
            decimated SVG export, or ``"centerline"`` for the raw smoothed
            trace (``EXPORT_RAW_TRACES`` mode).
        filename (str): Output filename within *output_dir*.  Defaults to
            ``"skeleton.svg"``; pass ``"skeleton_raw.svg"`` for the raw-trace
            export.

    Returns:
        str | None: The absolute path to the written SVG file, or ``None`` if
        no features contained any branches with at least two vertices (in which
        case a warning is printed and no file is created).
    """
    # ── Collect all centreline coordinates to compute the world bounding box ──
    # A single pass over all features and branches is necessary because the
    # scale factor and origin offset must be computed globally — using a
    # per-feature bounding box would give each branch a different scale and
    # make the output geometrically inconsistent.
    coords_iter = (
        (x, y)
        for feat in features
        for branch in feat.branches
        for x, y in branch.centerline
    )
    first = next(coords_iter, None)
    if first is None:
        print("  Warning: no skeleton branches to write.")
        return None

    xmin = xmax = first[0]
    ymin = ymax = first[1]
    for x, y in coords_iter:
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y

    # Add a margin so no branch vertex sits exactly on the SVG boundary.
    # 2% of the larger world-space dimension; for degenerate/point features,
    # fall back to 2% of whatever non-zero extent exists (or exactly 0 if both
    # dimensions are zero). No hard absolute minimum — a fixed world-unit floor
    # would dwarf the features when coordinates are in small units (e.g. degrees).
    world_extent = max(xmax - xmin, ymax - ymin)
    margin = world_extent * 0.02 if world_extent > 0 else 0.0

    # World-space dimensions of the padded bounding box.
    w_world  = (xmax - xmin) + 2 * margin
    h_world  = (ymax - ymin) + 2 * margin
    # Bottom-left corner of the padded world-space bounding box, used as the
    # coordinate origin when mapping world points to SVG pixel space.
    x_origin = xmin - margin
    y_origin = ymin - margin

    # Uniform scale: fit the longer dimension to OUTPUT_SIZE display pixels so the
    # SVG renders at a reasonable size in any viewer. The shorter dimension
    # scales proportionally, preserving aspect ratio.
    display_px = float(config.OUTPUT_SIZE)
    scale      = display_px / max(w_world, h_world) if max(w_world, h_world) > 0 else 1.0
    svg_w      = w_world * scale
    svg_h      = h_world * scale

    def to_svg(x, y):
        """
        Map a single world-space ``(x, y)`` coordinate to SVG pixel space.

        The mapping involves two steps:

        1. **Translation and scaling**: subtract the padded world origin
           (``x_origin``, ``y_origin``) and multiply by ``scale`` to convert
           from world units to SVG pixels.
        2. **Optional y-axis inversion** (when ``flip_y=True``): SVG places
           the origin at the top-left with y increasing downward, while
           geographic coordinate systems have y increasing upward. The
           inversion formula ``h_world - (y - y_origin)`` reflects the
           y-coordinate about the horizontal midline of the bounding box
           before scaling, so that north maps to the top of the SVG canvas.

        Both ``scale``, ``x_origin``, ``y_origin``, ``h_world``, and
        ``flip_y`` are captured from the enclosing scope.

        Args:
            x (float): World-space x-coordinate.
            y (float): World-space y-coordinate.

        Returns:
            tuple[float, float]: ``(sx, sy)`` coordinates in SVG pixel space,
            measured in pixels from the top-left corner of the viewBox.
        """
        sx = (x - x_origin) * scale
        # When flip_y is True, subtract the normalised y from the total world
        # height before scaling; this mirrors the y-axis so that larger world-y
        # values map to smaller SVG-y values (i.e. higher on the canvas).
        sy = (h_world - (y - y_origin)) * scale if flip_y else (y - y_origin) * scale
        return sx, sy

    # ── SVG document header ───────────────────────────────────────────────────
    # viewBox is set to the computed pixel dimensions; no explicit width/height
    # is emitted because FracPaQ reads geometry from viewBox coordinates only.
    # The .sk CSS class is declared here once and referenced by all polylines,
    # keeping the file smaller than per-element inline style attributes would.
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1"'
        f' viewBox="0 0 {svg_w:.6f} {svg_h:.6f}">',
        '  <defs>',
        '    <style>',
        '      .sk { fill: none; stroke: #000000; stroke-miterlimit: 10; }',
        '    </style>',
        '  </defs>',
    ]

    # ── Write one <polyline> per branch ───────────────────────────────────────
    # Use the branch attribute named by coords_attr (default "output_coords",
    # i.e. the uniform-spacing + RDP decimated list pre-computed by
    # _post_process_skeleton).  When coords_attr="centerline" the full-density
    # smoothed path is written instead (EXPORT_RAW_TRACES mode).
    features_written = set()
    for feat in features:
        for branch in feat.branches:
            cl = getattr(branch, coords_attr, None) or []
            if len(cl) < 2:
                continue

            # Convert all surviving world-space vertices to SVG pixel space.
            svg_pts = [to_svg(x, y) for x, y in cl]

            # Remove consecutive duplicate pixel coordinates that can arise
            # when two nearby world points round to the same 6-decimal SVG
            # value. FracPaQ raises an internal error on zero-length segments
            # (a polyline point that is identical to its predecessor), so
            # deduplication is a hard requirement, not a stylistic choice.
            deduped = [svg_pts[0]]
            for pt in svg_pts[1:]:
                if pt != deduped[-1]:
                    deduped.append(pt)

            # After deduplication a branch may have collapsed to a single
            # point (e.g. an extremely short branch whose vertices all rounded
            # to the same SVG coordinate). Skip it rather than emitting a
            # single-point <polyline>, which FracPaQ cannot process.
            if len(deduped) < 2:
                continue

            # Serialise to FracPaQ's expected "x1 y1 x2 y2 …" format with
            # six decimal places of precision for sub-pixel accuracy.
            pts = " ".join(f"{sx:.6f} {sy:.6f}" for sx, sy in deduped)
            safe_fid = re.sub(r"[^A-Za-z0-9_-]", "_", str(feat.id))
            polyline_id = f"feature_{safe_fid}_branch_{branch.id}"
            lines.append(f'  <polyline id="{polyline_id}" class="sk" points="{pts}"/>')
            features_written.add(feat.id)

    # Warn for features that had branches but produced no polyline output
    for feat in features:
        if feat.branches and feat.id not in features_written:
            print(f"  Warning: feature {feat.id!r} produced no polyline vertices in SVG output "
                  f"(all {len(feat.branches)} branch(es) collapsed to < 2 unique points).")

    lines.append("</svg>")

    # Write as a single joined string. Using "\n".join avoids a trailing
    # newline after </svg>, which some strict XML validators flag as an error.
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
