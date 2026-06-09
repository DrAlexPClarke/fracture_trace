"""
main.py — Entry point for the polygon skeletonisation and width-analysis pipeline.

Orchestrates parsing, parallel compute, sequential export, topology connection,
and global output (summary CSV + skeleton SVG).
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from config import (
    INPUT_FILE, OUTPUT_DIRECTORY, SAMPLING_INTERVAL, SMOOTHING,
    RASTER_RESOLUTION, N_WORKERS, SEPARATE_MULTIPOLYGONS,
    PRESERVE_TOPOLOGY, EXPORT_RAW_TRACES,
    EXPORT_SKELETON_OVERLAY, EXPORT_PROFILE_PLOT, EXPORT_PROFILE_DATA,
    EXPORT_PROFILE_FFT,
)
from config import _apply_extent_scale
from data_models import Branch, Feature
from helpers import _explode_multipolygons, _filter_small_features, _reconnect_topology
from parsers import (
    parse_shapefile, parse_svg, parse_pdf, parse_image,
    _dispatch_parser,
)
from skeletonisation import _dispatch_skeletoniser, _geometry_from_polygon, _is_line_feature
from analysis import _find_partial_thickness, _calculate_statistics
from exports import (
    _export_individual_csvs, _export_summary_csv,
    _plot_width_profile, _plot_tortuosity_fft, _plot_individual_feature_skeleton,
    _export_skeleton_svg,
)


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def _compute_features(feat, index, total):
    """
    Execute the complete compute pipeline for a single feature and return
    the updated feature dict together with a list of log lines.

    This function is designed to be submitted to a ``ThreadPoolExecutor``.
    All operations it performs — NumPy array manipulation, scikit-image
    morphological thinning, and Shapely geometry operations — release the
    Python GIL, so multiple calls can run with genuine parallelism on
    multi-core hardware.

    **Matplotlib is deliberately not called here.** Matplotlib's global
    state is not thread-safe; all rendering is deferred to the sequential
    export phase in the main thread (see :func:`_export_features`).

    **Pipeline steps**:

    1. **Skeletonise** (:func:`_dispatch_skeletoniser`): rasterise the polygon, thin it
       to a one-pixel-wide skeleton, and extract centreline branches using
       the method specified by ``SKELETONISATION_METHOD``.
    2. **Width measurement** (:func:`_find_partial_thickness`): cast
       perpendicular rays at each resampled sample point and record the
       partial thickness on each side.
    3. **Statistics** (:func:`_calculate_statistics`): aggregate the width
       measurements and geometric properties into ``feat.stats``.

    If skeletonisation produces no branches (e.g. the polygon is too thin
    to survive morphological thinning at the current ``RASTER_RESOLUTION``),
    steps 2 and 3 are skipped and a stats dict of ``None`` values is written
    directly so that downstream CSV export always has a complete row for
    every feature.

    Log lines are collected into a list and returned rather than printed
    directly, so the caller can print them atomically after the future
    completes. This prevents log output from different concurrent workers
    from interleaving in the terminal.

    Args:
        feat (Feature): A Feature dataclass instance as produced by a parser
            and :func:`_explode_multipolygons`. Must have at minimum a
            ``polygon`` field and an ``id`` field.
        index (int): 1-based position of this feature in the full feature list,
            used in the ``"Feature X of Y"`` progress log line.
        total (int): Total number of features being processed.

    Returns:
        tuple[Feature, list[str]]:
            - The same ``feat`` instance, mutated in place to populate the
              ``branches``, ``stats``, ``fft_data``, and ``skeleton_overlay_data``
              fields.
            - A list of log line strings describing what was computed,
              suitable for printing with ``"\n".join(lines)``.
    """
    lines = []
    fid   = feat.id
    lines.append(f"\nFeature {fid}")

    # ── Step 1: Skeletonise ───────────────────────────────────────────────────
    lines.append("  Skeletonising…")
    _dispatch_skeletoniser(feat)
    n_branches = len(feat.branches)
    total_pts  = sum(len(b.centerline) for b in feat.branches)
    lines.append(f"  {n_branches} branch(es), {total_pts} centreline point(s).")

    if n_branches == 0:
        # A zero-branch result means the polygon was too thin or small to
        # survive morphological thinning at the current raster resolution.
        # Width measurement and FFT are meaningless without a skeleton, so
        # skip both steps and populate the stats dict with explicit Nones.
        # This ensures the summary CSV always has a complete row for every
        # input polygon, making it clear the feature was processed but
        # yielded no measurements, rather than simply being absent.
        lines.append("  Skipping width measurement — no skeleton produced "
                     "(polygon may be too thin or small).")

        # Still compute the geometry-only fields (orientation, long-axis
        # length) so the summary CSV is as informative as possible even for
        # degenerate features.
        orientation_deg, long_axis_length = _geometry_from_polygon(feat.polygon)

        # Populate fft_data with empty arrays and null peaks so _export_features
        # and _plot_tortuosity_fft can run without an AttributeError even though
        # no FFT (fast Fourier transform) was actually computed.
        feat.fft_data = {"freqs": [], "magnitudes": [], "peaks": [None, None, None]}
        feat.stats = {
            "feature_id":             fid,
            "n_branches":             0,
            "n_samples":              0,
            "average_thickness":      None,
            "minimum_thickness":      None,
            "maximum_thickness":      None,
            "roughness_side_a":       None,
            "roughness_side_b":       None,
            "roughness":              None,
            "orientation_deg":        orientation_deg,
            "long_axis_length":       long_axis_length,
            "path_length":            None,
            "tortuosity":             None,
            "aspect_ratio_path":      None,
            "aspect_ratio_long_axis": None,
            "fft_peak_wavelength_1":  None,
            "fft_peak_wavelength_2":  None,
            "fft_peak_wavelength_3":  None,
        }
        return feat, lines

    # ── Step 2: Width measurement ─────────────────────────────────────────────
    if _is_line_feature(feat):
        # Line features carry no enclosing polygon, so ray-casting width
        # measurement is meaningless.  Skip straight to statistics; all width
        # fields will be None because branch.profile is empty.
        lines.append("  Line input — width measurement skipped.")
        _calculate_statistics(feat)
        s = feat.stats
        lines.append(
            f"  path_length={s.get('path_length') or 'n/a'}  "
            f"tortuosity={s.get('tortuosity') or 'n/a'}"
        )
    else:
        lines.append("  Measuring widths…")
        _find_partial_thickness(feat)

        # ── Step 3: Statistics ────────────────────────────────────────────────
        _calculate_statistics(feat)
        s = feat.stats

        # Log a one-line summary of the key statistics. The guard on
        # average_thickness covers the case where every sample point's rays
        # missed the boundary (all profile entries are None), which is rare but
        # can happen on very thin or oddly shaped features.
        if s.get("average_thickness") is not None:
            lines.append(
                f"  avg={s['average_thickness']:.3f}  "
                f"min={s['minimum_thickness']:.3f}  "
                f"max={s['maximum_thickness']:.3f}  "
                f"roughness={s['roughness']:.3f}"
            )
        else:
            lines.append("  No valid width measurements obtained.")

    return feat, lines

def _export_features(feat, output_dir):
    """
    Execute the export phase for a single feature: write profile CSVs and
    render Matplotlib plots to the output directory.

    **This function must only be called from the main thread.** Matplotlib
    maintains global figure state that is not protected by the GIL and is
    not safe to use from concurrent threads. All rendering is therefore
    deferred from the parallel compute phase (:func:`_compute_features`) and
    executed here sequentially after all workers have completed.

    Each output type is guarded by a global boolean flag (``EXPORT_*``), so
    individual outputs can be enabled or disabled without changing this
    function. The skeleton overlay is the only output rendered even for
    features that produced no branches, since it is the primary diagnostic
    tool for understanding why skeletonisation failed.

    Errors in optional plot rendering (FFT and skeleton overlay) are caught
    and logged rather than re-raised, so a single malformed feature does not
    abort the export of the remaining features.

    Args:
        feat (Feature): A fully computed Feature dataclass instance as
            returned by :func:`_compute_features`, with ``branches``,
            ``stats``, ``fft_data``, and ``skeleton_overlay_data`` fields populated.
        output_dir (str | os.PathLike): Directory in which to write all
            output files. Must already exist.

    Returns:
        list[str]: Log lines describing each file written (prefixed with
        ``"  → "``), or error messages for any plots that failed to render.
        Returns an empty list if no export flags are enabled or if the
        feature has no branches and overlay export is also disabled.
    """
    lines      = []
    n_branches = len(feat.branches)

    # Only write branch-dependent outputs (profile CSVs and graphs) when
    # at least one branch was produced. An empty branch list means skeletonisation
    # failed; writing an empty profile CSV or a blank graph would be misleading.
    if n_branches > 0:

        # ── Per-branch width profile CSV ──────────────────────────────────────
        if config.EXPORT_PROFILE_DATA:
            for csv_path in _export_individual_csvs(feat, output_dir):
                lines.append(f"  → {csv_path}")

        # ── Per-branch width profile graph ────────────────────────────────────
        if config.EXPORT_PROFILE_PLOT:
            for graph_path in _plot_width_profile(feat, output_dir):
                lines.append(f"  → {graph_path}")

        # ── Tortuosity FFT (fast Fourier transform) spectrum plot ────────────
        # Wrapped in try/except because FFT plotting can fail if the feature
        # has an unusual geometry that produces a degenerate deviation signal
        # (e.g. all sample points at the same location). The failure is logged
        # but does not block the CSV or other plot outputs.
        if config.EXPORT_PROFILE_FFT:
            try:
                fft_path = _plot_tortuosity_fft(feat, output_dir)
                lines.append(f"  → {fft_path}")
            except Exception as exc:
                lines.append(f"  [fft] plot failed: {exc}")

    # ── Skeleton overlay ──────────────────────────────────────────────────────
    # Rendered regardless of branch count, because the overlay is the primary
    # diagnostic tool for features where skeletonisation produced nothing —
    # it shows the rasterised footprint and (if available) the raw skeleton
    # grid, which often reveals whether the resolution was too coarse or the
    # polygon was too narrow to survive thinning.
    if config.EXPORT_SKELETON_OVERLAY:
        try:
            skeleton_overlay_path = _plot_individual_feature_skeleton(feat, output_dir)
            lines.append(f"  → {skeleton_overlay_path}")
        except Exception as exc:
            lines.append(f"  [overlay] plot failed: {exc}")

    return lines

def main():
    """
    Entry point for the polygon skeletonisation and width-analysis pipeline.

    Orchestrates the full five-stage workflow:

    1. **Parse**: read the input file via :func:`_dispatch_parser`, then
       normalise compound geometries with :func:`_explode_multipolygons`.

    2. **Compute** (parallel): submit each feature to a
       ``ThreadPoolExecutor`` that runs :func:`_compute_features` — rasterising,
       skeletonising, measuring widths, and computing statistics. Heavy
       numerical operations (NumPy, scikit-image, Shapely) release the GIL
       so all workers run concurrently. Log lines are collected inside each
       worker and printed atomically after the future resolves to prevent
       interleaved terminal output.

    3. **Export** (sequential, main thread): call :func:`_export_features` for
       each feature to write profile CSVs and render Matplotlib plots.
       Matplotlib's global figure state is not thread-safe, so this phase
       runs in the main thread after all compute workers have finished.

    4. **Topology connection** (optional): if ``PRESERVE_TOPOLOGY`` is enabled,
       call :func:`_reconnect_topology` to extend branch endpoints to the
       nearest point on neighbouring feature skeletons, restoring T-junction
       and end-to-end connections lost during independent skeletonisation.

    5. **Global outputs**: write the multi-feature summary CSV via
       :func:`_export_summary_csv` and the FracPaQ-compatible skeleton SVG
       via :func:`_export_skeleton_svg`.

    **Configuration** is read entirely from module-level constants
    (``INPUT_FILE``, ``OUTPUT_DIRECTORY``, ``SAMPLING_INTERVAL``, etc.),
    so no command-line argument parsing is required.

    The ``_flip_y`` flag is determined by majority vote across all features:
    if any feature carries ``_flip_y=True`` (i.e. originated from a Shapefile
    or other north-up coordinate system), the skeleton SVG is written with
    the y-axis inverted so geographic north faces upward in the output.

    Exits with code 1 if the input file does not exist, or with code 0 if
    parsing succeeds but yields no features.
    """
    input_path = Path(config.INPUT_FILE)

    if not input_path.exists():
        print(f"Error: file not found: {config.INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    # Create the output directory if it doesn't already exist. exist_ok=True
    # prevents a race condition if two processes start simultaneously and both
    # attempt to create the directory.
    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)

    # Print a configuration summary before any processing begins so the log
    # is self-documenting — the key parameters used for this run are always
    # visible at the top of the output even when the run is piped to a file.
    print("=" * 60)
    print("Polygon Skeletonisation and Width Analysis")
    print("=" * 60)
    n_workers = config.N_WORKERS if config.N_WORKERS is not None else os.cpu_count() or 1
    print(f"Input            : {config.INPUT_FILE}")
    print(f"Output directory : {config.OUTPUT_DIRECTORY}/")
    print(f"Sampling interval: {config.SAMPLING_INTERVAL}")
    print(f"Smoothing sigma  : {config.SMOOTHING}")
    print(f"Raster resolution: {config.RASTER_RESOLUTION}")
    print(f"Workers          : {n_workers}")
    print()

    # ── Stage 1: Parse ────────────────────────────────────────────────────────
    print("Parsing input…")
    features = _dispatch_parser(input_path)
    if config.SEPARATE_MULTIPOLYGONS:
        # Split every MultiPolygon feature into one feature per component Polygon.
        # This must happen before any parallel submission (the feature list is the
        # unit of parallelism) and, critically, before the per-feature rasterisation
        # buffer in _dispatch_skeletoniser:
        #
        #   Separation first → no zero-width connections remain in the dataset.
        #   Buffer second    → each component is expanded independently; previously
        #                      separated features can never be re-bridged because
        #                      every feature's raster grid is built from its own
        #                      local polygon copy.
        features = _explode_multipolygons(features)
    else:
        # Components of a compound feature are intentionally kept together.
        # _dispatch_skeletoniser applies a half-pixel outward buffer before
        # rasterisation; because the buffer is applied to the whole (multi-part)
        # polygon at once, it inflates and merges zero-width connections so the
        # rasterised skeleton stays connected across them.  Features whose
        # geometry is already a simple Polygon are unaffected by this path.
        pass
    n_parsed = len(features)
    # Scale all %-based config parameters to world units using the combined
    # bounding box of the loaded features (after artboard filtering).
    _apply_extent_scale(features)
    # Drop artefactual slivers (e.g. near-zero-width swells at fracture ends)
    # before any rasterisation cost is incurred.
    features = _filter_small_features(features)
    n_kept   = len(features)
    mode_note = ("connected skeleton" if not config.SEPARATE_MULTIPOLYGONS
                 else "separate features")
    if config.MINIMUM_FEATURE_SIZE > 0 and n_kept < n_parsed:
        print(f"  Found {n_parsed} feature(s) [{mode_note}]; "
              f"{n_parsed - n_kept} dropped "
              f"(bbox < {config.MINIMUM_FEATURE_SIZE}²), {n_kept} remaining.")
    else:
        print(f"  Found {n_parsed} feature(s) [{mode_note}].")

    if not features:
        print("No features found — exiting.")
        sys.exit(0)

    # Determine the y-axis flip convention for the final SVG from the parsed
    # features. Any feature with _flip_y=True originates from a coordinate
    # system where y increases upward (e.g. Shapefiles in projected CRS);
    # the SVG must invert the y-axis to match that convention visually.
    flip_y = any(f.flip_y for f in features)

    # ── Stage 2: Compute (parallel) ───────────────────────────────────────────
    print(f"Processing {len(features)} feature(s) with {n_workers} worker(s)…")

    # Each feature is submitted as an independent future. The futures_map dict
    # maps each future back to its source feature so we can report the feature
    # ID in error messages even if the future raises before returning a result.
    futures_map = {}
    n_features  = len(features)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, feat in enumerate(features, start=1):
            future = executor.submit(_compute_features, feat, idx, n_features)
            futures_map[future] = (feat, idx)

        # as_completed yields futures in the order they finish, not submission
        # order. Printing log lines only after each future resolves guarantees
        # that all lines from one feature are printed together, preventing
        # interleaved output from concurrent workers.
        #
        # The completion counter (n_done) is incremented here in the main
        # thread — the only place where ordering is known — and appended to
        # the header line that _compute_features already placed at lines[0].
        # This gives a reliable "[1 of N], [2 of N], …" sequence even though
        # futures complete in an unpredictable order.
        n_done = 0
        for future in as_completed(futures_map):
            feat, idx = futures_map[future]
            n_done += 1
            try:
                _feat, lines = future.result()
                if lines:
                    lines[0] = f"{lines[0]}  [{n_done} of {n_features}]"
                print("\n".join(lines))
            except Exception as exc:
                print(f"\nFeature {feat.id}  [{n_done} of {n_features}]  ✗ failed: {exc}")

    # ── Stage 3: Export single-feature plots (sequential, main thread) ─────────────────────────────
    # Matplotlib is not thread-safe, so all SVG plot rendering is deferred to
    # this sequential pass in the main thread. CSVs are also written here for
    # simplicity, even though they are thread-safe, to keep the two phases
    # cleanly separated.
    n_export = len(features)
    for export_idx, feat in enumerate(features, start=1):
        print(f"  Exporting feature {feat.id}  [{export_idx} of {n_export}]…")
        export_lines = _export_features(feat, config.OUTPUT_DIRECTORY)
        if export_lines:
            print("\n".join(export_lines))

    # ── Stage 4: Topology connection ───────────────────────────────
    # Must run after all features are fully computed and exported, because it
    # modifies skeleton geometry in place and those modifications should not
    # be reflected in the per-feature profile CSVs (which represent width
    # measurements within each polygon's own interior only).
    if config.PRESERVE_TOPOLOGY:
        print("\nConnecting topology…")
        n_ext = _reconnect_topology(features)
        print(f"  {n_ext} endpoint extension(s) applied.")

    # ── Stage 5: Global outputs ───────────────────────────────────────────────
    # The summary CSV and skeleton SVG are written after topology connection
    # so they reflect the final connected centreline geometry.
    summary_path = _export_summary_csv(features, config.OUTPUT_DIRECTORY)
    print(f"\n→ {summary_path}")

    svg_path = _export_skeleton_svg(features, config.OUTPUT_DIRECTORY, flip_y=flip_y)
    if svg_path:
        print(f"→ {svg_path}")

    if config.EXPORT_RAW_TRACES:
        raw_svg_path = _export_skeleton_svg(
            features, config.OUTPUT_DIRECTORY, flip_y=flip_y,
            coords_attr="centerline", filename="skeleton_raw.svg",
        )
        if raw_svg_path:
            print(f"→ {raw_svg_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
