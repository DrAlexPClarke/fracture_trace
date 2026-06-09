"""
analysis.py — Width measurement and statistics for the skeletonisation pipeline.

Computes per-branch arc-length resampling, perpendicular-ray width profiles,
aggregate statistics, and tortuosity FFT for each Feature produced by
skeletonisation.py.
"""

import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.prepared import prep

import config
from config import SAMPLING_INTERVAL, MAX_WIDTH_RAY_DISTANCE
from data_models import Feature
from skeletonisation import _geometry_from_polygon


# =============================================================================
# ANALYSIS
# =============================================================================

def _calculate_statistics(feature):
    """
    Compute aggregate width statistics and geometric properties for a feature
    and write them to ``feature.stats`` in place.

    The function is divided into two paths depending on whether width
    measurement data is available in the feature's branch profiles:

    **Width statistics** (only when profile data exists)

    - ``average_thickness``, ``minimum_thickness``, ``maximum_thickness``:
      descriptive statistics over all per-sample-point width measurements
      across all branches.
    - ``roughness_side_a`` / ``roughness_side_b``: standard deviation of the
      partial-thickness measurements on each side of the centreline
      independently. A rough wall produces high variance in its partial
      thickness; a smooth wall produces low variance.
    - ``roughness``: mean of the two per-side standard deviations, giving a
      single representative surface-roughness value for the feature.

    **Geometric properties** (always computed, even when no width data exists)

    - ``orientation_deg``: long-axis angle in degrees from the +x axis,
      normalised to ``[0, 180)`` via PCA on the polygon exterior ring.
    - ``long_axis_length``: extent of the polygon along its PCA principal axis.
    - ``path_length``: total arc length of all branch centrelines combined.
    - ``tortuosity``: ``path_length / chord_length``, where chord length is
      the straight-line distance from the first to the last centreline point
      across all branches. A perfectly straight feature returns ``1.0``;
      values greater than ``1.0`` indicate sinuosity.
    - ``aspect_ratio_path``: ``path_length / average_thickness``.
    - ``aspect_ratio_long_axis``: ``long_axis_length / average_thickness``.
    - ``fft_peak_wavelength_1/2/3``: the three dominant wavelengths (in world
      units) from the FFT (fast Fourier transform) of the lateral-deviation signal, sorted by
      descending spectral magnitude.

    An FFT summary dict is also written to ``feature.fft_data`` for optional
    plotting or export.

    Args:
        feature (Feature): Feature dataclass instance containing at minimum:

            - ``id`` (*str*): Feature identifier used in the stats record.
            - ``polygon`` (*shapely.geometry.Polygon*): Source geometry for
              PCA-based orientation and length calculations.
            - ``branches`` (*list[Branch]*): Branch instances from
              :func:`_dispatch_skeletoniser`, each optionally containing a
              non-empty ``profile`` list with per-sample measurement dicts or
              ``None`` values as produced by :func:`_find_partial_thickness`.

    Returns:
        Feature: The same ``feature`` instance, mutated in place to populate
        the ``stats`` and ``fft_data`` fields. Width-dependent stats are set
        to ``None`` when no profile data is available.
    """
    # Collect width measurements from every sample point across all branches.
    # A profile entry of None indicates a sample point where one or both rays
    # failed to hit the polygon boundary (e.g. near a branch tip); these are
    # excluded from the statistics without raising an error.
    widths = []
    side_a = []
    side_b = []

    for branch in feature.branches:
        for pt in branch.profile:
            if pt is None:
                continue
            widths.append(pt["width"])
            side_a.append(pt["side_a"])
            side_b.append(pt["side_b"])

    # ── Geometric properties (independent of width data) ─────────────────────
    # These are always computed first so they can be included in the null-stats
    # path below without duplicating the calculation logic.
    orientation_deg, long_axis_length = _geometry_from_polygon(feature.polygon)
    branches    = feature.branches
    path_length = _total_path_length(branches)

    # ── Tortuosity ────────────────────────────────────────────────────────────
    # Flatten all centreline points across branches into a single sequence to
    # get the overall start-to-end chord. Concatenating branch points this way
    # is only geometrically meaningful when branches are ordered end-to-end,
    # which is the case for the single-branch and directional methods. For
    # multi-branch features the tortuosity is still computed but should be
    # interpreted as an approximation.
    all_cl_pts = []
    for branch in branches:
        pts = branch.centerline
        if len(pts) >= 2:
            all_cl_pts.extend(pts)

    if len(all_cl_pts) >= 2:
        arr_cl    = np.array(all_cl_pts, dtype=float)
        chord_len = float(np.linalg.norm(arr_cl[-1] - arr_cl[0]))
        # Guard against a degenerate chord (e.g. a closed-loop centreline
        # where start == end). A chord of essentially zero implies the feature
        # curves back on itself; tortuosity is undefined in that case so we
        # return 1.0 as a neutral fallback rather than a division-by-zero NaN.
        tortuosity = path_length / chord_len if chord_len > 1e-9 else 1.0
    else:
        tortuosity = None

    # ── FFT (fast Fourier transform) of the lateral-deviation signal ──────────
    # The FFT captures dominant spatial wavelengths in the centreline's
    # lateral wandering, which are written to both the stats record and the
    # separate _fft dict for downstream plotting.
    fft_peaks, fft_freqs, fft_mags = _tortuosity_fft_peaks(branches)
    feature.fft_data = {
        "freqs":      fft_freqs,
        "magnitudes": fft_mags,
        "peaks":      fft_peaks,
    }

    # ── Null-stats path: no width measurements available ─────────────────────
    # Return a stats dict with None for all width-derived fields rather than
    # omitting the feature from the output entirely, so that downstream
    # consumers can include the feature in CSV exports with clearly marked
    # missing values.
    if not widths:
        feature.stats = {
            "feature_id":             feature.id,
            "n_branches":             len(branches),
            "n_samples":              0,
            "average_thickness":      None,
            "minimum_thickness":      None,
            "maximum_thickness":      None,
            "roughness_side_a":       None,
            "roughness_side_b":       None,
            "roughness":              None,
            "orientation_deg":        orientation_deg,
            "long_axis_length":       long_axis_length,
            # path_length of exactly 0 means no valid branches; store as None
            # rather than 0 so consumers can distinguish "not measured" from
            # "genuinely zero length".
            "path_length":            path_length if path_length > 0 else None,
            "tortuosity":             tortuosity,
            "aspect_ratio_path":      None,
            "aspect_ratio_long_axis": None,
            "fft_peak_wavelength_1":  fft_peaks[0],
            "fft_peak_wavelength_2":  fft_peaks[1],
            "fft_peak_wavelength_3":  fft_peaks[2],
        }
        return feature

    # ── Full stats path: width measurements present ───────────────────────────
    # Per-side roughness is the standard deviation of the partial thickness on
    # each wall independently. Using std rather than mean deviation aligns with
    # the standard surface-roughness convention (Ra / Rq) and is sensitive to
    # large local excursions that the mean would mask.
    ra  = float(np.std(side_a))
    rb  = float(np.std(side_b))
    avg = float(np.mean(widths))

    # Guard both aspect ratios against division by zero. A zero average
    # thickness would indicate a degenerate feature (all width measurements
    # collapsed to a point), in which case the ratio is meaningless.
    aspect_ratio_path = (
        path_length / avg if path_length > 0 and avg > 0 else None
    )
    aspect_ratio_long_axis = (
        long_axis_length / avg
        if long_axis_length is not None and avg > 0
        else None
    )

    feature.stats = {
        "feature_id":             feature.id,
        "n_branches":             len(branches),
        "n_samples":              len(widths),
        "average_thickness":      avg,
        "minimum_thickness":      float(np.min(widths)),
        "maximum_thickness":      float(np.max(widths)),
        "roughness_side_a":       ra,
        "roughness_side_b":       rb,
        # Combined roughness: arithmetic mean of the two per-side standard
        # deviations, giving equal weight to both walls of the feature.
        "roughness":              (ra + rb) / 2.0,
        "orientation_deg":        orientation_deg,
        "long_axis_length":       long_axis_length,
        "path_length":            path_length if path_length > 0 else None,
        "tortuosity":             tortuosity,
        "aspect_ratio_path":      aspect_ratio_path,
        "aspect_ratio_long_axis": aspect_ratio_long_axis,
        "fft_peak_wavelength_1":  fft_peaks[0],
        "fft_peak_wavelength_2":  fft_peaks[1],
        "fft_peak_wavelength_3":  fft_peaks[2],
    }
    return feature

def _resample_at_interval(coords):
    """
    Resample a polyline at evenly-spaced arc-length intervals of
    ``SAMPLING_INTERVAL`` world units.

    The resampled points are used as the measurement locations for the width
    profiling step (:func:`_find_partial_thickness`). Uniform arc-length
    spacing ensures that width measurements are distributed evenly along the
    centreline regardless of how the original skeleton vertices are spaced,
    which prevents dense vertex clusters near junctions from biassing the
    statistics.

    Interpolation is performed with ``numpy.interp`` on the cumulative
    arc-length parameter, which is equivalent to linear interpolation between
    consecutive skeleton vertices.

    Args:
        coords (list[tuple[float, float]]): Ordered ``(x, y)`` centreline
            coordinate pairs. Must contain at least two points; shorter
            inputs are returned unchanged.

    Returns:
        list[tuple[float, float]]: Resampled coordinate list. If the total
        arc length of the input is less than one ``SAMPLING_INTERVAL``, the
        original coordinate list is returned as-is (no interpolation is
        performed). Otherwise the output contains one point at each integer
        multiple of ``SAMPLING_INTERVAL`` along the arc, starting at
        ``t = 0``. The output always contains at least two points (start and
        total-length) when the arc is long enough for interpolation.
    """
    if len(coords) < 2:
        return coords

    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])

    # Compute the cumulative arc-length parameter: cumlen[i] is the total
    # distance along the polyline from the first vertex to vertex i.
    dists  = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total  = cumlen[-1]

    # If the polyline is shorter than one sampling interval, there is nowhere
    # to place an interior sample; return the original points so the caller
    # still has valid start/end coordinates for any tangent calculations.
    if total < config.SAMPLING_INTERVAL:
        return list(zip(xs.tolist(), ys.tolist()))

    # Generate sample positions at every integer multiple of SAMPLING_INTERVAL
    # from 0 up to (but not including) total. np.arange is used rather than
    # np.linspace because we want a fixed step size, not a fixed count.
    sample_t = np.arange(0.0, total, config.SAMPLING_INTERVAL)

    # np.arange can return only t=0 when total is exactly one interval due to
    # floating-point rounding. Downstream tangent estimation requires at least
    # two sample points (pts[0] and pts[1]), so force a minimum of two by
    # explicitly including the total arc-length endpoint.
    if len(sample_t) < 2:
        sample_t = np.array([0.0, total])

    # Linearly interpolate x and y at each sample position independently.
    # np.interp clamps out-of-range values to the array endpoints, so
    # sample_t values at exactly 0.0 or total are handled correctly.
    xs_r = np.interp(sample_t, cumlen, xs)
    ys_r = np.interp(sample_t, cumlen, ys)
    return list(zip(xs_r.tolist(), ys_r.tolist()))

def _find_partial_thickness(feature):
    """
    Measure the polygon width perpendicular to the centreline at every
    sample point and write the results to each branch's ``"profile"`` key.

    For each sample point the algorithm:

    1. Estimates the local tangent direction from the neighbouring sample
       points (central difference for interior points; forward/backward
       difference at the endpoints).
    2. Constructs a unit normal vector perpendicular to the tangent.
    3. Casts two rays from the sample point — one in the ``+normal``
       direction (side A) and one in the ``-normal`` direction (side B) —
       out to ``max_dist`` world units.
    4. Intersects each ray with the polygon boundary and records the distance
       to the nearest intersection point as the partial thickness on that side.
    5. Sums the two partial thicknesses as the total width at that point.

    Sample points where one or both rays fail to intersect the boundary
    (which can happen at branch tips that extend slightly beyond the polygon,
    or at numerical edge cases) are stored as ``None`` in the profile rather
    than raising an error, so they are cleanly excluded from statistics
    without aborting the measurement pass.

    **Performance notes**

    The polygon boundary is extracted once and wrapped in a Shapely
    ``prep()`` prepared geometry, which builds an internal STR-tree index.
    This makes the ``intersects`` pre-check in ``_ray_dist`` very fast,
    avoiding the more expensive full ``intersection()`` call for rays that
    miss the boundary entirely — which is the common case for interior sample
    points where only one of the two ray directions needs to be checked.

    Args:
        feature (Feature): Feature dataclass instance containing:

            - ``polygon`` (*shapely.geometry.Polygon*): The geometry whose
              boundary is used as the measurement surface.
            - ``branches`` (*list[Branch]*): Branch instances, each with a
              ``sample_points`` field holding the output of
              :func:`_resample_at_interval`.

    Returns:
        Feature: The same ``feature`` instance, with ``branch.profile``
        populated for every branch. Each profile is a list of the same length
        as ``branch.sample_points``, containing either a measurement dict
        or ``None``. Measurement dicts have the keys:

        - ``"dist"`` (*float*): Cumulative arc-length distance from the first
          sample point to this point, in world units.
        - ``"x"``, ``"y"`` (*float*): World coordinates of the sample point.
        - ``"side_a"`` (*float*): Partial thickness in the ``+normal``
          direction.
        - ``"side_b"`` (*float*): Partial thickness in the ``-normal``
          direction.
        - ``"width"`` (*float*): Total width (``side_a + side_b``).
    """
    polygon  = feature.polygon
    boundary = polygon.boundary

    # Wrap the boundary in a prepared geometry so GEOS builds an STR-tree
    # index over it once. All subsequent prep_bnd.intersects() calls use that
    # index for a fast bounding-box pre-check before the full intersection.
    prep_bnd = prep(boundary)

    # Compute the maximum ray length. If MAX_WIDTH_RAY_DISTANCE is set, use it
    # directly. Otherwise fall back to twice the polygon's bounding-box diagonal
    # — guaranteed to be longer than any chord, so a ray that should hit the
    # boundary always will.
    bounds   = polygon.bounds
    auto_max = math.sqrt((bounds[2]-bounds[0])**2 + (bounds[3]-bounds[1])**2) * 2
    max_dist = config.MAX_WIDTH_RAY_DISTANCE if config.MAX_WIDTH_RAY_DISTANCE and config.MAX_WIDTH_RAY_DISTANCE > 0 else auto_max

    def _ray_dist(ox, oy, nx, ny, sign, origin_pt):
        """
        Cast a ray from ``(ox, oy)`` in the direction ``sign * (nx, ny)`` and
        return the distance to the nearest polygon boundary intersection.

        The ray is a ``LineString`` from the origin to the point
        ``(ox ± nx * max_dist, oy ± ny * max_dist)``. A prepared-geometry
        pre-check is used to quickly reject rays that do not touch the
        boundary's bounding envelope before computing the full intersection.

        Intersection results are normalised to a flat list of ``Point``
        geometries regardless of whether Shapely returns a ``Point``,
        ``MultiPoint``, or ``GeometryCollection``. The nearest intersection
        that is more than ``1e-6`` world units from the origin is returned,
        discarding the degenerate case where the ray starts exactly on the
        boundary.

        Args:
            ox (float): Ray origin x-coordinate.
            oy (float): Ray origin y-coordinate.
            nx (float): Unit normal x-component (pre-normalised by the caller).
            ny (float): Unit normal y-component (pre-normalised by the caller).
            sign (int): ``+1`` for side A (positive normal direction) or
                ``-1`` for side B (negative normal direction).
            origin_pt (shapely.geometry.Point): Pre-constructed Shapely Point
                at ``(ox, oy)``. Passed in to avoid re-allocating it for
                each of the two ray directions at the same sample point.

        Returns:
            float | None: Distance from the origin to the nearest valid
            boundary intersection, or ``None`` if no intersection was found.
        """
        ex  = ox + sign * nx * max_dist
        ey  = oy + sign * ny * max_dist
        ray = LineString([(ox, oy), (ex, ey)])

        # Fast reject: if the ray's bounding box doesn't overlap the boundary's
        # STR-tree envelope, there is no intersection and we can skip the
        # more expensive .intersection() call entirely.
        if not prep_bnd.intersects(ray):
            return None

        try:
            inter = ray.intersection(boundary)
            if inter.is_empty:
                return None

            # Normalise the intersection to a flat list of Points. Shapely
            # can return Point, MultiPoint, or GeometryCollection depending
            # on the boundary topology at the intersection site. The
            # GeometryCollection branch also handles nested MultiPoints that
            # can appear when a ray clips a boundary vertex exactly.
            if inter.geom_type == "Point":
                geoms = [inter]
            elif inter.geom_type == "MultiPoint":
                geoms = list(inter.geoms)
            elif inter.geom_type == "GeometryCollection":
                geoms = [g for g in inter.geoms if g.geom_type == "Point"]
                for g in inter.geoms:
                    if g.geom_type == "MultiPoint":
                        geoms.extend(g.geoms)
            else:
                # Fallback for unexpected geometry types (e.g. a LineString
                # when the ray runs exactly along the boundary). Take the
                # first coordinate of the result as a single Point.
                geoms = [Point(inter.coords[0])]

            # Filter out intersections at the origin itself (distance < 1e-6),
            # which arise when the sample point sits exactly on the boundary.
            # Return the minimum distance among the remaining intersections,
            # which is the nearest boundary wall on this ray direction.
            dists = [d for d in (origin_pt.distance(g) for g in geoms)
                     if d > 1e-6]
            return min(dists) if dists else None

        except Exception:
            # Shapely can raise on degenerate inputs (collinear vertices,
            # NaN coordinates, etc.). Return None so the sample point is
            # marked as missing rather than crashing the whole profile pass.
            return None

    for branch in feature.branches:
        pts     = branch.sample_points
        n       = len(pts)
        profile = []
        cumulative_dist = 0.0   # Running arc-length from the first sample point.

        for i, (x, y) in enumerate(pts):

            # ── Tangent estimation ────────────────────────────────────────────
            # Central difference is used for interior points because it gives
            # a more accurate tangent estimate than a one-sided difference at
            # the cost of one extra point lookup. Forward/backward differences
            # are used at the endpoints where no second neighbour is available.
            if i == 0:
                dx, dy = pts[1][0] - x,             pts[1][1] - y
            elif i == n - 1:
                dx, dy = x - pts[-2][0],             y - pts[-2][1]
            else:
                dx, dy = pts[i+1][0] - pts[i-1][0], pts[i+1][1] - pts[i-1][1]

            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                # The tangent vector is degenerate (two identical consecutive
                # sample points). This should not occur after resampling at a
                # fixed interval, but if it does, record None and move on.
                profile.append(None)
                continue

            # Rotate the tangent vector 90° CCW to get the unit normal:
            # tangent (dx, dy) → normal (-dy, dx), then normalise.
            nx, ny = -dy / length, dx / length
            # Construct the origin Point once and share it between both ray
            # directions to avoid allocating two redundant Shapely objects.
            origin = Point(x, y)

            d_a = _ray_dist(x, y, nx, ny, +1, origin)   # +normal direction (side A)
            d_b = _ray_dist(x, y, nx, ny, -1, origin)   # −normal direction (side B)

            # Undirected segment orientation: bearing of the skeleton tangent
            # from geographic north (+Y up), measured clockwise, normalised to
            # [0°, 180°) so opposite directions map to the same angle (geological
            # strike convention, suitable for rose diagrams).
            #
            # SVG/PDF/raster coordinates have +Y pointing downward (screen space),
            # so negate dy before computing the azimuth to convert to the standard
            # geographic convention (+Y = north).  Shapefiles already use +Y = north
            # (flip_y=True) so no correction is needed there.
            geo_dy = dy if feature.flip_y else -dy
            orientation_deg = math.degrees(math.atan2(dx / length, geo_dy / length)) % 180.0

            # Accumulate arc-length distance from the previous sample point.
            # Skipped for i=0 since there is no predecessor to measure from.
            if i > 0:
                px, py = pts[i-1]
                cumulative_dist += math.sqrt((x-px)**2 + (y-py)**2)

            if d_a is not None and d_b is not None:
                profile.append({
                    "dist":            cumulative_dist,
                    "x":               x,
                    "y":               y,
                    "side_a":          d_a,
                    "side_b":          d_b,
                    "width":           d_a + d_b,
                    "orientation_deg": orientation_deg,
                })
            else:
                # One or both rays missed the boundary — store None so the
                # statistics step can cleanly exclude this measurement without
                # needing to inspect partial fields.
                profile.append(None)

        branch.profile = profile

    return feature

def _total_path_length(branches):
    """
    Compute the total arc length of all branch centrelines combined.

    Each branch's centreline is treated as a piecewise-linear path; the arc
    length is the sum of the Euclidean distances between consecutive vertices.
    Branch arc lengths are summed to give the total path length for the
    feature, which is used in tortuosity and aspect-ratio calculations.

    Args:
        branches (list[Branch]): Branch instances, each with a ``centerline``
            field holding an ordered list of ``(x, y)`` world coordinate
            pairs. Branches with fewer than two vertices contribute zero
            length.

    Returns:
        float: Total arc length in world units. Returns ``0.0`` if
        ``branches`` is empty or all branches have fewer than two vertices.
    """
    total = 0.0
    for branch in branches:
        pts = branch.centerline
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=float)
        # Compute all inter-vertex distances in one vectorised call using the
        # difference array. np.diff on a (N, 2) array produces an (N-1, 2)
        # array of displacement vectors; the Euclidean norm of each row is the
        # segment length.
        total += float(np.sum(
            np.sqrt(np.diff(arr[:, 0])**2 + np.diff(arr[:, 1])**2)
        ))
    return total

def _tortuosity_fft_peaks(branches, n_peaks=3):
    """
    Compute the dominant spatial frequencies of the lateral-deviation signal
    of a feature's centreline and return the corresponding wavelengths.

    The lateral-deviation signal captures the centreline's sideways wandering
    relative to its overall chord direction. It is constructed by projecting
    all sample points onto the unit normal of the chord vector, giving a 1-D
    time series whose frequency content encodes the spatial periodicity of
    the feature's curvature.

    **Signal construction**

    1. Concatenate all sample points from all branches into a single ordered
       array.
    2. Compute the chord vector from the first to the last sample point.
    3. If the chord length is negligible (the feature curves back on itself),
       substitute the PCA principal axis as the reference direction.
    4. Project all sample points onto the unit normal of the chord to obtain
       the 1-D lateral-deviation signal.
    5. Remove the mean (DC offset) so the FFT (fast Fourier transform) power
       is concentrated in the oscillatory components rather than the signal's
       absolute position.

    **Spectral analysis**

    The real FFT (``numpy.fft.rfft``) is applied to the zero-mean deviation
    signal. Frequencies are in cycles per world unit, computed assuming the
    sample points are spaced ``SAMPLING_INTERVAL`` apart (which is guaranteed
    by :func:`_resample_at_interval`). The DC component (index 0, frequency
    0) is excluded from peak selection because it represents the mean offset
    rather than any oscillatory wavelength.

    Args:
        branches (list[Branch]): Branch instances, each with a
            ``sample_points`` field holding at least two ``(x, y)`` tuples as
            produced by :func:`_resample_at_interval`.
        n_peaks (int): Number of dominant spectral peaks to return. Defaults
            to ``3``. The output is padded with ``None`` if fewer than
            ``n_peaks`` valid (non-DC, positive-frequency) peaks exist.

    Returns:
        tuple:
            - **peak_wavelengths** (*list[float | None]*): Wavelengths in
              world units of the ``n_peaks`` highest-magnitude spectral peaks,
              sorted by descending magnitude. Padded with ``None`` when fewer
              valid peaks are available.
            - **freqs** (*list[float]*): Full frequency array from
              ``numpy.fft.rfftfreq``, in cycles per world unit.
            - **magnitudes** (*list[float]*): FFT magnitude spectrum
              corresponding to ``freqs``.
    """
    # Collect sample points from all branches into a single array. A minimum
    # of 4 points is required for the FFT to produce at least 2 non-trivial
    # frequency bins; fewer points yield a spectrum that is too coarse to be
    # meaningful.
    all_pts = []
    for branch in branches:
        pts = branch.sample_points
        if len(pts) >= 2:
            all_pts.extend(pts)

    if len(all_pts) < 4:
        return [None] * n_peaks, [], []

    arr       = np.array(all_pts, dtype=float)
    chord_vec = arr[-1] - arr[0]
    chord_len = float(np.linalg.norm(chord_vec))

    if chord_len < 1e-9:
        # The centreline starts and ends at essentially the same point — it
        # forms a closed or near-closed loop. The chord direction is undefined,
        # so fall back to the PCA principal axis as a surrogate reference
        # direction for projecting lateral deviations.
        center   = arr.mean(axis=0)
        Cv       = arr - center
        cov      = (Cv.T @ Cv) / max(len(arr) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        chord_vec  = eigvecs[:, np.argmax(eigvals)]
        chord_len  = 1.0   # Normalisation denominator; chord_vec is already a unit vector.

    # Unit vectors along and perpendicular to the chord. The perpendicular
    # direction (rotated 90° CCW) defines the axis along which lateral
    # deviations are measured.
    chord_dir  = chord_vec / chord_len
    perp_dir   = np.array([-chord_dir[1], chord_dir[0]])

    # Project all sample points onto the perpendicular direction. Subtracting
    # arr[0] first translates the coordinate origin to the start of the
    # centreline so the projection measures displacement from the chord, not
    # from the coordinate system origin.
    deviations = (arr - arr[0]) @ perp_dir

    # Remove the mean (DC offset) so that the FFT energy is concentrated in
    # the oscillatory components. Without this step the DC bin would dominate
    # the spectrum whenever the centreline is laterally offset from the chord.
    deviations -= deviations.mean()

    n          = len(deviations)
    fft_vals   = np.fft.rfft(deviations)   # rfft = real-input fast Fourier transform (FFT)
    magnitudes = np.abs(fft_vals)
    # rfftfreq returns frequencies in cycles per sample; multiplying by
    # 1/SAMPLING_INTERVAL (via the d= parameter) converts to cycles per
    # world unit, so wavelengths can be expressed in the same units as the
    # polygon coordinates.
    freqs      = np.fft.rfftfreq(n, d=config.SAMPLING_INTERVAL)

    # ── Peak selection ────────────────────────────────────────────────────────
    # Zero out the DC component (index 0) before sorting so that the mean
    # offset does not occupy one of the top-n_peaks slots. A copy is used to
    # preserve the original magnitudes array for the return value.
    mag_work    = magnitudes.copy()
    mag_work[0] = 0.0

    # argsort ascending then reverse gives descending order; slice to n_peaks.
    peak_indices = np.argsort(mag_work)[::-1][:n_peaks]

    peak_wavelengths = []
    for idx in peak_indices:
        if idx > 0 and freqs[idx] > 0:
            # Convert frequency (cycles per world unit) to wavelength
            # (world units per cycle) by taking the reciprocal.
            peak_wavelengths.append(float(1.0 / freqs[idx]))
        else:
            # idx == 0 is the DC component (already zeroed out but may still
            # appear if all magnitudes are equal). freqs[idx] == 0 is the
            # same case. Either way, no valid wavelength can be assigned.
            peak_wavelengths.append(None)

    # Pad the list to exactly n_peaks entries in case the spectrum had fewer
    # valid (non-DC, positive-frequency) bins than requested.
    while len(peak_wavelengths) < n_peaks:
        peak_wavelengths.append(None)

    return peak_wavelengths, freqs.tolist(), magnitudes.tolist()
