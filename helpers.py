"""
helpers.py — Pure geometry/math utilities used by multiple modules.

Contains coordinate smoothing, decimation, pixel-to-world conversion,
multipolygon explosion, small-feature filtering, and topology reconnection.
"""

import copy
import math
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found — centreline smoothing unavailable.")
    print("         Install with:  pip install scipy")

import config
from data_models import Branch, Feature


def _smooth_coords(coords, resolution=None):
    """
    Apply Gaussian smoothing to a sequence of 2-D coordinates.

    The x and y coordinate arrays are smoothed independently using
    ``scipy.ndimage.gaussian_filter1d``. The global ``SMOOTHING`` constant
    is interpreted as a sigma in *world units*, which is converted to
    *pixel units* (array indices) by dividing by the raster resolution before
    being passed to the filter. This makes the smoothing behaviour invariant
    to ``RASTER_RESOLUTION``: a ``SMOOTHING`` value of 0.5 always blurs over
    roughly half a world unit regardless of pixel density, so changing the
    raster resolution does not inadvertently tighten or loosen the centreline.

    Smoothing is skipped entirely if scipy is unavailable or if ``SMOOTHING``
    is zero or negative, in which case the original coordinates are returned
    unchanged.

    Args:
        coords (list[tuple[float, float]]): Ordered sequence of ``(x, y)``
            world coordinate pairs to smooth. Must contain at least two points;
            single-point or empty inputs are returned as-is.
        resolution (float | None): The raster cell size in world units used
            to convert the world-space sigma to pixel-space. Falls back to
            the global ``RASTER_RESOLUTION`` constant if ``None``.

    Returns:
        list[tuple[float, float]]: A new coordinate list of the same length
        as ``coords``, with x and y values replaced by their Gaussian-smoothed
        equivalents. The list length is always preserved by
        ``gaussian_filter1d`` (no padding or truncation occurs).
    """
    # Return early for degenerate inputs that cannot be meaningfully smoothed.
    if len(coords) < 2:
        return coords

    # Split the (x, y) pairs into separate arrays for independent 1-D
    # smoothing. Treating x and y as separate signals is correct here because
    # the Gaussian kernel should act along the arc-length dimension (the array
    # index), not across the two spatial dimensions simultaneously.
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])

    if HAS_SCIPY and config.SMOOTHING > 0:
        res = resolution if resolution is not None else config.RASTER_RESOLUTION
        # Convert world-unit sigma to array-index sigma. One array index
        # corresponds to one raster pixel, which is approximately one
        # resolution unit in world space, so dividing by res gives the
        # correct scale factor.
        sigma_px = config.SMOOTHING / res if res > 0 else config.SMOOTHING
        xs = gaussian_filter1d(xs, sigma=sigma_px)
        ys = gaussian_filter1d(ys, sigma=sigma_px)

    return list(zip(xs.tolist(), ys.tolist()))


def _decimate_coords_with_uniform_spacing(coords, min_spacing):
    """
    Reduce the vertex count of a polyline by uniform arc-length resampling.

    Walks the polyline in order and emits a new output vertex each time the
    accumulated arc-length since the last emitted vertex reaches
    ``min_spacing``. This is the correct approach for decimating a
    smooth curve approximated by many short line segments: it places output
    vertices at equal spacing along the *curve's arc length* rather than
    thinning based on Euclidean proximity, so tight bends are handled
    correctly without under-sampling at high-curvature sections.

    The first and last vertices of the input are always preserved so that
    branch endpoints remain at their original positions and topological
    connections are not disturbed.

    Args:
        coords (list[tuple[float, float]]): Ordered sequence of ``(x, y)``
            world coordinate pairs forming the polyline to decimate. Must
            contain at least two points; shorter inputs are returned unchanged.
        min_spacing (float): Minimum arc-length distance between consecutive
            output vertices, in the same units as the coordinates. Pass
            ``0`` or a negative value to disable decimation and return the
            input unchanged.

    Returns:
        list[tuple[float, float]]: A decimated coordinate list. Always
        contains at least the first and last points of the input. Intermediate
        vertices are included only when the accumulated arc-length since the
        previous emitted vertex reaches ``min_spacing``.
    """
    if min_spacing <= 0 or len(coords) < 2:
        return coords

    # The first vertex is always emitted; subsequent vertices are added only
    # when sufficient arc-length has accumulated since the previous emission.
    result      = [coords[0]]
    accumulated = 0.0
    prev        = coords[0]

    for pt in coords[1:]:
        # Accumulate the Euclidean distance from the previous vertex.
        dx = pt[0] - prev[0]
        dy = pt[1] - prev[1]
        accumulated += math.sqrt(dx * dx + dy * dy)
        # Advance prev unconditionally so the arc-length measurement is
        # always relative to the actual polyline path, not only to the
        # last *emitted* vertex. This prevents over-densification in
        # sections where many short segments make up a longer arc.
        prev = pt

        if accumulated >= min_spacing:
            result.append(pt)
            # Reset accumulator: the next segment is measured from this
            # newly emitted vertex, not from the running total.
            accumulated = 0.0

    # Guarantee that the original endpoint is always present. If the final
    # pt triggered the spacing threshold exactly, it was already appended
    # in the loop above; the check avoids duplicating it.
    if result[-1] != coords[-1]:
        result.append(coords[-1])

    return result


def _decimate_coords_rdp(coords, epsilon):
    """
    Reduce the vertex count of a polyline using the Ramer–Douglas–Peucker
    (RDP) algorithm.

    RDP is a shape-preserving decimation algorithm: it retains a vertex only
    when it lies further than ``epsilon`` from the straight line connecting
    the current sub-range's start and end. This makes it adaptive — vertices
    are kept in high-curvature regions and discarded in straight sections —
    producing a smaller output than uniform arc-length resampling for
    piecewise-linear curves.

    The algorithm is recursive:

    1. Consider the segment from ``coords[start]`` to ``coords[end]``.
    2. Find the intermediate point with the greatest perpendicular distance
       to that segment.
    3. If that distance exceeds ``epsilon``, keep the point and recurse on
       both sub-ranges ``[start, max_idx]`` and ``[max_idx, end]``.
    4. Otherwise, discard all intermediate points in ``[start, end]``.

    The first and last points of the input are always retained.

    Args:
        coords (list[tuple[float, float]]): Ordered sequence of ``(x, y)``
            world coordinate pairs. Must contain at least three points to
            trigger simplification; shorter inputs are returned unchanged.
        epsilon (float): Maximum allowable perpendicular deviation of a
            discarded vertex from the simplified line, in the same units as
            the coordinates. Larger values produce more aggressive simplification.
            Pass ``0`` or a negative value to disable RDP and return the
            input unchanged.

    Returns:
        list[tuple[float, float]]: A simplified coordinate list that is a
        subset of the input. Always retains the first and last points. The
        simplified polyline deviates from the original by at most ``epsilon``
        at every discarded vertex.
    """
    if epsilon <= 0 or len(coords) < 3:
        return coords

    def _perpendicular_distance(pt, line_start, line_end):
        """
        Compute the perpendicular distance from a point to a line segment.

        Uses the triangle-area formula: the area of the triangle formed by
        ``pt``, ``line_start``, and ``line_end`` equals half the base length
        times the height, where the height is the perpendicular distance we
        want. Rearranging gives::

            distance = |cross(line_vec, start_to_pt)| / |line_vec|

        For degenerate segments where start and end are the same point, falls
        back to the direct Euclidean distance from ``pt`` to ``line_start``.

        Args:
            pt (tuple[float, float]): The point to measure from.
            line_start (tuple[float, float]): Start of the reference segment.
            line_end (tuple[float, float]): End of the reference segment.

        Returns:
            float: The perpendicular distance in the same units as the
            input coordinates.
        """
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]

        if dx == 0 and dy == 0:
            # Degenerate segment: start == end. Return the straight-line
            # distance from pt to the single point instead.
            dx = pt[0] - line_start[0]
            dy = pt[1] - line_start[1]
            return math.sqrt(dx * dx + dy * dy)

        # The numerator is the absolute value of the 2-D cross product of the
        # line vector (dx, dy) and the vector from line_start to pt. This
        # equals twice the area of the triangle; dividing by the base length
        # (the denominator) gives the perpendicular height.
        return abs(
            dy * pt[0] - dx * pt[1]
            + line_end[0] * line_start[1]
            - line_end[1] * line_start[0]
        ) / math.sqrt(dx * dx + dy * dy)

    def _rdp(coords, start, end):
        """
        Recursively simplify the sub-range ``coords[start:end+1]`` and return
        the indices of the vertices that should be retained.

        Base case: a range of zero or one intermediate vertex (``end <=
        start + 1``) is already maximally simplified; return just the two
        endpoint indices.

        Recursive case: find the vertex in ``(start, end)`` with the greatest
        perpendicular distance to the segment ``coords[start]``–``coords[end]``.
        If that distance exceeds ``epsilon``, keep the vertex and recurse on
        both sides. Otherwise, discard all intermediate vertices.

        The left and right recursive results share the index ``max_idx``;
        the overlap is removed by dropping the last element of the left result
        before concatenating.

        Args:
            coords (list[tuple[float, float]]): The full coordinate list
                (passed by reference; never modified).
            start (int): Inclusive start index of the current sub-range.
            end (int): Inclusive end index of the current sub-range.

        Returns:
            list[int]: Ordered list of indices into ``coords`` that should
            be kept for this sub-range, always including ``start`` and ``end``.
        """
        # Base case: one or zero intermediate points — nothing to simplify.
        if end <= start + 1:
            return [start, end]

        # Scan all intermediate points and record the one with the greatest
        # perpendicular deviation from the start-to-end segment.
        max_dist = 0.0
        max_idx  = start
        for i in range(start + 1, end):
            d = _perpendicular_distance(coords[i], coords[start], coords[end])
            if d > max_dist:
                max_dist = d
                max_idx  = i

        if max_dist > epsilon:
            # The farthest point exceeds the tolerance — keep it and recurse
            # on the two sub-ranges it creates.
            left  = _rdp(coords, start, max_idx)
            right = _rdp(coords, max_idx, end)
            # Both sub-results include max_idx (as the last element of left
            # and the first element of right). Remove the duplicate before
            # concatenating to avoid repeating that vertex in the output.
            return left[:-1] + right
        else:
            # All intermediate points are within epsilon of the straight line —
            # discard them entirely and keep only the two endpoints.
            return [start, end]

    # Run RDP over the full coordinate range and map the retained indices back
    # to their original coordinate tuples.
    indices = _rdp(coords, 0, len(coords) - 1)
    return [coords[i] for i in indices]


def _explode_multipolygons(features):
    """
    Ensure that every feature in the list contains exactly one simple Polygon.

    Called only when ``SEPARATE_MULTIPOLYGONS = True`` (the default).  When
    ``SEPARATE_MULTIPOLYGONS = False``, this function is skipped and compound
    features are kept intact; :func:`_dispatch_skeletoniser` applies a
    half-pixel outward buffer to the whole (multi-part) polygon before
    rasterisation, inflating and merging any zero-width connections so the
    skeleton remains connected across them.

    When ``SEPARATE_MULTIPOLYGONS = True``, this function runs *before* the
    per-feature rasterisation buffer in :func:`_dispatch_skeletoniser`, so the
    separation is genuine: zero-width connections become gaps between
    independent features, and the subsequent per-feature buffer expands each
    component independently without re-bridging them.

    Some upstream operations can yield ``MultiPolygon`` geometries in a
    feature's ``polygon`` field.  The most common causes are:

    - Shapely's ``buffer(0)`` repair splitting a self-intersecting ring into
      two or more disjoint polygons.
    - A parser directly returning a compound geometry for a complex shape.

    This function detects those cases and replaces each ``MultiPolygon``
    feature with one feature per component polygon, preserving all other
    fields (e.g. ``id``, ``flip_y``) via a shallow copy.  Component
    features are given IDs suffixed ``_0``, ``_1``, … to keep them
    traceable back to their source feature.  Simple ``Polygon`` features are
    passed through unchanged.

    Degenerate ``MultiPolygon`` components (empty geometries or non-Polygon
    members) are silently dropped; if all parts are degenerate the entire
    feature is discarded.

    Args:
        features (list[Feature]): Feature dataclass instances, each expected
            to have at least a ``polygon`` field and an ``id`` field.
            All other fields are preserved unchanged.

    Returns:
        list[Feature]: A new list of Feature instances in which every
        ``polygon`` field is a ``shapely.geometry.Polygon``. The order of
        simple-polygon features is preserved; exploded parts appear in the
        position of their source feature, in part-index order.
    """
    exploded = []
    for feat in features:
        geom = feat.polygon

        if isinstance(geom, MultiPolygon):
            # Filter out any non-Polygon or empty parts before enumerating,
            # so that part indices in the new IDs correspond only to valid,
            # usable geometries.
            parts = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]

            if not parts:
                # Every component was degenerate; drop the feature entirely
                # rather than emitting a feature with no geometry.
                continue

            for i, part in enumerate(parts):
                # Shallow-copy the source Feature so that all non-geometry fields
                # (e.g. metadata added by the caller) are inherited without
                # needing to know what attributes are present.
                new_feat = copy.copy(feat)
                new_feat.polygon = part
                # Suffix the ID so the origin feature can be identified while
                # still keeping all part IDs unique within the output list.
                new_feat.id = f"{feat.id}_{i}"
                exploded.append(new_feat)

            print(f"  [explode] Feature {feat.id} split into {len(parts)} polygon(s).")
        else:
            # Simple Polygon — pass through as-is without copying.
            exploded.append(feat)

    return exploded


def _filter_small_features(features):
    """
    Drop features whose polygon bounding box area is smaller than
    ``MINIMUM_FEATURE_SIZE²`` and return the survivors with sequential IDs.

    This filter runs before rasterisation, so dropped features incur no
    compute cost. It is intended to remove artefactual slivers — such as
    near-zero-width swells at the ends of fractures — that would otherwise
    produce degenerate or misleading skeletons.

    A notice is printed for each dropped feature showing its original ID and
    bounding box area.  Surviving features are renumbered from 1 upward so
    that the output IDs form a contiguous sequence with no gaps.

    If ``MINIMUM_FEATURE_SIZE`` is zero or negative the function returns
    ``features`` unchanged.

    Args:
        features (list[Feature]): Parsed and exploded feature list.

    Returns:
        list[Feature]: Surviving features with reassigned sequential IDs.
    """
    if config.MINIMUM_FEATURE_SIZE > 0:
        threshold = config.MINIMUM_FEATURE_SIZE ** 2
        kept = []

        for feat in features:
            if isinstance(feat.polygon, (LineString, MultiLineString)):
                # For line features, use arc length (not bounding box area)
                geom = feat.polygon
                if isinstance(geom, MultiLineString):
                    line_length = sum(line.length for line in geom.geoms)
                else:
                    line_length = geom.length
                if line_length < config.MINIMUM_FEATURE_SIZE:
                    print(f"  [filter] Dropped line feature {feat.id!r} "
                          f"(length {line_length:.4g} < {config.MINIMUM_FEATURE_SIZE:.4g})")
                else:
                    kept.append(feat)
            else:
                minx, miny, maxx, maxy = feat.polygon.bounds
                bbox_area = (maxx - minx) * (maxy - miny)
                if bbox_area < threshold:
                    print(f"  [filter] Dropped feature {feat.id!r} "
                          f"(bbox area {bbox_area:.4g} < {threshold:.4g})")
                else:
                    kept.append(feat)
    else:
        kept = features

    # Always reassign sequential 1-based IDs so that output numbering is
    # consistent regardless of the parser's internal counter (which may be
    # 0-indexed or use non-numeric strings) and regardless of whether any
    # features were actually filtered.  This guarantees that file names
    # (feature_1_..., feature_2_...) always match the feature_id column in
    # summary.csv with no gaps or offset.
    for new_id, feat in enumerate(kept, start=1):
        feat.id = str(new_id)

    return kept


def _pixels_to_world(pixels, x_min, y_min, resolution=None):
    """
    Convert a list of raster pixel coordinates to world-space coordinates.

    Applies the inverse of the rasterisation grid mapping used in
    :func:`_rasterise_polygon`: a pixel at ``(row, col)`` maps to the world
    point ``(x_min + col * resolution, y_min + row * resolution)``.

    Note that rows correspond to the y-axis and columns to the x-axis, so
    the row and column indices are swapped in the output tuples.

    Args:
        pixels (list[tuple[int, int]]): Ordered list of ``(row, col)`` integer
            pixel coordinates, as produced by :func:`_extract_branches` or
            :func:`_find_diameter_path`.
        x_min (float): World x-coordinate of the left edge of pixel column 0,
            as returned by :func:`_rasterise_polygon`.
        y_min (float): World y-coordinate of the top edge of pixel row 0,
            as returned by :func:`_rasterise_polygon`.
        resolution (float | None): Raster cell size in world units. If
            ``None``, falls back to the global ``RASTER_RESOLUTION`` constant.

    Returns:
        list[tuple[float, float]]: World ``(x, y)`` coordinates corresponding
        to the input pixel positions, in the same order as ``pixels``.
    """
    # Fall back to the global constant when no resolution is supplied, so
    # callers that don't track per-feature resolution can omit the argument.
    res = resolution if resolution is not None else config.RASTER_RESOLUTION
    return [
        (x_min + c * res,   # column index → world x
         y_min + r * res)   # row index    → world y
        for r, c in pixels
    ]


def _reconnect_topology(features):
    """
    Restore topological connections between features whose source polygons
    intersect or touch, ensuring skeleton branches are properly connected for
    downstream analysis (e.g. fluid-flow path tracing).

    For each adjacent polygon pair the algorithm:

    1. Considers all branch endpoints from *both* features (both ends of
       every branch on each side).
    2. For each endpoint, computes its distance to the nearest point on the
       *other* feature's skeleton.
    3. Selects the single endpoint — across all candidates from both features
       — that lies closest to the other skeleton.  This is the end most
       geometrically in need of extension.  A small branch-ID penalty biases
       the search toward spine (branch 0) endpoints over side-branch tips,
       which tend to terminate near the polygon wall and can otherwise win the
       distance comparison when a spine tip is the intended target.
    4. Extends that endpoint by appending (or prepending) the nearest point on
       the other feature's skeleton, adding one straight connecting segment.
       Only the source endpoint is extended — this intentionally supports
       T-junction topology where one feature's tip meets the body of another.

    Adjacency is detected in two stages: a fast STR-tree bounding-box
    pre-filter followed by an exact ``polygon.distance()`` check, so only
    pairs that genuinely intersect or lie within ``CONNECT_BUFFER`` of each
    other are processed.  Each pair is extended at most once.

    Args:
        features (list[Feature]): Feature objects as produced by a parser
            and :func:`_explode_multipolygons`. Each must have:

            - ``polygon``: The source polygon or line geometry.
            - ``branches`` (*list[Branch]*): Branch objects produced by
              :func:`_dispatch_skeletoniser`, each with an ``output_coords``
              attribute holding the decimated ``(x, y)`` world coordinate list
              used for SVG export.

    Returns:
        int: Number of endpoint extensions applied.
    """
    # Maximum gap between two polygon footprints for them to be considered
    # adjacent. Using max(SMOOTHING, ...) scales with the smoothing strength,
    # accommodating the small pull-back Gaussian smoothing applies near branch
    # endpoints.
    CONNECT_BUFFER = max(config.SMOOTHING, config.RASTER_RESOLUTION * 3)

    # Tiny distance penalty added per unit of branch ID to break ties in
    # favour of the main spine (branch 0) over side branches.  Side branches
    # terminate near the polygon wall and can otherwise appear closer to the
    # neighbouring feature than the spine tip.  The penalty is negligible
    # compared to real spatial distances but deterministic.
    BRANCH_ID_PENALTY = 1e-6

    polys = [feat.polygon for feat in features]
    tree  = STRtree(polys)   # O(log n) bounding-box queries

    n_extensions = 0
    processed    = set()     # (min_i, max_j) pairs already handled

    for i, feat_a in enumerate(features):
        if not feat_a.branches:
            continue

        poly_a     = polys[i]
        poly_a_buf = poly_a.buffer(CONNECT_BUFFER)

        for j in tree.query(poly_a_buf):
            if j == i:
                continue

            pair = (min(i, j), max(i, j))
            if pair in processed:
                continue
            processed.add(pair)

            feat_b = features[j]
            if not feat_b.branches:
                continue

            # Exact adjacency: polygons must intersect or be within
            # CONNECT_BUFFER (the STRtree only tests bounding boxes).
            if polys[i].distance(polys[j]) > CONNECT_BUFFER:
                continue

            # Pre-build LineStrings from output_coords (the decimated export
            # path) for both skeletons.  Using output_coords rather than
            # centerline ensures that the proximity search operates on the same
            # coordinate set that will be written to skeleton.svg, and that the
            # straight connecting segment is added only to the export geometry —
            # leaving centerline (smoothed dense path) and sample_points (width
            # measurement locations) completely undisturbed.
            lines_a = [(b, LineString(b.output_coords))
                       for b in feat_a.branches if len(b.output_coords) >= 2]
            lines_b = [(b, LineString(b.output_coords))
                       for b in feat_b.branches if len(b.output_coords) >= 2]

            if not lines_a or not lines_b:
                continue

            # Find the single endpoint — across all endpoints of both features
            # — that is closest to the other feature's skeleton.  That is the
            # end to extend, and the projection onto the other skeleton is its
            # target.  A small branch-ID penalty biases toward spine endpoints.
            best_dist      = float("inf")
            best_ep_branch = None
            best_ep_tail   = None
            best_target_xy = None

            for src_lines, dst_lines in [(lines_a, lines_b), (lines_b, lines_a)]:
                for branch, _ in src_lines:
                    oc = branch.output_coords
                    for coord, is_tail in [(oc[0], False), (oc[-1], True)]:
                        ep_pt = Point(coord)
                        for _, dst_line in dst_lines:
                            proj = dst_line.interpolate(dst_line.project(ep_pt))
                            d    = ep_pt.distance(proj)
                            d   += branch.id * BRANCH_ID_PENALTY
                            if d < best_dist:
                                best_dist      = d
                                best_ep_branch = branch
                                best_ep_tail   = is_tail
                                best_target_xy = (proj.x, proj.y)

            # Skip if no valid candidate was found or the extension would be
            # trivially zero-length (endpoint already lies on the other skeleton).
            if best_ep_branch is None or best_dist < 1e-9:
                continue

            # Append or prepend the target point to extend only output_coords.
            # centerline and sample_points are intentionally left unchanged —
            # the connecting segment is a topological export artefact, not part
            # of the physical measurement geometry.
            output_coords = best_ep_branch.output_coords
            if best_ep_tail:
                best_ep_branch.output_coords = output_coords + [best_target_xy]
            else:
                best_ep_branch.output_coords = [best_target_xy] + output_coords
            n_extensions += 1

    return n_extensions
