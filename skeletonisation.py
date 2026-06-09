"""
skeletonisation.py — Polygon-to-skeleton pipeline.

Everything that turns a Feature's polygon into skeleton branches:
rasterisation, Lee thinning, branch extraction, directional skeletonisation,
pruning, post-processing, and topology snapping.
"""

import math
from collections import deque, defaultdict

import numpy as np
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon, Polygon, Point
)

import config
from config import (
    RASTER_RESOLUTION, RASTER_BUFFER, MAX_RASTER_PIXELS, MIN_RASTER_PIXELS,
    SMOOTHING, OUTPUT_RESOLUTION, RDP_EPSILON, SAMPLING_INTERVAL,
    SKELETONISATION_METHOD, CURVATURE_THRESHOLD, SOLIDITY_THRESHOLD,
    ASPECT_RATIO_THRESHOLD, ESCAPE_THRESHOLD, BRANCHING_THRESHOLD,
    SEPARATE_MULTIPOLYGONS, MIN_BRANCH_PIXELS, MIN_BRANCH_PERCENT,
    EXPORT_SKELETON_OVERLAY,
)
from data_models import Branch, Feature
from helpers import (
    _smooth_coords, _decimate_coords_with_uniform_spacing, _decimate_coords_rdp,
    _pixels_to_world,
)

# --- optional dependencies ---------------------------------------------------

try:    # scikit-image
    from skimage.morphology import skeletonize as _skimage_skeletonize
    from skimage.draw import polygon as _skimage_draw_polygon
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found — skeletonisation unavailable.")
    print("         Install with:  pip install scikit-image")

try:    # scipy
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found — centreline smoothing unavailable.")
    print("         Install with:  pip install scipy")


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _is_line_feature(feature):
    """Return True if the feature's geometry is a line/polyline rather than a polygon."""
    return isinstance(feature.polygon, (LineString, MultiLineString))


def _polygon_pca(polygon):
    """
    Compute a Principal Component Analysis (PCA) decomposition of a polygon's
    exterior ring vertices and return the results needed by both
    :func:`_geometry_from_polygon` and :func:`_directional_skeleton`.

    PCA (Principal Component Analysis) treats each exterior vertex as a sample
    point and finds the eigenvectors of the 2 × 2 covariance matrix. The
    eigenvector corresponding to the largest eigenvalue points along the axis
    of greatest spatial variance — the polygon's long axis.

    Args:
        polygon (shapely.geometry.Polygon): The polygon to decompose.

    Returns:
        tuple | None:
            ``(coords, center, C, long_vec, perp_vec, eigvals)`` where:

            - ``coords`` (*numpy.ndarray*, shape N×2): Exterior ring vertices.
            - ``center`` (*numpy.ndarray*, shape 2): Mean vertex position.
            - ``C`` (*numpy.ndarray*, shape N×2): Mean-centred vertices.
            - ``long_vec`` (*numpy.ndarray*, shape 2): Unit vector along the
              long (principal) axis.
            - ``perp_vec`` (*numpy.ndarray*, shape 2): Unit vector perpendicular
              to ``long_vec``, rotated 90° counter-clockwise.
            - ``eigvals`` (*numpy.ndarray*, shape 2): Eigenvalues in ascending
              order (smallest first).

            Returns ``None`` if the exterior ring has fewer than two vertices.
    """
    # MultiPolygon has no single .exterior ring.  Use the convex hull of the
    # whole geometry, which is always a simple Polygon and gives a good
    # approximation of the overall orientation and long-axis extent.
    source = polygon.convex_hull if not isinstance(polygon, Polygon) else polygon
    if source.is_empty or not hasattr(source, "exterior"):
        return None

    coords = np.array(source.exterior.coords)
    if len(coords) < 2:
        return None

    center = coords.mean(axis=0)
    C      = coords - center
    cov    = (C.T @ C) / max(len(coords) - 1, 1)

    # eigh is used instead of eig because the covariance matrix is symmetric
    # positive semi-definite; eigh is faster and guarantees real eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(cov)

    long_vec = eigvecs[:, np.argmax(eigvals)]          # long axis (max variance)
    perp_vec = np.array([-long_vec[1], long_vec[0]])   # perpendicular, 90° CCW

    return coords, center, C, long_vec, perp_vec, eigvals


def _geometry_from_polygon(polygon):
    """
    Compute the orientation and long-axis length of a polygon (or line) using
    Principal Component Analysis (PCA) on its exterior ring (or line) vertices.

    For line features (LineString / MultiLineString), orientation is computed
    via PCA on the line's coordinate vertices, and long_axis_length is the
    total arc length of the line.

    PCA on the boundary vertices treats each vertex as a sample point and
    finds the eigenvectors of the covariance matrix. The eigenvector
    corresponding to the largest eigenvalue points along the axis of greatest
    spatial variance — which for an elongated polygon is its long axis.

    The orientation is normalised to the range ``[0, 180)`` degrees so that
    opposite directions (e.g. 30° and 210°) are treated as equivalent. This
    matches the geological convention for fracture strike, where direction is
    unsigned.

    The long-axis length is the total extent of the boundary vertices when
    projected onto the principal axis vector — i.e. the difference between the
    maximum and minimum projections.

    Args:
        polygon (shapely.geometry.Polygon | shapely.geometry.LineString |
                 shapely.geometry.MultiLineString): The geometry to analyse.

    Returns:
        tuple[float | None, float | None]:
            - **orientation_deg** (*float*): The angle of the long axis in
              degrees, measured counter-clockwise from the positive x-axis,
              in the range ``[0, 180)``. ``None`` if insufficient vertices.
            - **long_axis_length** (*float*): The projected extent along the
              principal axis (polygon), or total arc length (line). ``None``
              if insufficient vertices.
    """
    # For line features: use PCA on line coords, arc length as long_axis_length
    if isinstance(polygon, (LineString, MultiLineString)):
        try:
            if isinstance(polygon, MultiLineString):
                all_coords = [c for line in polygon.geoms for c in line.coords]
            else:
                all_coords = list(polygon.coords)
            if len(all_coords) < 2:
                return None, None
            pts = np.array(all_coords, dtype=float)
            center = pts.mean(axis=0)
            C = pts - center
            cov = (C.T @ C) / max(len(pts) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            long_vec = eigvecs[:, np.argmax(eigvals)]
            angle_rad = np.arctan2(float(long_vec[1]), float(long_vec[0]))
            orientation_deg = float(np.degrees(angle_rad)) % 180.0
            # Long axis length = total arc length of the line
            diffs = np.diff(pts, axis=0)
            long_axis_length = float(np.sum(np.sqrt((diffs**2).sum(axis=1))))
            return orientation_deg, long_axis_length
        except Exception:
            return None, None

    pca = _polygon_pca(polygon)
    if pca is None:
        return None, None

    _coords, _center, C, long_vec, _perp_vec, _eigvals = pca

    # atan2 gives the signed angle in (-π, π]; the modulo normalises to [0°, 180°).
    angle_rad       = np.arctan2(float(long_vec[1]), float(long_vec[0]))
    orientation_deg = float(np.degrees(angle_rad)) % 180.0

    # Project all boundary vertices onto the long axis and measure the span.
    proj             = C @ long_vec
    long_axis_length = float(proj.max() - proj.min())

    return orientation_deg, long_axis_length


# =============================================================================
# RASTERISATION
# =============================================================================

def _rasterise_polygon(polygon):
    """
    Rasterise a Shapely ``Polygon`` or ``MultiPolygon`` into a boolean NumPy
    array suitable for morphological skeletonisation.

    The polygon is rasterised in isolation within a bounding-box grid padded
    by five raster cells on each side, so that the skeleton cannot touch the
    array boundary and its topology is cleanly self-contained.

    **Resolution management** — two safety mechanisms prevent memory
    exhaustion on large or complex polygons:

    1. *Pre-flight scaling*: before allocating the grid, the function checks
       whether the nominal ``RASTER_RESOLUTION`` would exceed
       ``MAX_RASTER_PIXELS``. If so, it solves analytically for the coarsest
       resolution that fits within the cap and coarsens automatically.
    2. *MemoryError rescue loop*: even after scaling, the OS allocation can
       fail on memory-constrained systems. The loop doubles the resolution
       (halving the grid in each dimension) and retries until allocation
       succeeds.

    Interior rings (holes) in the polygon are rasterised after the exterior
    by setting their pixels to ``False``, correctly representing donut-shaped
    or compound features.

    Args:
        polygon (shapely.geometry.Polygon | shapely.geometry.MultiPolygon):
            The geometry to rasterise. ``MultiPolygon`` inputs are iterated
            part-by-part so every component is painted onto the same grid.

    Returns:
        tuple:
            - **grid** (*numpy.ndarray[bool]*): 2-D boolean array of shape
              ``(H, W)`` where ``True`` marks pixels inside the polygon.
            - **x_min** (*float*): World x-coordinate of the left edge of
              pixel column 0.
            - **y_min** (*float*): World y-coordinate of the top edge of
              pixel row 0.
            - **resolution** (*float*): The raster cell size actually used,
              in the same units as the polygon coordinates. May be larger
              than ``RASTER_RESOLUTION`` if auto-coarsening was triggered.
    """
    # Normalise both Polygon and MultiPolygon inputs to a flat list of parts
    # so that the rasterisation loop below handles both types identically.
    parts = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]

    x0, y0, x1, y1 = polygon.bounds
    resolution = config.RASTER_RESOLUTION  # effective (auto-scaled) raster resolution

    # ── Pre-flight resolution check ───────────────────────────────────────────
    # Compute the grid dimensions that RASTER_RESOLUTION would produce,
    # including the five-cell padding added later. If the total pixel count
    # exceeds MAX_RASTER_PIXELS, solve for the smallest resolution that fits.
    pad_world = 5 * resolution
    w_world   = (x1 - x0) + 2 * pad_world
    h_world   = (y1 - y0) + 2 * pad_world
    n_pixels  = (int(math.ceil(w_world / resolution)) + 1) * \
                (int(math.ceil(h_world / resolution)) + 1)

    if MIN_RASTER_PIXELS > 0 and n_pixels < MIN_RASTER_PIXELS:
        # Derivation: n ≈ (w/r) * (h/r)  →  r = sqrt(w*h / n_min)
        resolution = math.sqrt(w_world * h_world / MIN_RASTER_PIXELS)
        raw_r = config.RASTER_RESOLUTION
        print(f"    ⚠ RASTER_RESOLUTION={raw_r} would create a "
              f"{int(math.ceil(h_world/raw_r))+1}×{int(math.ceil(w_world/raw_r))+1} grid "
              f"({n_pixels} pixels). "
              f"Auto-scaling to {resolution:.6g} to meet "
              f"{MIN_RASTER_PIXELS:,} pixel minimum.")
        # Recompute grid dimensions with the refined resolution so the MAX
        # check below always uses consistent, up-to-date values.
        pad_world = 5 * resolution
        w_world   = (x1 - x0) + 2 * pad_world
        h_world   = (y1 - y0) + 2 * pad_world
        n_pixels  = (int(math.ceil(w_world / resolution)) + 1) * \
                    (int(math.ceil(h_world / resolution)) + 1)

    if n_pixels > MAX_RASTER_PIXELS:
        # Derivation: n ≈ (w/r) * (h/r)  →  r = sqrt(w*h / n_max)
        resolution = math.sqrt(w_world * h_world / MAX_RASTER_PIXELS)
        raw_r = config.RASTER_RESOLUTION
        print(f"    ⚠ RASTER_RESOLUTION={raw_r} would create a "
              f"{int(math.ceil(h_world/raw_r))+1}×{int(math.ceil(w_world/raw_r))+1} grid "
              f"({n_pixels/1e6:.1f} M pixels). "
              f"Auto-scaling to {resolution:.6g} to stay within "
              f"{MAX_RASTER_PIXELS//1_000_000} M pixel cap.")

    # ── Memory-safe grid allocation ───────────────────────────────────────────
    # Retry with doubled resolution on MemoryError. Each doubling quarters the
    # grid area, so convergence is fast even from very large initial sizes.
    while True:
        pad   = 5 * resolution
        x_min = x0 - pad
        y_min = y0 - pad
        x_max = x1 + pad
        y_max = y1 + pad
        W     = int(math.ceil((x_max - x_min) / resolution)) + 1
        H     = int(math.ceil((y_max - y_min) / resolution)) + 1

        try:
            grid = np.zeros((H, W), dtype=bool)
            break   # allocation succeeded — exit the retry loop
        except MemoryError:
            resolution *= 2.0
            print(f"    ⚠ MemoryError allocating {H}×{W} grid — "
                  f"doubling resolution to {resolution:.6g} and retrying.")

    def ring_to_px(ring):
        """
        Convert a sequence of world ``(x, y)`` coordinates to pixel-space
        column (``px``) and row (``py``) arrays.

        The mapping is a simple linear scale: pixel index =
        ``(world_coord - grid_origin) / resolution``. Floating-point pixel
        coordinates are intentional — ``skimage.draw.polygon`` accepts them
        and interpolates sub-pixel boundaries automatically.

        Args:
            ring (list[tuple[float, float]]): An ordered list of world
                ``(x, y)`` coordinate pairs forming a closed or open ring,
                as returned by ``shapely.geometry.polygon.LinearRing.coords``.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                - **px** (*ndarray[float]*): Column indices (x direction).
                - **py** (*ndarray[float]*): Row indices (y direction).
        """
        px = np.array([(x - x_min) / resolution for x, y in ring])
        py = np.array([(y - y_min) / resolution for x, y in ring])
        return px, py

    # ── Rasterise each polygon part ───────────────────────────────────────────
    for part in parts:
        if part.is_empty or not hasattr(part, "exterior"):
            continue

        # Paint the exterior ring as filled (True).
        px, py = ring_to_px(list(part.exterior.coords))
        try:
            rr, cc = _skimage_draw_polygon(py, px, shape=(H, W))
            grid[rr, cc] = True
        except MemoryError:
            # Guard against the extremely unlikely case where draw_polygon's
            # internal temporary arrays trigger a MemoryError even though the
            # grid itself allocated successfully. Skip the part rather than
            # crashing the entire run.
            pass

        # Punch out each interior ring (hole) by setting its pixels to False.
        # Interior coords are processed after the exterior so that the fill
        # order is correct regardless of winding direction.
        for interior in part.interiors:
            px, py = ring_to_px(list(interior.coords))
            try:
                rr, cc = _skimage_draw_polygon(py, px, shape=(H, W))
                grid[rr, cc] = False
            except MemoryError:
                pass

    return grid, x_min, y_min, resolution


def _add_rasterisation_buffer(polygon, feature_id):
    """
    Apply a configurable outward buffer to a polygon for use during rasterisation.

    An outward buffer of ``RASTER_BUFFER`` world units is applied to the local
    rasterisation polygon so that thin features (and thin necks within wider
    features) always occupy at least one raster pixel in cross-section and
    therefore survive morphological thinning.  Without this, any cross-section
    narrower than one pixel is quantised to zero pixels and lost, disconnecting
    the skeleton.  Increase ``RASTER_BUFFER`` if geometry is still being lost.

    This function operates on a LOCAL copy of the polygon and must never be
    called with ``feature.polygon`` as its target — the original geometry is
    intentionally preserved on the feature for two reasons:

    1. Width measurement (:func:`_find_partial_thickness`) intersects
       perpendicular rays with ``feature.polygon`` via Shapely, so measured
       widths reflect the true input geometry regardless of the rasterisation
       buffer.
    2. The directional-mode centreline (:func:`_directional_skeleton`) is
       computed by intersecting cross-section lines with ``feature.polygon``
       directly, so centreline positions are also unaffected.  The existing
       ``_escape_check`` in :func:`_validate_directional_skeleton` already
       trims any endpoint overruns that extend beyond the original polygon
       boundary.

    For ``SEPARATE_MULTIPOLYGONS = True`` (the default): each feature is
    already a single ``Polygon`` by the time this is called (separated by
    :func:`_explode_multipolygons` before the processing loop), so the buffer
    only expands individual features outward without bridging independent
    neighbours.

    For ``SEPARATE_MULTIPOLYGONS = False``: the buffer additionally bridges
    zero-width connections between kept-together ``MultiPolygon`` components:

    - **Pinch point** — walls touch at a single point.
    - **Coincident section** — walls coincide over a finite length.
    - **MultiPolygon gap** — components share a boundary but no area.

    Shapely automatically merges the overlapping dilated discs/strips into a
    single ``Polygon``.  If the result is still a ``MultiPolygon`` the gap is
    wider than half a pixel and the skeleton may remain disconnected.

    Args:
        polygon (shapely.geometry.Polygon | shapely.geometry.MultiPolygon):
            The rasterisation polygon to expand (a local copy of
            ``feature.polygon``, never the original).
        feature_id (str): Feature ID used in the disconnect warning message.

    Returns:
        shapely.geometry.Polygon | shapely.geometry.MultiPolygon: The buffered
        polygon. For ``SEPARATE_MULTIPOLYGONS = True`` this is always a simple
        ``Polygon``; for ``SEPARATE_MULTIPOLYGONS = False`` it is a ``Polygon``
        if the buffer successfully merged all components, or a ``MultiPolygon``
        if a genuine gap remains.
    """
    stroke_radius = config.RASTER_BUFFER
    buffered = polygon.buffer(stroke_radius, join_style="round", cap_style="round")

    if not SEPARATE_MULTIPOLYGONS and isinstance(buffered, MultiPolygon):
        print(f"    ⚠ [connect] Feature {feature_id}: components did not "
              f"merge after stroke buffer — skeleton may be disconnected. "
              f"Consider enabling SEPARATE_MULTIPOLYGONS = True.")

    return buffered


def _skeleton_overlay_grid(arr):
    """
    Conditionally retain a raster grid for the skeleton overlay export.

    Returns the array unchanged when skeleton overlay export is enabled,
    or ``None`` when it is disabled.  Storing ``None`` instead of the full
    array prevents tens or hundreds of megabytes of raster data accumulating
    in memory across many features during a parallel run.

    Args:
        arr (numpy.ndarray | None): The grid to conditionally retain.

    Returns:
        numpy.ndarray | None: ``arr`` if ``EXPORT_SKELETON_OVERLAY`` is
        truthy, otherwise ``None``.
    """
    return arr if EXPORT_SKELETON_OVERLAY else None


# =============================================================================
# BRANCH EXTRACTION AND GRAPH ALGORITHMS
# =============================================================================

def _extract_branches(skeleton):
    """
    Decompose a binary skeleton array into an ordered list of pixel paths,
    one path per topological branch.

    Each skeleton pixel is classified by the number of its 8-connected
    neighbours that are also skeleton pixels:

    - **Endpoint** (1 neighbour): a tip of a branch arm.
    - **Normal** (exactly 2 neighbours): an interior point of a branch with
      no bifurcation.
    - **Junction** (3 or more neighbours): a point where two or more branches
      meet.

    Branches are then traced by walking from each endpoint or junction along
    normal pixels until the next endpoint or junction is reached. Each
    endpoint–junction or junction–junction segment becomes one branch. Every
    pixel edge is recorded in ``visited_edges`` so that the same segment is
    never traced twice even when a junction is shared by multiple branches.

    Isolated loops (which have no endpoints) are handled by picking an
    arbitrary skeleton pixel as the start node, ensuring the entire loop is
    captured as a single closed branch.

    Args:
        skeleton (numpy.ndarray[bool]): 2-D boolean array produced by
            morphological skeletonisation, where ``True`` marks skeleton
            pixels.

    Returns:
        list[list[tuple[int, int]]]: A list of branches. Each branch is an
        ordered list of ``(row, col)`` integer tuples tracing a path from
        one endpoint or junction to the next. The list is empty if the
        skeleton contains no pixels.
    """
    rows, cols = np.where(skeleton)
    if len(rows) == 0:
        return []

    # Build a set of all skeleton pixel coordinates for O(1) membership tests
    # during neighbour lookups. A set is substantially faster than repeatedly
    # indexing the 2-D array for large skeletons.
    skel_set = set(zip(rows.tolist(), cols.tolist()))

    def nbrs(r, c):
        """
        Return the 8-connected skeleton neighbours of pixel ``(r, c)``.

        Iterates over all eight surrounding positions (including diagonals)
        and returns only those that are present in the skeleton set, excluding
        the pixel itself via the ``(dr or dc)`` guard.

        Args:
            r (int): Row index of the target pixel.
            c (int): Column index of the target pixel.

        Returns:
            list[tuple[int, int]]: List of ``(row, col)`` coordinates of
            neighbouring skeleton pixels. May be empty for an isolated pixel.
        """
        return [(r+dr, c+dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and (r+dr, c+dc) in skel_set]

    # Classify every skeleton pixel into endpoints and junctions. Normal pixels
    # (degree 2) are implicit — they are not stored separately because the
    # tracing loop advances through them without branching.
    endpoints = set()
    junctions = set()
    for r, c in skel_set:
        n = len(nbrs(r, c))
        if n == 1:
            endpoints.add((r, c))
        elif n >= 3:
            junctions.add((r, c))

    # Every branch trace must start at either an endpoint or a junction. For
    # isolated loops (which have neither), fall back to an arbitrary pixel so
    # the loop is still captured. The `or` short-circuits if the union is
    # non-empty, so the fallback is only evaluated when needed.
    start_nodes = endpoints | junctions or {next(iter(skel_set))}

    # Track which pixel edges have already been traced to prevent duplicate
    # branches. Edges are stored as sorted (min, max) pairs so that the same
    # edge is represented identically regardless of traversal direction.
    visited_edges = set()
    branches      = []

    for start in start_nodes:
        for nxt in nbrs(*start):
            # Canonical edge key: always store with the lexicographically
            # smaller coordinate first so direction doesn't create duplicates.
            ek = (min(start, nxt), max(start, nxt))
            if ek in visited_edges:
                continue

            # Begin a new branch with the first edge already committed.
            branch = [start, nxt]
            visited_edges.add(ek)
            prev, cur = start, nxt

            # Walk forward along normal pixels until we hit an endpoint,
            # junction, or a dead end (no unvisited forward neighbours).
            while cur not in endpoints and cur not in junctions:
                # Exclude the pixel we just came from to avoid backtracking.
                candidates = [n for n in nbrs(*cur) if n != prev]
                if not candidates:
                    break   # isolated tip — the branch has naturally ended
                nxt2 = candidates[0]
                ek2  = (min(cur, nxt2), max(cur, nxt2))
                if ek2 in visited_edges:
                    break   # loop detected — stop to avoid infinite cycling
                visited_edges.add(ek2)
                branch.append(nxt2)
                prev, cur = cur, nxt2

            branches.append(branch)

    return branches


def _find_diameter_path(skeleton):
    """
    Find the longest pixel-hop path through a morphological skeleton using
    the standard two-pass BFS tree-diameter algorithm.

    The "diameter" of a tree is its longest root-to-leaf path. For a
    fracture skeleton this reliably identifies the main backbone axis because:

    - Side branches and stub artefacts produced by thinning are always
      shorter than the true end-to-end length of the feature.
    - BFS hop count is a good proxy for arc length on a pixel skeleton, where
      all edges have unit weight (diagonal adjacency is treated identically to
      cardinal adjacency for simplicity).

    If the skeleton contains multiple disconnected components (e.g. a feature
    that was incompletely connected in the raster), only the largest component
    by pixel count is used. Pixels in smaller components are excluded from
    the path but remain visible in the skeleton overlay as discarded stubs.

    For closed-loop skeletons (which have no degree-1 endpoints), an
    arbitrary pixel is used as the BFS seed, which still produces the correct
    diameter for any tree-shaped component with a single loop.

    Args:
        skeleton (numpy.ndarray[bool]): 2-D boolean skeleton array as produced
            by morphological skeletonisation.

    Returns:
        list[tuple[int, int]]: Ordered list of ``(row, col)`` pixel
        coordinates tracing the diameter path from endpoint ``E1`` to
        endpoint ``E2``. Returns an empty list if the skeleton contains no
        pixels.
    """
    rows, cols = np.where(skeleton)
    if len(rows) == 0:
        return []

    pixels    = list(zip(rows.tolist(), cols.tolist()))
    pixel_set = set(pixels)

    def _nbrs(r, c):
        """
        Return all 8-connected skeleton neighbours of ``(r, c)``.

        Identical in structure to the ``nbrs`` closure in
        :func:`_extract_branches` but scoped to this function's own
        ``pixel_set``.

        Args:
            r (int): Row index.
            c (int): Column index.

        Returns:
            list[tuple[int, int]]: Neighbour pixel coordinates.
        """
        return [(r+dr, c+dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and (r+dr, c+dc) in pixel_set]

    # Pre-build the adjacency dict once so BFS doesn't recompute neighbours
    # on every visit. For large skeletons this is significantly faster than
    # calling _nbrs inside the BFS loop.
    adj = {p: _nbrs(*p) for p in pixel_set}

    # ── Find connected components and keep the largest ────────────────────────
    # A skeleton can fragment into multiple components if the raster contains
    # gaps or if morphological thinning separated a near-touching feature.
    # Only the dominant component is meaningful for the diameter path.
    visited    = set()
    components = []
    for seed in pixel_set:
        if seed in visited:
            continue
        # Standard BFS flood-fill to collect all pixels in this component.
        comp  = []
        queue = deque([seed])
        visited.add(seed)
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    # Select the component with the most pixels as the primary skeleton.
    component = max(components, key=len)
    comp_set  = set(component)
    # Rebuild adjacency restricted to the chosen component so BFS cannot
    # accidentally cross into a smaller disconnected fragment.
    cadj = {p: [nb for nb in adj[p] if nb in comp_set] for p in component}

    # Degree-1 pixels are the natural start/end candidates for the diameter.
    endpoints = [p for p in component if len(cadj[p]) == 1]
    if not endpoints:
        # Closed loop: no degree-1 pixels exist, so any pixel can serve as
        # the BFS seed without affecting the diameter result.
        endpoints = [component[0]]

    def _bfs(start):
        """
        Run BFS from ``start`` over the largest skeleton component and return
        the hop-distance and parent maps for the entire reachable subgraph.

        Args:
            start (tuple[int, int]): The ``(row, col)`` pixel from which BFS
                begins. Must be a member of ``component``.

        Returns:
            tuple[dict, dict]:
                - **dist** (*dict[tuple, int]*): Maps each reachable pixel to
                  its BFS hop distance from ``start``.
                - **parent** (*dict[tuple, tuple | None]*): Maps each reachable
                  pixel to the pixel it was reached from, or ``None`` for
                  ``start`` itself. Used to reconstruct the path after the
                  second BFS pass.
        """
        dist   = {start: 0}
        parent = {start: None}
        q      = deque([start])
        while q:
            node = q.popleft()
            for nb in cadj[node]:
                if nb not in dist:
                    dist[nb]   = dist[node] + 1
                    parent[nb] = node
                    q.append(nb)
        return dist, parent

    # ── Two-pass diameter ─────────────────────────────────────────────────────
    # Pass 1: BFS from any endpoint. The farthest reachable endpoint (E1) is
    # guaranteed to be one end of the diameter by the tree-diameter theorem.
    d0, _     = _bfs(endpoints[0])
    reachable = [p for p in endpoints if p in d0]
    e1        = max(reachable, key=lambda p: d0[p])

    # Pass 2: BFS from E1. The farthest reachable endpoint (E2) is the other
    # end of the diameter. The parent map from this pass is used to reconstruct
    # the full path by walking backwards from E2 to E1.
    d1, par1   = _bfs(e1)
    reachable2 = [p for p in endpoints if p in d1]
    e2         = max(reachable2, key=lambda p: d1[p])

    # ── Path reconstruction ───────────────────────────────────────────────────
    # Walk the parent chain backwards from E2 until we reach E1 (parent=None).
    # Reversing at the end gives the path in E1→E2 order.
    path, node = [], e2
    while node is not None:
        path.append(node)
        node = par1[node]
    path.reverse()
    return path


def _prune_and_merge_branches(raw_branches, min_pixels, min_percent,
                              protected_pixels=None):
    """
    Iteratively remove stub branches from a skeleton branch graph and merge
    the segments they leave behind into a single continuous path.

    A "stub" is a short branch arm — a protrusion artefact introduced by
    morphological thinning at surface irregularities, junctions, or
    terminations. Three criteria must all hold for a branch to be classified
    as a stub:

    1. **Not protected** — the protected-pixels flag is clear (see below).
    2. **Dangling endpoint** — at least one endpoint pixel is *not* shared by
       any other alive branch (i.e. its degree in the adjacency graph is 1).
       A segment whose *both* endpoints connect to other branches is a
       **bridge**: it keeps two parts of the skeleton connected and must
       *never* be removed regardless of its length.
    3. **Short enough** — satisfies at least one of:

       - **Absolute length**: fewer than ``min_pixels`` pixels.
       - **Relative length**: shorter than ``min_percent`` percent of the
         total skeleton pixel count (disabled when ``min_percent`` is 0).

    **Protected pixels**: when ``protected_pixels`` is supplied it should be a
    set of ``(row, col)`` pixel coordinates belonging to the main branch of the
    feature (typically the diameter path from :func:`_find_diameter_path`).
    Any branch segment that contains one or more pixels from this set is
    treated as **immune** from stub removal, regardless of its length.  This
    prevents short connecting segments along the main spine — which arise when
    two junctions sit close together — from being pruned away and
    disconnecting the skeleton.  Protection is re-evaluated each time a merge
    produces a new branch, so a merged branch that absorbs any protected pixel
    inherits the immunity.

    **Why iterative removal is required**

    A naive batch approach (remove all stubs in one pass) fails when multiple
    stubs cluster near each other, because the short inter-junction segments
    connecting adjacent stub roots are themselves below the threshold and are
    erroneously removed along with the stubs, creating false breaks in the
    main branch::

        left(3px)─J1─mid(3px)─J2─right(10px)   MIN_BRANCH_PIXELS=5
        S1(2px) at J1,  S2(2px) at J2

        Batch:      remove S1, S2, left, mid → only right(10px) survives → BREAK
        Iterative (smallest-first):
          remove S1 → J1 freed → merge left+mid = 5px ✓
          remove S2 → J2 freed → merge 5px+right = 14px ✓
          → single continuous branch, no gaps

    Processing smallest-first guarantees that genuine stubs are always removed
    before the slightly-longer main-branch segments they split, so those
    segments have grown via merging before they could be mis-classified.

    **Fallback**: if every branch is below the threshold (i.e. the entire
    feature is smaller than ``min_pixels``), the single longest raw branch is
    returned rather than an empty list, ensuring a feature is never silently
    deleted by the pruning step.

    Args:
        raw_branches (list[list[tuple[int, int]]]): Branch pixel lists as
            returned by :func:`_extract_branches`. Each inner list is an
            ordered sequence of ``(row, col)`` tuples.
        min_pixels (int): Absolute pixel-count threshold. Branches shorter
            than this are candidates for removal.
        min_percent (float): Relative threshold as a percentage of total
            skeleton pixels. Branches below this fraction are candidates for
            removal. Pass ``0`` to disable percentage-based pruning.
        protected_pixels (set[tuple[int, int]] | None): Optional set of pixel
            coordinates that belong to the main branch and must never be
            culled.  Pass ``None`` (default) to disable protection.

    Returns:
        list[list[tuple[int, int]]]: Pruned and merged branch pixel lists.
        Each inner list represents one surviving branch in ``(row, col)``
        pixel coordinates. May return the single longest raw branch if all
        branches were below the threshold.
    """
    if not raw_branches:
        return []

    # Normalise: an empty set and None both mean "no protection".
    _protected = protected_pixels if protected_pixels else set()

    def _is_protected(pixels):
        """Return True if any pixel in the list is in the protected set."""
        return any(tuple(p) in _protected for p in pixels)

    # Wrap each raw pixel list in a dict so we can update individual branches
    # (mark as dead, replace with merged version) without shifting list indices.
    # The "alive" flag avoids physically removing entries from the list, which
    # would invalidate any stored index references.
    # The "protected" flag marks branches that share pixels with the main branch
    # spine; these are immune from stub culling regardless of their length.
    branches = [
        {
            "pixels":    list(px),
            "start":     px[0],
            "end":       px[-1],
            "alive":     True,
            "protected": _is_protected(px),
        }
        for px in raw_branches
        if len(px) >= 2   # Discard degenerate single-pixel branches upfront.
    ]
    if not branches:
        return []

    # Total pixel count is fixed at construction time and used as the
    # denominator for the percentage threshold throughout the pruning loop.
    total_pixels = sum(len(b["pixels"]) for b in branches)

    def _is_stub(b):
        """
        Return ``True`` if branch ``b`` is a candidate for removal.

        A branch qualifies as a stub only when ALL of the following hold:

        1. It is not protected (main-spine flag is clear).
        2. It has at least one *dangling* endpoint — i.e. an endpoint pixel
           that is not shared by any other alive branch.  A segment whose
           both endpoints connect to other branches is a **bridge**: removing
           it would disconnect the skeleton, so it is never a stub regardless
           of its length.
        3. Its pixel count is below the absolute threshold (``min_pixels``)
           *or* below the relative threshold (``min_percent`` of the total
           skeleton pixel count).

        The caller's ``adj`` dict is captured by reference so this function
        always sees the current live adjacency at call time.

        Args:
            b (dict): A branch dict with ``"pixels"``, ``"protected"``,
                ``"start"``, and ``"end"`` keys.

        Returns:
            bool: ``True`` if the branch is eligible for stub pruning.
        """
        # Protected branches (main-spine segments) are never stubs.
        if b["protected"]:
            return False

        # Bridge check: if both endpoints already connect to at least one
        # *other* alive branch (degree ≥ 2 in the adjacency graph), this
        # segment bridges two junctions.  Removing it would split the
        # skeleton, so we must keep it regardless of length.
        start_degree = len(adj[b["start"]])
        end_degree   = len(adj[b["end"]])
        if start_degree >= 2 and end_degree >= 2:
            return False

        n = len(b["pixels"])
        if n < min_pixels:
            return True
        # Percentage check: disabled when min_percent is zero to allow
        # callers to opt out of relative thresholding entirely.
        if min_percent > 0 and total_pixels > 0:
            if (n / total_pixels * 100.0) < min_percent:
                return True
        return False

    # Node-to-branch adjacency: maps each endpoint pixel to the set of
    # branch indices that have that pixel as their start or end node.
    # Stored as a defaultdict so new nodes can be added during merging
    # without explicit initialisation.
    adj = defaultdict(set)
    for i, b in enumerate(branches):
        adj[b["start"]].add(i)
        adj[b["end"]].add(i)

    def _remove(i):
        """
        Mark branch ``i`` as dead and remove it from the adjacency index.

        Args:
            i (int): Index into the ``branches`` list.
        """
        b = branches[i]
        if not b["alive"]:
            return
        adj[b["start"]].discard(i)
        adj[b["end"]].discard(i)
        b["alive"] = False

    def _add(pixels):
        """
        Append a new branch to the ``branches`` list and register it in the
        adjacency index.

        The new branch inherits ``"protected": True`` if any of its pixels
        appear in the protected set, so that merged segments that absorb a
        main-spine pixel are themselves immune from subsequent culling.

        Args:
            pixels (list[tuple[int, int]]): Ordered pixel list for the new
                branch. Must contain at least two pixels.

        Returns:
            int: Index of the newly added branch in ``branches``.
        """
        idx = len(branches)
        b   = {
            "pixels":    pixels,
            "start":     pixels[0],
            "end":       pixels[-1],
            "alive":     True,
            "protected": _is_protected(pixels),
        }
        branches.append(b)
        adj[b["start"]].add(idx)
        adj[b["end"]].add(idx)
        return idx

    def _try_merge_at(node):
        """
        Merge two branches through ``node`` if exactly two live branches
        meet there, effectively dissolving the junction.

        This is called after a stub is removed to check whether its root
        junction has become a simple pass-through point (degree 2), in which
        case the two remaining branches should be joined into one continuous
        path.

        If the node has more or fewer than two live branches, no action is
        taken — a degree-3+ node is still a genuine junction, and a degree-0
        or degree-1 node is a free endpoint.

        Args:
            node (tuple[int, int]): The ``(row, col)`` pixel at which to
                attempt the merge. Typically the start or end of a just-removed
                stub branch.
        """
        # Refresh the adjacency set to remove any stale (dead) branch refs
        # before checking the live degree.
        live = {i for i in adj[node] if branches[i]["alive"]}
        adj[node] = live
        if len(live) != 2:
            return

        i, j = tuple(live)
        if i == j:
            # Self-loop guard: a single branch looping back to the same node
            # should not be merged with itself.
            return

        bi, bj = branches[i], branches[j]

        # Orient both pixel lists so that bi ends at the shared node and bj
        # begins there. This guarantees that concatenating px_i + px_j[1:]
        # produces a continuous path with the junction pixel appearing once.
        px_i = list(reversed(bi["pixels"])) if bi["start"] == node else list(bi["pixels"])
        px_j = list(bj["pixels"]) if bj["start"] == node else list(reversed(bj["pixels"]))

        # Remove both source branches before adding the merged one so that
        # adjacency counts are correct when _add registers the new endpoints.
        _remove(i)
        _remove(j)
        # px_i[-1] == node == px_j[0]: drop px_j[0] to avoid duplicating the
        # shared junction pixel in the merged path.
        _add(px_i + px_j[1:])

    # ── Iterative stub removal ────────────────────────────────────────────────
    # Each iteration: find all current live stubs, remove the shortest one,
    # attempt merges at both of its endpoints, and repeat until no stubs remain.
    # Removing one stub per iteration (smallest first) ensures that newly
    # merged branches are re-evaluated in the next iteration rather than
    # accidentally pruned before they can absorb their neighbours.
    while True:
        live_stubs = [
            (i, len(b["pixels"]))
            for i, b in enumerate(branches)
            if b["alive"] and _is_stub(b)
        ]
        if not live_stubs:
            break   # No more stubs — pruning is complete.

        # Select the shortest stub to remove this iteration.
        stub_idx             = min(live_stubs, key=lambda x: x[1])[0]
        stub                 = branches[stub_idx]
        start_node, end_node = stub["start"], stub["end"]

        _remove(stub_idx)
        # After removing the stub, both of its endpoint nodes may now have
        # degree 2 and qualify for merging. Check and merge each independently.
        _try_merge_at(start_node)
        _try_merge_at(end_node)

    surviving = [b["pixels"] for b in branches if b["alive"]]

    # ── Fallback: never return an empty list ──────────────────────────────────
    # If the threshold was set so aggressively that every branch was pruned,
    # return the longest raw branch rather than silently deleting the feature.
    if not surviving:
        # raw_branches is guaranteed non-empty (checked at function entry),
        # so max() always returns a branch — no None guard needed.
        surviving = [max(raw_branches, key=len)]

    return surviving


# =============================================================================
# SINUOSITY
# =============================================================================

def _compute_sinuosity(pixels):
    """
    Compute the sinuosity of a pixel-coordinate path.

    Sinuosity is defined as the ratio of the path's total arc length to the
    straight-line (chord) distance from its first to its last pixel. It
    quantifies how much the path deviates from a straight line:

    - A perfectly straight path has sinuosity = 1.0.
    - A gently meandering path has sinuosity slightly above 1.0.
    - A strongly curved or folded path has sinuosity >> 1.0.

    Arc length is computed as the sum of Euclidean hop distances between
    consecutive pixels. Diagonal pixel hops contribute ``√2`` (≈ 1.414) and
    cardinal hops contribute ``1.0``, matching the true pixel-space distances.

    This function operates on raw pixel coordinates (as produced by
    :func:`_find_diameter_path`) rather than world coordinates, so the result
    is dimensionless and resolution-independent.

    Args:
        pixels (list[tuple[int, int]]): Ordered list of ``(row, col)`` pixel
            coordinates. Must contain at least two pixels for a meaningful
            result.

    Returns:
        float: Sinuosity of the path. Returns ``1.0`` for paths shorter than
        two pixels (where curvature cannot be measured), and also returns
        ``1.0`` when the chord length is effectively zero (a closed or nearly
        closed loop) to avoid division by zero.
    """
    if len(pixels) < 2:
        return 1.0

    # Vectorised arc-length: sum of Euclidean hop distances along the path.
    arr   = np.array(pixels, dtype=float)
    diffs = np.diff(arr, axis=0)
    arc   = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    # Chord: straight-line distance from the first pixel to the last.
    chord = float(np.hypot(arr[-1, 0] - arr[0, 0], arr[-1, 1] - arr[0, 1]))

    # A chord of essentially zero means the path begins and ends at the same
    # pixel (a closed loop or a path that doubles back on itself). In that
    # case sinuosity is technically undefined; return 1.0 as a neutral
    # fallback that will not incorrectly trigger the curvature threshold.
    return arc / chord if chord > 1e-9 else 1.0


# =============================================================================
# DIRECTIONAL SKELETONISATION
# =============================================================================

def _directional_skeleton(polygon, resolution):
    """
    Compute a single-branch centreline by slicing the polygon perpendicular to
    its PCA long axis and returning the midpoint of each cross-section.

    This approach was developed to address two specific failure modes of
    morphological thinning (Lee's algorithm) on individually-traced fractures:

    - **Flat terminations**: where a fracture abuts another, the thinned
      skeleton forks into two corner pixels at the end. A perpendicular
      cross-section at the same location has a single midpoint, producing a
      clean single-path terminus.
    - **Short-fall at rounded ends**: thinning stops slightly short of the
      true polygon boundary. Cross-sections span the full long-axis extent,
      so the resulting path reaches the actual polygon edge.

    The algorithm works entirely in world-space Shapely geometry, avoiding the
    coordinate quantisation errors that would arise from a raster-to-world
    round-trip.

    **Algorithm**:

    1. Compute the polygon's long axis and perpendicular via PCA on the
       exterior ring.
    2. Project all boundary vertices onto both axes to determine the full
       long-axis extent and the maximum perpendicular half-width.
    3. Divide the long-axis range into ``n`` bins of width ≈ ``resolution``.
    4. For each bin, construct a perpendicular line through the bin centre and
       intersect it with the polygon.
    5. Take the midpoint of the longest resulting line segment as the
       centreline point for that bin.
    6. Return all midpoints in long-axis order.

    **Limitations**: relies on a single global long axis. Works well for
    approximately linear features. Strongly curved features should use
    ``"single_branch"`` or ``"multi_branch"`` instead (the caller in
    :func:`_dispatch_skeletoniser` switches automatically based on sinuosity).

    Args:
        polygon (shapely.geometry.Polygon): The polygon to skeletonise.
        resolution (float): Approximate slice width in world units, typically
            the effective raster resolution. Controls the density of midpoints
            in the output.

    Returns:
        list[tuple[float, float]]: Ordered list of ``(x, y)`` world
        coordinates tracing the centreline from one end of the long axis to
        the other. Returns a single-element list containing the polygon
        centroid if the long-axis span is less than half a resolution cell.
        Returns an empty list if the exterior ring has fewer than three
        vertices.
    """
    # MultiPolygon has no single .exterior ring; _polygon_pca handles this by
    # delegating to the convex hull, so the early guard only applies to simple
    # Polygons.
    if isinstance(polygon, Polygon) and len(polygon.exterior.coords) < 3:
        return []

    # ── PCA (Principal Component Analysis) to find the long axis ─────────────
    # Delegates to _polygon_pca so that the covariance decomposition is not
    # duplicated when _geometry_from_polygon is later called on the same polygon
    # during _calculate_statistics.
    pca = _polygon_pca(polygon)
    if pca is None:
        return []
    _ext, center, C, long_vec, perp_vec, _eigvals = pca

    # ── Axis extents from projected boundary vertices ─────────────────────────
    p_long = C @ long_vec   # Scalar projections along the long axis.
    p_perp = C @ perp_vec   # Scalar projections along the perpendicular.
    lo, hi     = float(p_long.min()), float(p_long.max())
    span       = hi - lo

    # The perpendicular half-width of the cutting lines must exceed the widest
    # part of the polygon. Adding one resolution cell provides a small margin
    # so that lines are guaranteed to fully cross the polygon boundary.
    perp_half  = float(max(abs(p_perp.min()), abs(p_perp.max()))) + resolution

    # A span smaller than half a cell means the polygon is essentially a point
    # along the long axis; return the centroid as a degenerate centreline.
    if span < resolution * 0.5:
        return [(float(center[0]), float(center[1]))]

    # ── Slice the polygon ─────────────────────────────────────────────────────
    # Divide the long-axis range into n_slices bins. The bin centres are used
    # as the along-axis positions for each perpendicular cutting line.
    n_slices = max(int(round(span / resolution)), 1)
    edges    = np.linspace(lo, hi, n_slices + 1)
    centres  = (edges[:-1] + edges[1:]) * 0.5

    pts = []
    for pos in centres:
        # Convert the scalar long-axis position back to a world-coordinate
        # point by adding the scaled long-axis vector to the polygon centroid.
        sc = center + pos * long_vec
        # Build the perpendicular cutting line through this point. The line
        # extends perp_half in both directions to guarantee it crosses the
        # polygon regardless of how wide it is at this cross-section.
        p1 = sc + perp_half * perp_vec
        p2 = sc - perp_half * perp_vec

        try:
            inter = polygon.intersection(LineString([p1.tolist(), p2.tolist()]))
        except Exception:
            # Shapely can raise on numerically degenerate geometries; skip this
            # slice rather than aborting the entire centreline computation.
            continue

        if inter.is_empty:
            continue

        if inter.geom_type == "LineString":
            # Simple single-segment intersection — midpoint is the mean of
            # all coordinate pairs along the segment.
            ic  = np.array(inter.coords)
            mid = ic.mean(axis=0)

        elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
            # The cutting line crossed a non-convex polygon boundary multiple
            # times, producing several disjoint segments. Use the midpoint of
            # the longest segment as the best representative cross-section
            # centre, ignoring the shorter slivers from re-entrant concavities.
            lines = [g for g in inter.geoms if g.geom_type == "LineString"]
            if not lines:
                continue
            ic  = np.array(max(lines, key=lambda l: l.length).coords)
            mid = ic.mean(axis=0)

        elif inter.geom_type == "Point":
            # The cutting line grazed the polygon at a single boundary point
            # (e.g. a very acute tip). Use that point directly as the midpoint.
            mid = np.array([inter.x, inter.y])

        else:
            # Unexpected geometry type (e.g. Polygon from a near-degenerate
            # intersection) — skip this slice.
            continue

        pts.append((float(mid[0]), float(mid[1])))

    return pts


# =============================================================================
# METHOD SELECTION
# =============================================================================

def _select_skeletonisation_method(polygon):
    """
    Inspect a polygon's geometry and recommend a skeletonisation strategy.

    This function is the first stage of the ``"auto"`` decision tree.  It
    operates entirely on the Shapely polygon — no rasterisation is performed —
    so it runs in negligible time for any polygon the parser can produce.

    Three nested predicates are evaluated in order.  The first one that fires
    short-circuits the rest:

    1. **Interior holes** (``_interiors_check``): a polygon with one or more
       interior rings is ring-like or donut-shaped.  The directional method
       cannot represent topological complexity, so Lee's method is used.
    2. **Solidity** (``_solidity_check``): ``polygon.area / convex_hull.area``.
       A low value indicates significant non-convexity — concave pockets,
       branching arms, or L-shapes — that would cause the directional
       cross-section axis to stray outside the polygon.
    3. **Aspect ratio** (``_aspect_ratio_check``): long axis / short axis of the
       minimum rotated bounding rectangle.  A compact feature (low ratio) is
       unlikely to have a well-defined long axis for the directional method.

    If all checks pass the function returns ``"directional"`` as a tentative
    recommendation; the caller should still validate the computed directional
    skeleton with :func:`_validate_directional_skeleton`.

    Args:
        polygon (shapely.geometry.Polygon): The source polygon to inspect.

    Returns:
        tuple:
            - **method** (*str*): ``"directional"`` or ``"lee"``.
            - **reason** (*str*): Human-readable explanation of the decision.
            - **diagnostics** (*dict*): Numeric metrics computed during the
              checks (``"has_hole"``, ``"solidity"``, ``"aspect_ratio"``), for
              use in log messages and the skeleton overlay title.
    """

    def _interiors_check():
        """Return True if the polygon has at least one interior ring (hole)."""
        # MultiPolygon has no .interiors attribute; treat it as hole-free for
        # the purposes of method selection (holes are rare in compound features).
        if not isinstance(polygon, Polygon):
            return False
        return any(True for _ in polygon.interiors)

    def _solidity_check():
        """
        Return the polygon's solidity: area / convex-hull area.

        A value close to 1.0 means the polygon fills its convex hull tightly
        (elongated, smooth-sided).  Lower values indicate concavities, re-
        entrant corners, or multi-armed branching shapes.
        """
        hull_area = polygon.convex_hull.area
        if hull_area == 0:
            return 0.0
        return polygon.area / hull_area

    def _aspect_ratio_check():
        """
        Return the aspect ratio (long / short axis) of the minimum rotated
        bounding rectangle.

        Uses Shapely's ``minimum_rotated_rectangle`` which returns a 4-vertex
        polygon; the two unique edge lengths give the axis lengths.
        Returns ``inf`` for degenerate (zero-width) rectangles.
        """
        min_rot_rect = polygon.minimum_rotated_rectangle  # MRR: Minimum Rotated Rectangle
        coords       = list(min_rot_rect.exterior.coords)  # 5 points, first = last
        edge0  = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        edge1  = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        long_ax, short_ax = max(edge0, edge1), min(edge0, edge1)
        return long_ax / short_ax if short_ax > 0 else float("inf")

    # Always compute all three metrics before making any decision, so that
    # diagnostics always carries the full set of values regardless of which
    # check triggers the gate.  This allows the skeleton overlay plot and
    # terminal log to display all parameters even when the decision tree
    # short-circuited at an early stage.
    has_hole     = _interiors_check()
    solidity     = _solidity_check()
    aspect_ratio = _aspect_ratio_check()

    diagnostics = {
        "has_hole":     has_hole,
        "solidity":     solidity,
        "aspect_ratio": aspect_ratio,
    }

    if has_hole:
        return "lee", "interior hole detected", diagnostics

    if solidity < SOLIDITY_THRESHOLD:
        return ("lee",
                f"solidity={solidity:.3f} < SOLIDITY_THRESHOLD={SOLIDITY_THRESHOLD}",
                diagnostics)

    if aspect_ratio < ASPECT_RATIO_THRESHOLD:
        return ("lee",
                f"aspect ratio={aspect_ratio:.2f} < ASPECT_RATIO_THRESHOLD={ASPECT_RATIO_THRESHOLD}",
                diagnostics)

    return ("directional",
            f"solidity={solidity:.3f}, aspect ratio={aspect_ratio:.2f}",
            diagnostics)


def _validate_directional_skeleton(diameter, polygon, effective_resolution):
    """
    Attempt to build a directional skeleton and validate it against three
    failure criteria.

    This function is the second stage of the ``"auto"`` and
    ``"directional_and_single_branch"`` decision trees (and is also invoked
    by the explicit ``"directional"`` mode).  It wraps
    :func:`_directional_skeleton` and subjects the result to three nested
    checks, each of which can abort and recommend a Lee fallback.

    **Checks performed (in order)**:

    1. ``_curvature_check`` — computes the sinuosity of the Lee skeleton's
       diameter path.  A sinuosity above ``CURVATURE_THRESHOLD`` means the
       feature bends enough for the PCA cross-section axis to produce
       overlapping or gapped slices.
    2. ``_closed_loop_check`` — detects ring-like skeletons where the diameter
       path's endpoints are closer together than 10 % of the total arc length.
       Such features have no meaningful "start" or "end" for a directional
       traverse.
    3. ``_escape_check`` — after computing the directional polyline, counts the
       fraction of *interior* vertices (excluding head/tail, which are allowed
       to sit on the polygon boundary) that lie outside the source polygon.
       Head/tail vertices that are outside are trimmed first; if interior
       vertices still escape beyond ``ESCAPE_THRESHOLD`` the skeleton is
       considered invalid.

    Args:
        diameter (list[tuple[int, int]]): Lee skeleton diameter path in pixel
            coordinates, as returned by :func:`_find_diameter_path`.
        polygon (shapely.geometry.Polygon): The source polygon.
        effective_resolution (float): Effective raster resolution, used to build the
            inward-buffered containment polygon for the escape check.

    Returns:
        tuple:
            - **valid** (*bool*): ``True`` if the directional skeleton passed
              all checks and can be used.
            - **world_centreline** (*list[tuple[float, float]] | None*): The
              (possibly trimmed) directional polyline in world coordinates, or
              ``None`` if validation failed before computing it.
            - **sinuosity** (*float*): Sinuosity of the diameter path.
            - **reason** (*str*): Human-readable explanation of the outcome.
            - **dir_diag** (*dict*): Per-check diagnostic values for the title
              overlay.  Keys: ``"sinuosity"`` (*float*), ``"is_closed_loop"``
              (*bool | None*, ``None`` when curvature check aborted first),
              ``"n_escaped"`` (*int*), ``"n_total_interior"`` (*int*).
    """

    def _curvature_check():
        """
        Return (passed, sinuosity).  Fails when sinuosity > CURVATURE_THRESHOLD
        and CURVATURE_THRESHOLD > 0.
        """
        sinuosity = _compute_sinuosity(diameter)
        passed    = not (CURVATURE_THRESHOLD > 0 and sinuosity > CURVATURE_THRESHOLD)
        return passed, sinuosity

    def _closed_loop_check():
        """
        Return True (loop detected) if the diameter endpoints are within 10 %
        of the arc length of each other — indicating a ring-like skeleton with
        no clear start/end.
        """
        if len(diameter) < 2:
            return True
        chord = math.hypot(diameter[-1][0] - diameter[0][0],
                           diameter[-1][1] - diameter[0][1])
        arc   = sum(
            math.hypot(diameter[i + 1][0] - diameter[i][0],
                       diameter[i + 1][1] - diameter[i][1])
            for i in range(len(diameter) - 1)
        )
        return arc > 0 and chord < arc * 0.10

    def _escape_check(world_centreline):
        """
        Trim endpoint overruns and measure the fraction of arc length that
        lies outside the polygon.

        Returns (passed, trimmed_cl, outside_arc_length, interior_arc_length).

        Vertex-counting is deliberately avoided: a long skeleton segment can
        cross well outside the polygon between two vertices that are themselves
        inside, producing a false "escaped=0/N" count even though a large
        fraction of the path length is outside.  Measuring arc length via
        ``LineString.difference(polygon)`` catches those cases correctly.

        Fails when ``outside_arc_length / interior_arc_length > ESCAPE_THRESHOLD``.
        """
        if len(world_centreline) < 2:
            return False, world_centreline, 0.0, 0.0

        # Use a one-cell inward buffer to absorb floating-point boundary
        # placement; fall back to the original polygon if the buffer collapses.
        tol_poly   = polygon.buffer(-effective_resolution)
        check_poly = tol_poly if not tol_poly.is_empty else polygon

        # Trim contiguous outside vertices from the head and tail.  These are
        # expected overruns at polygon terminations and do not indicate that the
        # directional method is wrong — trim them before the arc-length test.
        outside = [not check_poly.contains(Point(p)) for p in world_centreline]

        trim_start = 0
        while trim_start < len(outside) and outside[trim_start]:
            trim_start += 1
        trim_end = len(outside)
        while trim_end > trim_start and outside[trim_end - 1]:
            trim_end -= 1

        trimmed = world_centreline[trim_start:trim_end]
        if len(trimmed) < 2:
            # Entire centreline was outside (or nearly so).
            return False, trimmed, 0.0, 0.0

        # Measure how much of the *interior* (trimmed) arc length lies outside
        # the polygon using set-difference on the LineString geometry.  This
        # correctly accounts for segments that stray outside between vertices.
        trimmed_line  = LineString(trimmed)
        interior_len  = trimmed_line.length
        if interior_len < 1e-9:
            return False, trimmed, 0.0, 0.0

        try:
            outside_geom   = trimmed_line.difference(check_poly)
            outside_len    = outside_geom.length if not outside_geom.is_empty else 0.0
        except Exception:
            # If the Shapely difference fails (e.g. invalid geometry), fall
            # back to vertex-counting so the check still produces a result.
            n_out = sum(1 for p in outside[trim_start:trim_end] if p)
            n_tot = trim_end - trim_start
            outside_len  = interior_len * (n_out / n_tot) if n_tot else 0.0

        frac = outside_len / interior_len

        if frac > ESCAPE_THRESHOLD:
            return False, trimmed, outside_len, interior_len

        return True, trimmed, outside_len, interior_len

    # Always compute all three checks — and the directional polyline that the
    # escape check requires — before making any decision.  This ensures that
    # dir_diag always carries the full set of values regardless of which check
    # triggers the gate, so the skeleton overlay plot and terminal log can
    # display every parameter even when the decision tree short-circuited.
    curve_ok, sinuosity = _curvature_check()
    is_closed_loop      = _closed_loop_check()
    world_centreline    = _directional_skeleton(polygon, effective_resolution)

    if len(world_centreline) >= 2:
        escape_ok, trimmed_cl, n_out, n_tot = _escape_check(world_centreline)
    else:
        escape_ok, trimmed_cl, n_out, n_tot = False, world_centreline, 0, 0

    dir_diag = {
        "sinuosity":       sinuosity,
        "is_closed_loop":  is_closed_loop,
        "escaped_length":  n_out,   # arc length (world units) outside polygon
        "interior_length": n_tot,   # total interior arc length (world units)
    }

    # ── Return earliest failure ───────────────────────────────────────────────
    if not curve_ok:
        return (False, trimmed_cl, sinuosity,
                f"sinuosity={sinuosity:.2f} > CURVATURE_THRESHOLD={CURVATURE_THRESHOLD}",
                dir_diag)

    if is_closed_loop:
        return (False, trimmed_cl, sinuosity, "closed-loop skeleton detected", dir_diag)

    if len(world_centreline) < 2:
        return (False, None, sinuosity,
                "directional skeleton produced fewer than 2 vertices", dir_diag)

    if not escape_ok:
        pct = f"{n_out / n_tot:.1%}" if n_tot > 1e-9 else "?"
        return (False, trimmed_cl, sinuosity,
                f"escaped polygon: {pct} of interior arc length outside"
                f" > ESCAPE_THRESHOLD={ESCAPE_THRESHOLD:.0%}",
                dir_diag)

    n_trimmed = len(world_centreline) - len(trimmed_cl)
    trim_note = f", {n_trimmed} endpoint vertex/vertices trimmed" if n_trimmed else ""
    esc_pct   = f"{n_out / n_tot:.1%}" if n_tot > 1e-9 else "0.0%"
    return (True, trimmed_cl, sinuosity,
            f"sinuosity={sinuosity:.2f}, {esc_pct} of interior arc length outside{trim_note}",
            dir_diag)


def _select_single_or_multi_branch_skeletonisation(pruned_branches, diameter_set):
    """
    Examine the post-pruning skeleton to choose between single-branch and
    multi-branch representation.

    This function is the third stage of the ``"auto"`` decision tree, called
    after :func:`_prune_and_merge_branches` has already removed stub artefacts.
    The **main** pixel count is defined as the union of all pruned-branch pixels
    that overlap the diameter path (the true spine, as returned by
    :func:`_find_diameter_path`).  The off-main fraction is then:

    .. code-block:: text

        off_main_fraction = (total_px − spine_px) / total_px

    This is compared against ``BRANCHING_THRESHOLD`` to decide the method.

    .. note::
        Earlier versions identified the "main branch" as the longest single
        segment from :func:`_extract_branches`.  Because that function splits
        the skeleton at every junction, the diameter path is fragmented into
        several segments; taking the longest fragment severely under-counted
        the spine and inflated ``off_main_fraction``.

    Using post-pruning branches (rather than the raw skeleton) is intentional:
    rasterisation noise and endpoint fraying at flat terminations create many
    short side-arms that are legitimate stubs, not genuine bifurcations.
    Evaluating on the pruned skeleton ensures only topologically meaningful
    branches count toward the branching fraction.

    Args:
        pruned_branches (list[list[tuple[int, int]]]): Branch pixel lists as
            returned by :func:`_prune_and_merge_branches`.
        diameter_set (set[tuple[int, int]]): Set of pixel coordinates that form
            the diameter (spine) path, as computed by
            :func:`_find_diameter_path`.

    Returns:
        tuple:
            - **method** (*str*): ``"single_branch"`` or ``"multi_branch"``.
            - **off_main_fraction** (*float*): Fraction of total post-pruning
              pixels that lie outside the spine, in ``[0.0, 1.0]``.
    """
    if not pruned_branches:
        return "single_branch", 0.0

    total_px = sum(len(b) for b in pruned_branches)
    if total_px == 0:
        return "single_branch", 0.0

    # Count how many pruned pixels coincide with the diameter (spine) path.
    # We union the pixel sets first to avoid double-counting junction pixels
    # that appear in more than one branch segment.
    all_pruned_px = set()
    for b in pruned_branches:
        all_pruned_px.update(map(tuple, b))

    spine_px         = len(all_pruned_px & diameter_set)
    off_px           = len(all_pruned_px) - spine_px
    # Use the deduplicated total so the fraction is consistent.
    off_main_fraction = off_px / max(len(all_pruned_px), 1)

    if off_main_fraction > BRANCHING_THRESHOLD:
        return "multi_branch", off_main_fraction
    return "single_branch", off_main_fraction


# =============================================================================
# SKELETONISATION STRATEGIES
# =============================================================================

def _directional_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution):
    """
    Attempt to skeletonise *feature* using the directional (PCA cross-section)
    method and write the result into ``feature`` if successful.

    Delegates all validation to :func:`_validate_directional_skeleton`, which
    runs the curvature, closed-loop, and polygon-escape checks in sequence.
    If any check fails the function returns without touching ``feature`` so
    the caller can fall back to a Lee-based method.

    Args:
        feature: The :class:`Feature` dataclass instance to populate.
        grid (numpy.ndarray): Boolean raster grid from
            :func:`_rasterise_polygon`.
        skeleton (numpy.ndarray): Thinned skeleton from
            :func:`_skimage_skeletonize`.
        x_min (float): World x-coordinate of pixel column 0.
        y_min (float): World y-coordinate of pixel row 0.
        effective_resolution (float): Effective raster resolution in world units.

    Returns:
        tuple:
            - **success** (*bool*): ``True`` if the directional skeleton was
              valid and ``feature`` has been populated.
            - **sinuosity** (*float*): Sinuosity of the Lee diameter path,
              regardless of success.
            - **world_centreline** (*list | None*): The accepted (possibly trimmed)
              directional polyline, or ``None`` on failure.
            - **diameter** (*list*): Lee skeleton diameter path (pixel coords),
              available to the caller for reuse as the single_branch fallback.
            - **reason** (*str*): Human-readable outcome description.
    """
    polygon  = feature.polygon
    diameter = _find_diameter_path(skeleton)
    valid, world_centreline, sinuosity, reason, dir_diag = _validate_directional_skeleton(
        diameter, polygon, effective_resolution
    )

    if not valid:
        # Return the computed (but invalid) centreline rather than None so
        # that callers which deliberately want the raw directional result
        # (e.g. "directional" mode with a malformed-feature warning) can still
        # use it.  Callers that fall back to Lee on failure simply ignore it.
        return False, sinuosity, world_centreline, diameter, reason, dir_diag

    # Store raw world coords; _post_process_skeleton will smooth and resample.
    branches = []
    if len(world_centreline) >= 2:
        branches.append(Branch(id=0, centerline=list(world_centreline)))

    feature.branches   = branches
    feature.skeleton_overlay_data = {
        "raster_grid":       _skeleton_overlay_grid(grid),
        "skeleton_grid":     _skeleton_overlay_grid(skeleton),
        "raw_branches":      [],
        "merged_branches":   [],
        "directional_world": world_centreline,
        "skeleton_method":   "directional",
        "sinuosity":         sinuosity,
        "x_min":             x_min,
        "y_min":             y_min,
        "resolution":        effective_resolution,
    }
    return True, sinuosity, world_centreline, diameter, reason, dir_diag


def _single_branch_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution,
                                diameter=None, sinuosity=None,
                                skeleton_method_label=None):
    """
    Skeletonise *feature* as a single-branch path using the Lee diameter.

    Traces the longest end-to-end path through the morphological skeleton tree
    (the "diameter") and uses it as the sole surviving branch.  No stub
    pruning is applied — the diameter algorithm inherently ignores side arms by
    following only the longest root-to-leaf route.

    This function is called both by the explicit ``"single_branch"`` mode and
    as a fallback within ``"directional_and_single_branch"``, ``"auto"``, and
    ``"single_or_multi_branch"`` modes when the directional skeleton fails
    validation or is not attempted.

    Args:
        feature: The :class:`Feature` dataclass instance to populate.
        grid (numpy.ndarray): Boolean raster grid.
        skeleton (numpy.ndarray): Thinned Lee skeleton.
        x_min (float): World x-coordinate of pixel column 0.
        y_min (float): World y-coordinate of pixel row 0.
        effective_resolution (float): Effective raster resolution.
        diameter (list | None): Pre-computed diameter path; recomputed if
            ``None``.
        sinuosity (float | None): Pre-computed sinuosity; stored in
            ``skeleton_overlay_data`` if provided.
        skeleton_method_label (str | None): Override for the ``"skeleton_method"``
            debug key (e.g. ``"single_branch (curved fallback)"``).  Defaults
            to ``"single_branch"``.

    Returns:
        Feature: ``feature``, mutated in place.
    """
    if diameter is None:
        diameter = _find_diameter_path(skeleton)

    raw_branches    = _extract_branches(skeleton)
    merged_branches = [diameter] if len(diameter) >= 2 else []

    # Store raw world coords; _post_process_skeleton will smooth and resample.
    branches = []
    for bid, pixels in enumerate(merged_branches):  # bid = branch ID (zero-based index)
        world = _pixels_to_world(pixels, x_min, y_min, effective_resolution)
        if len(world) >= 2:
            branches.append(Branch(id=bid, centerline=world))

    feature.branches   = branches
    feature.skeleton_overlay_data = {
        "raster_grid":     _skeleton_overlay_grid(grid),
        "skeleton_grid":   _skeleton_overlay_grid(skeleton),
        "raw_branches":    raw_branches,
        "merged_branches": merged_branches,
        "skeleton_method":     skeleton_method_label or "single_branch",
        "sinuosity":       sinuosity,
        "x_min":           x_min,
        "y_min":           y_min,
        "resolution":      effective_resolution,
    }
    return feature


def _multi_branch_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution,
                               diameter=None):
    """
    Skeletonise *feature* as a full multi-branch medial axis with stub pruning.

    Multi-branch is guaranteed to be a **strict superset** of single-branch:
    branch 0 is always the full diameter path (identical to what
    :func:`_single_branch_skeletonise` would return), and every additional
    branch is a non-spine segment from the pruned skeleton.  This ensures the
    main spine is never fragmented or accidentally dropped, regardless of how
    stub-pruning classifies individual skeleton segments.

    All skeleton branches are extracted and passed through iterative
    stub-pruning (:func:`_prune_and_merge_branches`).  Any pruned branch
    whose pixels overlap the diameter path by more than 50 % is considered a
    spine re-trace and suppressed; the rest are appended as side branches.

    Args:
        feature: The :class:`Feature` dataclass instance to populate.
        grid (numpy.ndarray): Boolean raster grid.
        skeleton (numpy.ndarray): Thinned Lee skeleton.
        x_min (float): World x-coordinate of pixel column 0.
        y_min (float): World y-coordinate of pixel row 0.
        effective_resolution (float): Effective raster resolution.
        diameter (list | None): Pre-computed diameter path used to derive
            protected pixels; recomputed if ``None``.

    Returns:
        Feature: ``feature``, mutated in place.
    """
    if diameter is None:
        diameter = _find_diameter_path(skeleton)

    # The diameter path IS the main spine — always use it as branch 0.
    # This guarantees multi-branch is a strict superset of single-branch:
    # the spine can never be fragmented or accidentally pruned, regardless
    # of how _prune_and_merge_branches classifies individual skeleton segments.
    diameter_set     = set(map(tuple, diameter))
    protected_pixels = diameter_set   # also passed to pruner for stub removal

    raw_branches    = _extract_branches(skeleton)
    merged_branches = _prune_and_merge_branches(
        raw_branches, MIN_BRANCH_PIXELS, MIN_BRANCH_PERCENT,
        protected_pixels=protected_pixels,
    )

    # Branch 0 is always the diameter (spine) path, converted directly to world
    # coordinates.  Additional branches are any pruned segment that is NOT
    # predominantly spine pixels (spine segments are already represented by
    # branch 0, so adding them again would create duplicates).
    branches = []
    spine_world = _pixels_to_world(list(diameter), x_min, y_min, effective_resolution)
    if len(spine_world) >= 2:
        branches.append(Branch(id=0, centerline=spine_world))

    for pixels in merged_branches:
        px_set              = set(map(tuple, pixels))
        spine_overlap_frac  = len(px_set & diameter_set) / max(len(px_set), 1)
        if spine_overlap_frac > 0.5:
            # This segment is mostly spine pixels — already represented as
            # branch 0; skip to avoid duplication.
            continue
        world = _pixels_to_world(pixels, x_min, y_min, effective_resolution)
        if len(world) >= 2:
            branches.append(Branch(id=len(branches), centerline=world))

    feature.branches   = branches
    feature.skeleton_overlay_data = {
        "raster_grid":     _skeleton_overlay_grid(grid),
        "skeleton_grid":   _skeleton_overlay_grid(skeleton),
        "raw_branches":    raw_branches,
        "merged_branches": merged_branches,
        "skeleton_method":     "multi_branch",
        "x_min":           x_min,
        "y_min":           y_min,
        "resolution":      effective_resolution,
    }
    return feature


# =============================================================================
# TOPOLOGY SNAPPING
# =============================================================================

def _nearest_point_on_polyline(pt, polyline):
    """
    Return the closest point on *polyline* to *pt* and the squared distance.

    Iterates over every segment of the polyline and finds the foot of the
    perpendicular from *pt* to that segment (clamped to the segment's
    endpoints).  The segment giving the smallest squared distance is
    returned.

    Args:
        pt (tuple[float, float]): Query point ``(x, y)``.
        polyline (list[tuple[float, float]]): Ordered vertex list; must
            contain at least one point.

    Returns:
        tuple[tuple[float, float], float]: ``(closest_pt, sq_dist)``
        where *closest_pt* is the foot of the perpendicular in world
        units and *sq_dist* is the squared Euclidean distance from *pt*.
    """
    px, py = pt
    best_sq  = float("inf")
    best_pt  = polyline[0]

    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        dx, dy = bx - ax, by - ay
        seg_sq = dx * dx + dy * dy
        if seg_sq == 0.0:
            # Degenerate segment (zero length) — snap to the vertex itself.
            t = 0.0
        else:
            t = ((px - ax) * dx + (py - ay) * dy) / seg_sq
            t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        sq = (px - cx) ** 2 + (py - cy) ** 2
        if sq < best_sq:
            best_sq = sq
            best_pt = (cx, cy)

    return best_pt, best_sq


def _snap_branches_to_neighbours(branches, coords_attr, max_snap_dist):
    """
    Snap each branch endpoint to the branch it is topologically connected to.

    After any coordinate transformation (smoothing, decimation) branches that
    shared an exact junction point beforehand may have drifted slightly apart,
    creating a visible topological gap in the SVG output.  This function
    re-establishes connectivity for each endpoint individually.

    **Algorithm** — for every endpoint *E* of every side branch:

    1. **Spine-junction test**: compute the nearest point on branch 0 (the
       spine) and its distance *D0*.  If *D0* ≤ *max_snap_dist*, *E* connects
       directly to the spine (a T-junction) → snap *E* onto the spine polyline.
    2. **Side-branch-junction test**: if *E* did not pass the spine test, scan
       the endpoints of every other branch and find the nearest one, distance
       *Dj*.  If *Dj* ≤ *max_snap_dist*, *E* connects to that side branch
       → snap *E* onto that branch's polyline.
    3. **Leaf**: if neither test triggers, *E* is a dangling leaf endpoint —
       leave it unchanged.

    The two-stage priority (spine first, side-branch second) is essential for
    complex features where some branches connect to the spine directly (stage 1)
    while others connect to intermediate side branches that eventually lead back
    to the spine (stage 2).  The old single-stage approach, which snapped every
    endpoint onto the spine regardless of actual topology, dragged distant
    endpoints across the feature and created long spurious spokes.

    Must be called **after** *coords_attr* has reached its final form for all
    branches:

    - ``"centerline"`` — call after smoothing, before decimation, to keep
      ``skeleton_raw.svg`` topologically connected.
    - ``"output_coords"`` — call after decimation; decimation is applied
      independently per branch so the pre-decimation snap point may no longer
      lie on the decimated spine, requiring a second snapping pass.

    Args:
        branches (list[Branch]): All branches of a feature.  ``branches[0]``
            is the spine; remaining entries are side branches.  The attribute
            named by *coords_attr* is modified **in place**.
        coords_attr (str): Branch coordinate attribute to snap, e.g.
            ``"centerline"`` or ``"output_coords"``.
        max_snap_dist (float): Maximum distance (world units) within which an
            endpoint is considered to be at a junction.  Endpoints farther than
            this from every other branch are treated as leaves and not snapped.
            Typically ``SMOOTHING + 2 * effective_resolution``.
    """
    if len(branches) < 2:
        return   # Nothing to snap for single-branch features.

    spine = getattr(branches[0], coords_attr, None) or []
    if len(spine) < 2:
        return

    max_sq = max_snap_dist * max_snap_dist

    for i, branch in enumerate(branches):
        cl = getattr(branch, coords_attr, None)
        if not cl or len(cl) < 2:
            continue

        for ep_idx in (0, -1):
            ep = cl[ep_idx]

            # ── Stage 1: spine-junction test ─────────────────────────────────
            # If this endpoint lies within max_snap_dist of the spine polyline,
            # treat it as a T-junction onto the spine and snap there.
            # Skip this test for branch 0 itself (the spine) to avoid
            # self-modification.
            if i != 0:
                snap_pt, sq = _nearest_point_on_polyline(ep, spine)
                if sq <= max_sq:
                    cl[ep_idx] = snap_pt
                    continue   # Snapped to spine — move on to the next endpoint.

            # ── Stage 2: side-branch-junction test ───────────────────────────
            # Endpoint is not close to the spine (or this is the spine itself).
            # Scan every other branch's endpoints to find the nearest one.
            # If it is within max_snap_dist, this is a side-branch-to-side-branch
            # (or spine-to-spine-end) junction — snap onto that branch's polyline.
            best_ep_sq   = float("inf")
            best_snap_pt = None

            for j, other in enumerate(branches):
                if j == i:
                    continue
                other_cl = getattr(other, coords_attr, None) or []
                if len(other_cl) < 2:
                    continue
                # Test both endpoints of the other branch to locate the junction.
                for other_ep in (other_cl[0], other_cl[-1]):
                    ep_sq = (ep[0] - other_ep[0]) ** 2 + (ep[1] - other_ep[1]) ** 2
                    if ep_sq < best_ep_sq:
                        best_ep_sq   = ep_sq
                        # Snap to the nearest point on the *full polyline*
                        # (not just the endpoint), in case the junction lands
                        # at a T-point along the other branch's interior.
                        best_snap_pt, _ = _nearest_point_on_polyline(ep, other_cl)

            if best_ep_sq <= max_sq and best_snap_pt is not None:
                cl[ep_idx] = best_snap_pt
            # else: leaf endpoint — leave unchanged.


# =============================================================================
# POST-PROCESSING
# =============================================================================

def _post_process_skeleton(feature):
    """
    Apply all post-skeletonisation processing to every branch of *feature*.

    This function runs after the raw skeleton has been extracted and the
    branch pixel lists have been converted to world-space coordinates.  It
    executes three transformations in sequence, each building on the previous:

    1. **Gaussian smoothing** (:func:`_smooth_coords`): reduces rasterisation
       staircase artefacts along the branch centreline.  The effective raster
       resolution stored in ``skeleton_overlay_data`` is used to convert the world-unit
       sigma to pixel-space, ensuring smoothing strength is invariant to
       ``RASTER_RESOLUTION``.  Result stored in ``branch.centerline``.

    2. **Arc-length resampling** (:func:`_resample_at_interval`): produces
       evenly-spaced measurement locations along the smoothed centreline at
       ``SAMPLING_INTERVAL`` world-unit intervals.  These are the positions
       used by the width-profiling step.  Result stored in
       ``branch.sample_points``.

    3. **Centreline snap** (:func:`_snap_branches_to_neighbours` on
       ``"centerline"``): for multi-branch features, each branch endpoint is
       snapped to whichever neighbouring branch it is topologically closest to
       — the spine if within range, otherwise the nearest side-branch endpoint.
       This keeps ``skeleton_raw.svg`` (``EXPORT_RAW_TRACES`` mode) topologically
       connected, because that file writes ``centerline`` directly.

    4. **Output decimation** — two-pass vertex reduction for the SVG export:

       a. Uniform spacing (:func:`_decimate_coords_with_uniform_spacing`) at
          ``OUTPUT_RESOLUTION`` world units removes overly dense vertices
          introduced by rasterisation.
       b. Ramer–Douglas–Peucker simplification (:func:`_decimate_coords_rdp`)
          at ``RDP_EPSILON`` tolerance further reduces vertex count in
          low-curvature regions while preserving sharp bends.

       Result stored in ``branch.output_coords``.  This is the coordinate
       list written to ``skeleton.svg``; ``branch.centerline`` retains the
       full-density smoothed path for use by the width-profiling and overlay
       plot steps.

    5. **Output-coords snap** (:func:`_snap_branches_to_neighbours` on
       ``"output_coords"``): decimation is applied independently to each branch,
       so the pre-decimation snap point may no longer lie on the decimated
       polyline of its neighbour.  A second topology-aware snap against the
       final ``output_coords`` of all branches re-establishes connectivity in
       ``skeleton.svg``.

    Branches whose centreline collapses to fewer than two points after
    smoothing are removed from ``feature.branches`` so that downstream steps
    never receive degenerate inputs.

    Args:
        feature: A :class:`Feature` dataclass instance whose ``branches`` list
            has been populated by one of the ``_*_skeletonise`` functions.
            ``feature.skeleton_overlay_data["resolution"]`` is used as the effective
            raster resolution for smoothing; falls back to ``RASTER_RESOLUTION``
            if absent.

    Returns:
        Feature: ``feature``, mutated in place.
    """
    effective_resolution = config.RASTER_RESOLUTION
    if feature.skeleton_overlay_data:
        effective_resolution = feature.skeleton_overlay_data.get("resolution", config.RASTER_RESOLUTION)

    # ── Step 1: Gaussian smoothing (all branches) ─────────────────────────────
    # Smooth every branch first so the spine centreline is finalised before the
    # side-branch snapping step uses it as a reference.
    live_branches = []
    for branch in feature.branches:
        branch.centerline = _smooth_coords(branch.centerline, effective_resolution)
        if len(branch.centerline) < 2:
            continue   # Smoothing collapsed the branch — discard it.
        live_branches.append(branch)
    feature.branches = live_branches

    # ── Step 2: arc-length resampling for width measurement ───────────────────
    from analysis import _resample_at_interval
    for branch in feature.branches:
        branch.sample_points = _resample_at_interval(branch.centerline)

    # ── Step 3: snap centrelines to topological neighbours ───────────────────
    # Snapping centreline here keeps skeleton_raw.svg (EXPORT_RAW_TRACES)
    # connected, because centreline is written directly to that file.
    # The threshold is SMOOTHING (displacement budget from Gaussian smoothing)
    # plus two pixels' worth of world units (rasterisation quantisation).
    snap_threshold = config.SMOOTHING + 2 * effective_resolution
    _snap_branches_to_neighbours(feature.branches, "centerline", snap_threshold)

    # ── Step 4: output decimation for SVG export ──────────────────────────────
    for branch in feature.branches:
        out = _decimate_coords_with_uniform_spacing(branch.centerline, config.OUTPUT_RESOLUTION)
        out = _decimate_coords_rdp(out, config.RDP_EPSILON)
        branch.output_coords = out

    # ── Step 5: snap output_coords to topological neighbours ─────────────────
    # Decimation is applied independently to each branch, so a branch endpoint
    # that was snapped onto its neighbour's pre-decimation centreline may no
    # longer lie on the decimated polyline.  Snapping again here — against the
    # final output_coords of all branches — re-establishes topological
    # connectivity in the main skeleton.svg output.
    _snap_branches_to_neighbours(feature.branches, "output_coords", snap_threshold)

    return feature


# =============================================================================
# DISPATCH HELPERS
# =============================================================================

def _dispatch_lee_path(feature, grid, skeleton, x_min, y_min, effective_resolution,
                       geo_method, geo_reason, geo_diag, dir_diag,
                       allow_multi_branch, diameter=None):
    """
    Run the Lee skeleton path (stage 3 of the decision tree) and populate *feature*.

    Extracts branches, prunes stubs with diameter-path protection, then calls
    :func:`_select_single_or_multi_branch_skeletonisation` to decide the
    end-point.  If *allow_multi_branch* is ``False``, the result is always
    ``single_branch`` regardless of the measured off-main fraction.

    This helper is shared by ``"auto"``, ``"directional_and_single_branch"``,
    and ``"single_or_multi_branch"`` modes so that the Lee logic is not
    duplicated.

    Args:
        feature: :class:`Feature` instance to populate.
        grid, skeleton, x_min, y_min, effective_resolution: Raster parameters.
        geo_method (str): Geometry-gate recommendation (``"directional"`` or
            ``"lee"``); stored in ``auto_decision`` for the overlay plot.
        geo_reason (str): Human-readable geometry-gate outcome.
        geo_diag (dict): Numeric geometry metrics from
            :func:`_select_skeletonisation_method`.
        dir_diag (dict | None): Directional-gate metrics from
            :func:`_directional_skeletonise`, or ``None`` if directional was
            not attempted.
        allow_multi_branch (bool): If ``True``, the branching threshold may
            select ``multi_branch``; if ``False``, ``single_branch`` is always
            the outcome (used by ``"directional_and_single_branch"``).
        diameter (list | None): Pre-computed diameter path; computed here if
            ``None``.
    """
    if diameter is None:
        diameter = _find_diameter_path(skeleton)

    protected_pixels = set(map(tuple, diameter))
    raw_branches     = _extract_branches(skeleton)
    pruned_branches  = _prune_and_merge_branches(
        raw_branches, MIN_BRANCH_PIXELS, MIN_BRANCH_PERCENT,
        protected_pixels=protected_pixels,
    )

    if allow_multi_branch:
        lee_method, off_frac = _select_single_or_multi_branch_skeletonisation(
            pruned_branches, protected_pixels,
        )
    else:
        # Force single_branch regardless of the branching threshold.
        _sb, off_frac = _select_single_or_multi_branch_skeletonisation(
            pruned_branches, protected_pixels,
        )
        lee_method = "single_branch"

    n_stubs_pruned  = len(raw_branches)     - len(pruned_branches)
    n_pixels_pruned = sum(len(b) for b in raw_branches) \
                    - sum(len(b) for b in pruned_branches)

    # Build a compact all-parameters suffix for the terminal log so every
    # diagnostic value is visible regardless of which gate triggered.
    sol_v   = geo_diag.get("solidity")
    ar_v    = geo_diag.get("aspect_ratio")
    hole_v  = geo_diag.get("has_hole", False)
    sin_v   = (dir_diag or {}).get("sinuosity")
    loop_v  = (dir_diag or {}).get("is_closed_loop")
    esc_len = (dir_diag or {}).get("escaped_length")
    int_len = (dir_diag or {}).get("interior_length")
    geo_log = (
        f"sol={sol_v:.3f}(≥{SOLIDITY_THRESHOLD})" if sol_v is not None else ""
    )
    geo_log += f"  AR={ar_v:.2f}(≥{ASPECT_RATIO_THRESHOLD})" if ar_v is not None else ""
    geo_log += f"  holes={int(hole_v)}"
    dir_log = ""
    if sin_v is not None:
        dir_log += f"  sin={sin_v:.2f}(≤{CURVATURE_THRESHOLD})"
    if loop_v is not None:
        dir_log += f"  loop={'Y' if loop_v else 'N'}"
    if esc_len is not None and int_len is not None:
        esc_pct = f"{esc_len / int_len:.1%}" if int_len > 1e-9 else "0.0%"
        dir_log += f"  esc={esc_pct}(≤{ESCAPE_THRESHOLD:.0%})"

    tag = "auto" if allow_multi_branch else "dir+sb"
    if lee_method == "multi_branch":
        print(f"    [{tag}] multi_branch  |  off_main={off_frac:.1%} > {BRANCHING_THRESHOLD:.0%}"
              f"  |  {geo_log}{dir_log}")
        _multi_branch_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution,
                                  diameter=diameter)
        skel_label = f"multi_branch ({tag})"
    else:
        forced_note = " (forced)" if not allow_multi_branch and off_frac > BRANCHING_THRESHOLD else ""
        cmp = ">" if off_frac > BRANCHING_THRESHOLD else "≤"
        print(f"    [{tag}] single_branch{forced_note}  |  off_main={off_frac:.1%} {cmp} {BRANCHING_THRESHOLD:.0%}"
              f"  |  {geo_log}{dir_log}")
        _single_branch_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution,
                                   diameter=diameter,
                                   skeleton_method_label=f"single_branch ({tag})")
        skel_label = f"single_branch ({tag})"

    feature.skeleton_overlay_data["skeleton_method"] = skel_label
    feature.skeleton_overlay_data["auto_decision"] = {
        "geo_method":        geo_method,
        "geo_reason":        geo_reason,
        "final_method":      lee_method,
        "off_main_fraction": off_frac,
        "n_stubs_pruned":    n_stubs_pruned,
        "n_pixels_pruned":   n_pixels_pruned,
        **geo_diag,
        **(dir_diag or {}),
    }


def _dispatch_with_geometry_gate(feature, grid, skeleton, x_min, y_min, effective_resolution,
                                 polygon, allow_multi_branch):
    """
    Run the full geometry-gate + directional-attempt decision tree.

    This implements stages 1–3 of the ``"auto"`` decision tree and is shared
    by ``"auto"`` and ``"directional_and_single_branch"`` modes.  The only
    difference between the two modes is controlled by *allow_multi_branch*:

    - ``True``  → ``"auto"``: the Lee path may produce ``multi_branch``.
    - ``False`` → ``"directional_and_single_branch"``: the Lee path always
      produces ``single_branch``.

    Args:
        feature, grid, skeleton, x_min, y_min, effective_resolution:
            Standard raster parameters.
        polygon: The source Shapely polygon for the geometry gate.
        allow_multi_branch (bool): See above.
    """
    # Stage 1 — pre-rasterisation geometry check.
    geo_method, geo_reason, geo_diag = _select_skeletonisation_method(polygon)

    directional_succeeded = False
    dir_diag              = None
    diameter              = None

    if geo_method == "directional":
        # Stage 2 — validate the directional skeleton.
        success, sinuosity, world_centreline, diameter, dir_reason, dir_diag = \
            _directional_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution)
        tag = "auto" if allow_multi_branch else "dir+sb"
        # Build a compact all-parameters log line for the terminal.
        sol_v   = geo_diag.get("solidity")
        ar_v    = geo_diag.get("aspect_ratio")
        hole_v  = geo_diag.get("has_hole", False)
        sin_v   = (dir_diag or {}).get("sinuosity")
        loop_v  = (dir_diag or {}).get("is_closed_loop")
        esc_len = (dir_diag or {}).get("escaped_length")
        int_len = (dir_diag or {}).get("interior_length")
        geo_log = f"sol={sol_v:.3f}(≥{SOLIDITY_THRESHOLD})" if sol_v is not None else ""
        geo_log += f"  AR={ar_v:.2f}(≥{ASPECT_RATIO_THRESHOLD})" if ar_v is not None else ""
        geo_log += f"  holes={int(hole_v)}"
        dir_log = ""
        if sin_v is not None:
            dir_log += f"  sin={sin_v:.2f}(≤{CURVATURE_THRESHOLD})"
        if loop_v is not None:
            dir_log += f"  loop={'Y' if loop_v else 'N'}"
        if esc_len is not None and int_len is not None:
            esc_pct = f"{esc_len / int_len:.1%}" if int_len > 1e-9 else "0.0%"
            dir_log += f"  esc={esc_pct}(≤{ESCAPE_THRESHOLD:.0%})"
        if success:
            print(f"    [{tag}] directional ✓  |  {geo_log}{dir_log}")
            feature.skeleton_overlay_data["auto_decision"] = {
                "geo_method": "directional", "geo_reason": geo_reason,
                "final_method": "directional", "dir_reason": dir_reason,
                **geo_diag,
                **dir_diag,
            }
            directional_succeeded = True
        else:
            print(f"    [{tag}] directional ✗  →  Lee  |  {geo_log}{dir_log}")
    else:
        tag = "auto" if allow_multi_branch else "dir+sb"

    if not directional_succeeded:
        # Stage 3 — Lee path.
        _dispatch_lee_path(
            feature, grid, skeleton, x_min, y_min, effective_resolution,
            geo_method=geo_method, geo_reason=geo_reason, geo_diag=geo_diag,
            dir_diag=dir_diag, allow_multi_branch=allow_multi_branch,
            diameter=diameter,
        )


def _dispatch_line_feature(feature):
    """
    Populate a line/polyline feature's branches directly from its geometry.

    Line features skip rasterisation and skeletonisation entirely — the input
    geometry IS the centreline.  A LineString becomes a single Branch (id=0);
    a MultiLineString produces one Branch per component.

    The branches are passed through _post_process_skeleton for Gaussian
    smoothing, arc-length resampling, and output decimation, identical to
    polygon features.
    """
    geom = feature.polygon

    if isinstance(geom, MultiLineString):
        line_list = list(geom.geoms)
    else:
        line_list = [geom]

    branches = []
    for bid, line in enumerate(line_list):
        coords = list(line.coords)
        if len(coords) >= 2:
            branches.append(Branch(id=bid, centerline=coords))

    feature.branches = branches
    feature.skeleton_overlay_data = {
        "raster_grid":     None,
        "skeleton_grid":   None,
        "raw_branches":    [],
        "merged_branches": [],
        "skeleton_method": "line_input",
        "x_min":           0.0,
        "y_min":           0.0,
        "resolution":      config.RASTER_RESOLUTION,
    }
    return _post_process_skeleton(feature)


def _dispatch_skeletoniser(feature):
    """
    Compute centreline skeleton branches for a single polygon feature and
    write the results back into the feature in place.

    This is the main entry point for skeletonisation.  It selects and
    executes the appropriate strategy according to ``SKELETONISATION_METHOD``:

    - **``"auto"``**: Full automatic decision tree — geometry gate →
      directional attempt → Lee single_branch or multi_branch.  See
      :func:`_select_skeletonisation_method` (stage 1),
      :func:`_validate_directional_skeleton` (stage 2), and
      :func:`_select_single_or_multi_branch_skeletonisation` (stage 3).

    - **``"directional"``**: Attempts :func:`_directional_skeletonise` with
      no fallback.  If validation fails (high curvature, closed loop, or too
      many skeleton vertices escaping the polygon) the feature receives no
      skeleton and a warning is printed.  Use ``"directional_and_single_branch"``
      for a robust version that falls back to Lee single_branch.

    - **``"directional_and_single_branch"``**: Runs the same geometry gate
      and directional attempt as ``"auto"`` but forces ``single_branch`` when
      the Lee path is taken — ``multi_branch`` is never an end-point.

    - **``"single_or_multi_branch"``**: Lee-only path.  Extracts and prunes
      skeleton branches, then uses
      :func:`_select_single_or_multi_branch_skeletonisation` to choose
      ``single_branch`` or ``multi_branch`` based on ``BRANCHING_THRESHOLD``.
      No directional stage is run.

    - **``"single_branch"``**: Calls :func:`_single_branch_skeletonise`
      directly.

    - **``"multi_branch"``**: Calls :func:`_multi_branch_skeletonise`
      directly.

    A one-line ``[auto]`` log message is printed for every feature processed
    in ``"auto"`` mode, summarising the geometry metrics and the chosen method.

    Args:
        feature: A :class:`Feature` dataclass instance with a ``polygon``
            attribute set.

    Returns:
        Feature: ``feature``, mutated in place.

    Raises:
        ImportError: If scikit-image is not installed.
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for skeletonisation.")

    # ── Line / polyline input — skip rasterisation and skeletonisation ────────
    if _is_line_feature(feature):
        return _dispatch_line_feature(feature)

    # ── Raster-image input — use stored pixel mask directly ──────────────────
    # For features parsed from raster images (JPEG/PNG/TIFF), the polygon was
    # extracted by find_contours which self-intersects at any 1-pixel-wide neck.
    # buffer(0) then splits the polygon at the neck, permanently losing that
    # connection.  We bypass the polygon→rasterise round-trip entirely and use
    # the original pixel mask, which retains the true topology.
    if getattr(feature, 'raw_mask', None) is not None:
        c_min, r_min = feature.raw_mask_origin
        mask_crop    = feature.raw_mask

        # Dilate by RASTER_BUFFER / RASTER_RESOLUTION pixels to ensure thin necks
        # survive morphological thinning (mirrors what _add_rasterisation_buffer
        # does for vector features, but applied directly in pixel space).
        buf_px = max(0, round(config.RASTER_BUFFER / config.RASTER_RESOLUTION))
        if buf_px > 0 and HAS_SCIPY:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * buf_px + 1, 2 * buf_px + 1), dtype=bool)
            mask_crop = binary_dilation(mask_crop, structure=struct)

        # Pad by 5 pixels on all sides so the skeleton never touches the array
        # boundary (same padding as _rasterise_polygon uses).
        pad  = 5
        grid = np.pad(mask_crop, pad, mode='constant', constant_values=False)
        x_min = float(c_min - pad)
        y_min = float(r_min - pad)
        effective_resolution = 1.0   # 1 pixel = 1 world unit for raster images

        skeleton = _skimage_skeletonize(grid)

        # For geometry-gate dispatch methods that require a polygon argument,
        # fall back to the (possibly split) feature polygon — it is used only
        # for the geometry gate heuristic, not for rasterisation.
        polygon = feature.polygon

    else:
        polygon = _add_rasterisation_buffer(feature.polygon, feature.id)

        # Rasterise and thin — shared by every code path.
        grid, x_min, y_min, effective_resolution = _rasterise_polygon(polygon)  # effective_resolution = effective (auto-scaled) raster resolution
        skeleton = _skimage_skeletonize(grid)                           # one-pixel-wide skeleton array produced by Lee's thinning algorithm

    method = SKELETONISATION_METHOD

    # ── Explicit single_branch mode ───────────────────────────────────────────
    if method == "single_branch":
        _single_branch_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution)

    # ── Explicit multi_branch mode ────────────────────────────────────────────
    elif method == "multi_branch":
        _multi_branch_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution)

    # ── Directional-only mode (no geometry gate, no fallback) ─────────────────
    # Attempts the PCA cross-section directional skeleton.  There is no
    # geometry gate and no Lee fallback — the directional skeleton is always
    # used regardless of whether validation passes.  If validation fails, a
    # warning is printed and the (possibly erroneous) skeleton is written to the
    # feature so it is never silently absent from the output.  Use
    # "directional_and_single_branch" if a robust Lee fallback is preferred.
    elif method == "directional":
        success, sinuosity, world_centreline, diameter, reason, _dir_diag = \
            _directional_skeletonise(feature, grid, skeleton, x_min, y_min, effective_resolution)
        if not success:
            print(f"    ⚠ [directional] {reason} — feature may be malformed; "
                  f"outputting directional skeleton regardless.")
            # Populate the feature with the raw directional result even though
            # validation failed.  world_centreline may be None if the skeleton
            # produced fewer than 2 vertices; in that case branches is empty.
            if world_centreline and len(world_centreline) >= 2:
                feature.branches = [Branch(id=0, centerline=list(world_centreline))]
            else:
                feature.branches = []
            feature.skeleton_overlay_data = {
                "raster_grid":       _skeleton_overlay_grid(grid),
                "skeleton_grid":     _skeleton_overlay_grid(skeleton),
                "raw_branches":      [],
                "merged_branches":   [],
                "directional_world": world_centreline,
                "skeleton_method":   "directional (validation failed)",
                "sinuosity":         sinuosity,
                "x_min":             x_min,
                "y_min":             y_min,
                "resolution":        effective_resolution,
            }

    # ── directional_and_single_branch: geometry gate + dir attempt + Lee sb ───
    # Follows the identical decision tree as "auto" (geometry gate → directional
    # attempt) but the Lee fallback always produces single_branch — multi_branch
    # is never an end-point of this mode.
    elif method == "directional_and_single_branch":
        _dispatch_with_geometry_gate(
            feature, grid, skeleton, x_min, y_min, effective_resolution, polygon,
            allow_multi_branch=False,
        )

    # ── single_or_multi_branch: Lee-only path, no directional stage ───────────
    # Runs the Lee skeleton, prunes stubs, and uses
    # _select_single_or_multi_branch_skeletonisation to decide the end-point.
    # The geometry gate and directional attempt are both skipped entirely.
    elif method == "single_or_multi_branch":
        _dispatch_lee_path(
            feature, grid, skeleton, x_min, y_min, effective_resolution,
            geo_method="lee", geo_reason="single_or_multi_branch mode (no directional stage)",
            geo_diag={}, dir_diag=None,
            allow_multi_branch=True,
        )

    # ── Auto mode: full decision tree ─────────────────────────────────────────
    else:
        _dispatch_with_geometry_gate(
            feature, grid, skeleton, x_min, y_min, effective_resolution, polygon,
            allow_multi_branch=True,
        )

    # ── Post-processing ───────────────────────────────────────────────────────
    # Smooth, resample, and decimate every branch, regardless of which
    # skeletonisation method was used.
    return _post_process_skeleton(feature)
