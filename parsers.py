"""
parsers.py — Everything that reads input files and returns lists of Features.

Handles Shapefile, SVG, PDF, and raster image inputs.  Optional dependencies
(geopandas, Pillow, PyMuPDF, scikit-image) are detected at import time with
graceful fallback warnings.
"""

import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import (
    GeometryCollection, LinearRing, LineString, MultiLineString,
    MultiPoint, MultiPolygon, Point, Polygon,
)

import config
from data_models import Feature

# --- optional dependencies (warn but don't crash at import time) --------------

try:    # geopandas
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: geopandas not found — shapefile parsing unavailable.")
    print("         Install with:  pip install geopandas")

try:    # scikit-image
    from skimage.measure import find_contours, label
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found — image parsing unavailable.")
    print("         Install with:  pip install scikit-image")

try:    # Pillow
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not found — image parsing unavailable.")
    print("         Install with:  pip install pillow")

try:    # PyMuPDF
    import fitz          # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not found — PDF parsing unavailable.")
    print("         Install with:  pip install PyMuPDF")

# Compiled regex for a floating-point number as it appears in SVG attribute strings.
_SVG_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_SVG_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)   # (a, b, c, d, e, f)
# SVG matrix(a,b,c,d,e,f) maps (x,y) → (a·x + c·y + e,  b·x + d·y + f


# Parse Shapefile
def parse_shapefile(filepath):
    """
    Parse polygon and line features from an ESRI Shapefile (or any
    OGR-supported format) into a list of Feature dataclass instances.

    Each polygon part becomes one feature. MultiPolygon geometries are exploded
    into individual parts with IDs suffixed ``_0``, ``_1``, …; single-polygon
    rows keep a plain, unsuffixed ID. Feature IDs are resolved in priority order:
    ``"id"`` → ``"FID"`` → ``"fid"`` → GeoDataFrame integer index.

    LineString and MultiLineString geometries are handled identically: each
    component line becomes one Feature with the line stored in the ``polygon``
    field (the field name is kept for compatibility with the rest of the
    pipeline).  MultiLineString parts receive an ``_N`` suffix in their ID.
    Width measurement is skipped automatically for these features because
    there is no enclosing polygon.

    All features carry ``flip_y=True`` because Shapefile coordinates use a
    north-up Cartesian system, opposite to the screen-space convention assumed
    by the rest of the pipeline.

    Args:
        filepath: Path to the ``.shp`` file or any format accepted by
            ``geopandas.read_file``.

    Returns:
        list[Feature]: Feature dataclass instances with fields ``id``,
        ``polygon`` (holds Polygon or LineString geometry), and ``flip_y``
        (always ``True``).

    Raises:
        ImportError: If ``geopandas`` is not installed.
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for shapefile parsing.")

    gdf      = gpd.read_file(filepath)

    # CRS warning for geographic coordinates
    crs = gdf.crs
    if crs is not None:
        try:
            if crs.is_geographic:
                print(f"  Warning: input CRS appears to be geographic (lat/lon): {crs.to_string()}.")
                print("           Coordinates are in degrees; consider reprojecting to a projected CRS.")
                print("           RASTER_RESOLUTION should be set to ~9e-9 (≈1 mm) for degree units.")
        except Exception:
            pass

    features = []
    _z_warned = [False]  # one-time Z-coordinate warning flag

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Drop Z coordinates from 3D geometries, with a one-time warning
        if geom is not None and not geom.is_empty and geom.has_z:
            if not _z_warned[0]:
                print("  Warning: 3D geometry detected — Z coordinates will be dropped.")
                _z_warned[0] = True
            from shapely.ops import transform as _shp_transform
            geom = _shp_transform(lambda x, y, z=None: (x, y), geom)

        fid = row.get("id", row.get("FID", row.get("fid", idx)))

        if isinstance(geom, (Polygon, MultiPolygon)):
            geom_list = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

            for part_idx, poly in enumerate(geom_list):
                if not isinstance(poly, Polygon) or poly.is_empty:
                    continue

                part_id = f"{fid}_{part_idx}" if len(geom_list) > 1 else str(fid)
                features.append(Feature(id=part_id, polygon=poly, flip_y=True))

        elif isinstance(geom, (LineString, MultiLineString)):
            lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
            for part_idx, line in enumerate(lines):
                if line.is_empty:
                    continue
                part_id = f"{fid}_{part_idx}" if len(lines) > 1 else str(fid)
                features.append(Feature(id=part_id, polygon=line, flip_y=True))

        elif isinstance(geom, LinearRing):
            # A LinearRing is already a closed polygon ring
            poly = Polygon(geom)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                features.append(Feature(id=str(fid), polygon=poly, flip_y=True))

        elif isinstance(geom, (Point, MultiPoint)):
            # Points become 1-world-unit diameter circles
            pts = list(geom.geoms) if isinstance(geom, MultiPoint) else [geom]
            for part_idx, pt in enumerate(pts):
                part_id = f"{fid}_{part_idx}" if len(pts) > 1 else str(fid)
                circle = pt.buffer(0.5, resolution=32)
                features.append(Feature(id=part_id, polygon=circle, flip_y=True))

        elif isinstance(geom, GeometryCollection):
            # Recursively decompose — create sub-features for each component
            sub_features = []
            for part_idx, sub_geom in enumerate(geom.geoms):
                if sub_geom.is_empty:
                    continue
                part_id = f"{fid}_{part_idx}"
                if isinstance(sub_geom, (Polygon, MultiPolygon)):
                    sub_list = list(sub_geom.geoms) if isinstance(sub_geom, MultiPolygon) else [sub_geom]
                    for p in sub_list:
                        if isinstance(p, Polygon) and not p.is_empty:
                            sub_features.append(Feature(id=part_id, polygon=p, flip_y=True))
                elif isinstance(sub_geom, (LineString, MultiLineString)):
                    sub_lines = list(sub_geom.geoms) if isinstance(sub_geom, MultiLineString) else [sub_geom]
                    for l in sub_lines:
                        if not l.is_empty:
                            sub_features.append(Feature(id=part_id, polygon=l, flip_y=True))
                elif isinstance(sub_geom, (Point, MultiPoint)):
                    sub_pts = list(sub_geom.geoms) if isinstance(sub_geom, MultiPoint) else [sub_geom]
                    for p in sub_pts:
                        sub_features.append(Feature(id=part_id, polygon=p.buffer(0.5, resolution=32), flip_y=True))
            features.extend(sub_features)

    return features

def parse_svg(filepath):
    """
    Parse closed polygon shapes from an SVG file into a list of feature dicts.

    Opens the SVG at ``filepath``, walks the entire element tree, and converts
    ``<polygon>``, ``<rect>``, and ``<path>`` elements into Shapely
    ``Polygon`` objects. The following aspects of SVG geometry are handled:

    **Coordinate transforms** — ``transform`` attributes are parsed at every
    level of the element hierarchy and composed into a cumulative
    transformation matrix (CTM) using matrix multiplication, so that nested
    ``translate``, ``scale``, ``rotate``, ``skewX/Y``, and ``matrix``
    operations are all correctly accumulated before coordinates are mapped to
    world space. The root ``<svg>`` element's ``viewBox`` is also factored in
    as an initial viewport transform, ensuring that files exported by
    Illustrator, Inkscape, QGIS, and similar tools are parsed consistently.

    **Element types**:

    - ``<polygon>`` — The ``points`` attribute is tokenised with a float regex
      and converted directly to a coordinate list (always treated as closed).
    - ``<rect>`` — The ``x``, ``y``, ``width``, and ``height`` attributes are
      used to synthesise an explicit five-point closed ring.
    - ``<path>`` — The ``d`` attribute is forwarded to
      :func:`_parse_svg_path_d`, which handles all standard SVG path commands
      including cubic and quadratic Bézier curves (approximated by sampling)
      and arc segments (approximated as straight lines to the endpoint).
      Paths that contain a Z/z close-path command are treated as polygons;
      open paths are treated as line features.
    - ``<line>`` — The ``x1``, ``y1``, ``x2``, ``y2`` attributes define a
      two-point line segment, stored as a LineString feature.
    - ``<polyline>`` — The ``points`` attribute defines an open multi-segment
      line, stored as a LineString feature.

    **Validity repair** — After applying the CTM, each polygon is passed
    through ``Polygon.buffer(0)`` if Shapely reports it as invalid (e.g. a
    self-intersecting ring), and zero-area or empty results are discarded.

    Subtrees rooted at ``<defs>``, ``<symbol>``, ``<mask>``, or ``<clipPath>``
    elements are skipped entirely because their contents are not directly
    rendered.

    Feature IDs are taken from the element's ``id`` attribute when present;
    otherwise a monotonically increasing integer counter is used.

    All returned features carry a ``_flip_y`` flag set to ``False`` because
    SVG coordinates already use a top-left origin (Y increases downward),
    matching the screen-space convention used by the rest of the pipeline.

    Args:
        filepath (str | os.PathLike): Path to the ``.svg`` file to parse.

    Returns:
        list[Feature]: Feature dataclass instances, one per valid polygon or
        line element encountered in document order. Each instance contains:

        - ``id`` (*str*): The element's ``id`` attribute, or a string
          representation of an auto-incremented integer counter if the
          attribute is absent.
        - ``polygon`` (*shapely.geometry.Polygon* or
          *shapely.geometry.LineString*): The geometry of the feature in
          world-space coordinates after all transforms have been applied.
          ``<polygon>``, ``<rect>``, and closed ``<path>`` elements produce
          Polygon objects; ``<line>``, ``<polyline>``, and open ``<path>``
          elements produce LineString objects.
        - ``flip_y`` (*bool*): Always ``False`` for SVG-sourced features.

    Raises:
        xml.etree.ElementTree.ParseError: If ``filepath`` is not
            well-formed XML.
        FileNotFoundError: If ``filepath`` does not exist.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    def strip_ns(tag):
        # Strip the '{namespace}' prefix so tags match by local name.
        return tag.split("}")[-1] if "}" in tag else tag

    features = []

    fid = [0]   # mutable counter incremented by the _walk closure

    # Derive an initial current transformation matrix (CTM) from the root
    # <svg> element's viewBox and width/height attributes. This converts from
    # the file's internal user-unit coordinate space to the output coordinate
    # space before any per-element transforms are applied.
    root_tag = strip_ns(root.tag)
    root_ctm = _svg_viewport_transform(root) if root_tag == "svg" else _SVG_IDENTITY

    # These container elements hold reusable definitions or clipping regions
    # that are referenced elsewhere in the document; their children are not
    # rendered directly, so descending into them would produce duplicate or
    # invisible geometry.  Note: "symbol" is intentionally excluded here so
    # we can handle it explicitly in _walk (to register it without recursing).
    _SKIP_TAGS = {"defs", "mask", "clipPath"}

    # Pre-pass: register all <symbol> elements by id so that <use> can
    # reference them later during the main walk.
    _symbol_registry = {}

    def _register_symbols(elem):
        tag = strip_ns(elem.tag)
        if tag == "symbol":
            sym_id = elem.get("id")
            if sym_id:
                _symbol_registry[sym_id] = elem
        for child in elem:
            _register_symbols(child)

    _register_symbols(root)

    # Pre-pass: extract CSS class rules from all <style> elements.
    # Illustrator, Inkscape, and other SVG exporters commonly define fill and
    # stroke via CSS classes (e.g. `.st4 { fill: none; }`) rather than as
    # inline style= or presentation attributes.  Without parsing these rules,
    # every element would fall back to the inherited default fill of "black",
    # misclassifying unfilled shapes (outlines, borders) as solid polygons.
    #
    # Only class selectors (starting with ".") are supported; ID selectors,
    # element selectors, and pseudo-classes are ignored.  Combined selectors
    # such as `.st2, .st3 { fill: #fff; }` are split on commas so each class
    # name is registered separately.
    _css_rules: dict = {}   # { "classname": { "property": "value", ... } }
    for _se in root.iter():
        if strip_ns(_se.tag) == "style":
            _css_text = _se.text or ""
            for _m in re.finditer(r'([^{]+)\{([^}]*)\}', _css_text):
                _selector_part = _m.group(1)
                _body          = _m.group(2)
                # Parse declarations: "property: value; ..."
                _props: dict = {}
                for _decl in _body.split(';'):
                    if ':' in _decl:
                        _prop, _val = _decl.split(':', 1)
                        _props[_prop.strip().lower()] = _val.strip()
                # Register each class selector individually
                for _sel in _selector_part.split(','):
                    _sel = _sel.strip()
                    if _sel.startswith('.'):
                        _cn = _sel[1:].strip()
                        if _cn:
                            _css_rules.setdefault(_cn, {}).update(_props)

    def _walk(elem, ctm, fill="black", fill_opacity=1.0):
        """Recursively visit every element, accumulating transforms and fill style."""
        tag     = strip_ns(elem.tag)
        elem_id = elem.get("id", str(fid[0]))

        # Compose any transform declared on this element into the running CTM.
        # The result is a new matrix that maps the element's local coordinates
        # directly into the root coordinate space, accounting for every
        # ancestor transform along the way.
        t_str = elem.get("transform", "")
        if t_str.strip():
            ctm = _compose_svg_matrix(ctm, _parse_svg_transform(t_str))

        # Drop elements with display:none, visibility:hidden, or opacity:0
        _style = elem.get("style", "")
        if (re.search(r"display\s*:\s*none", _style) or
                re.search(r"visibility\s*:\s*hidden", _style) or
                re.search(r"opacity\s*:\s*0(?:\.0+)?\b", _style)):
            return
        if (elem.get("display", "").strip().lower() == "none" or
                elem.get("visibility", "").strip().lower() == "hidden"):
            return
        _opacity_attr = elem.get("opacity", "1")
        try:
            if float(_opacity_attr) == 0.0:
                return
        except (ValueError, TypeError):
            pass

        # ── Compute effective fill for this element ───────────────────────────
        # SVG fill is an inherited property: each element starts with the
        # parent's computed fill and may override it via presentation attributes
        # or inline style.  We propagate (fill, fill_opacity) down the tree so
        # that <path> elements inside a filled <g> are correctly classified as
        # polygons even when they carry no fill attribute of their own.
        #
        # Priority (highest → lowest): inline style > presentation attribute >
        # inherited value.  "inherit" and "currentColor" are treated as
        # pass-through (keep the inherited value).
        eff_fill         = fill
        eff_fill_opacity = fill_opacity

        # 1. Presentation attributes (lower priority; overridden by inline style)
        _fa = elem.get("fill")
        if _fa is not None:
            _fv = _fa.strip().lower()
            if _fv not in ("inherit", "currentcolor"):
                eff_fill = _fv
        _foa = elem.get("fill-opacity")
        if _foa is not None:
            try:
                eff_fill_opacity = float(_foa.strip())
            except ValueError:
                pass

        # 1.5. CSS class rules (override presentation attributes; overridden by
        #      inline style).  The element's class= attribute may contain multiple
        #      space-separated class names; each is looked up in _css_rules.  Later
        #      class names in the list take precedence over earlier ones (mirrors
        #      standard CSS cascade behaviour for same-specificity rules).
        _ca = elem.get("class", "")
        if _ca:
            for _cls in _ca.split():
                _cls_props = _css_rules.get(_cls, {})
                _cv = _cls_props.get("fill")
                if _cv is not None and _cv.lower() not in ("inherit", "currentcolor"):
                    eff_fill = _cv.lower()
                _cov = _cls_props.get("fill-opacity")
                if _cov is not None:
                    try:
                        eff_fill_opacity = float(_cov)
                    except ValueError:
                        pass

        # 2. Inline style (higher priority; overrides presentation attributes)
        _fm = re.search(r"(?:^|;)\s*fill\s*:\s*([^;]+)", _style, re.IGNORECASE)
        if _fm:
            _fv = _fm.group(1).strip().lower()
            if _fv not in ("inherit", "currentcolor"):
                eff_fill = _fv
        _fom = re.search(r"(?:^|;)\s*fill-opacity\s*:\s*([^;]+)", _style, re.IGNORECASE)
        if _fom:
            try:
                eff_fill_opacity = float(_fom.group(1).strip())
            except ValueError:
                pass

        # A shape is considered visually filled if the computed fill colour is
        # not transparent and fill-opacity is greater than zero.  White fills
        # (#fff / #ffffff / white / rgb(255,255,255)) are treated the same as
        # "none" — they make the shape invisible against a white canvas and
        # indicate the author intended an outline, not a solid filled polygon.
        _NOT_FILLED_VALUES = {
            "none", "transparent",
            "#fff", "#ffffff", "white",
            "rgb(255,255,255)", "rgb(255, 255, 255)",
        }
        is_filled = eff_fill.lower() not in _NOT_FILLED_VALUES and eff_fill_opacity > 0

        # Handle <symbol>: already registered in pre-pass; don't recurse here
        # to avoid rendering symbol contents outside of a <use> context.
        if tag == "symbol":
            return

        # Handle <use>: instantiate the referenced symbol or element.
        elif tag == "use":
            href = elem.get("href") or elem.get("{http://www.w3.org/1999/xlink}href", "")
            ref_id = href.lstrip("#")
            x_off = float(elem.get("x", 0))
            y_off = float(elem.get("y", 0))
            use_translate = (1.0, 0.0, 0.0, 1.0, x_off, y_off)
            use_ctm = _compose_svg_matrix(ctm, use_translate)
            target = _symbol_registry.get(ref_id)
            if target is not None:
                for child in target:
                    _walk(child, use_ctm, eff_fill, eff_fill_opacity)
            return

        # 'coords' will be populated by whichever element-type branch matches;
        # it holds raw local-space (x, y) pairs before the CTM is applied.
        # 'is_closed_shape' distinguishes polygon-producing elements (True)
        # from line-producing elements (False).
        coords          = None
        is_closed_shape = True   # default; overridden for line elements

        if tag == "polygon":
            # The SVG 'points' attribute is a whitespace/comma-separated run of
            # floats. The regex captures every number (including scientific
            # notation) regardless of separator, then pairs them up as (x, y).
            nums   = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
                                 elem.get("points", ""))
            coords = [(float(nums[j]), float(nums[j + 1]))
                      for j in range(0, len(nums) - 1, 2)]
            is_closed_shape = True

        elif tag == "circle":
            cx_v = float(elem.get("cx", 0))
            cy_v = float(elem.get("cy", 0))
            r    = float(elem.get("r", 0))
            if r > 0:
                n_pts = 64
                coords = [(cx_v + r * math.cos(2*math.pi*k/n_pts),
                           cy_v + r * math.sin(2*math.pi*k/n_pts))
                          for k in range(n_pts)]
                is_closed_shape = True
            else:
                # r=0: degenerate point; represent as single-point list so the
                # single-point handler below expands it to a 1-unit circle.
                coords = [(cx_v, cy_v)]
                is_closed_shape = True

        elif tag == "ellipse":
            cx_v = float(elem.get("cx", 0))
            cy_v = float(elem.get("cy", 0))
            rx   = float(elem.get("rx", 0))
            ry   = float(elem.get("ry", 0))
            if rx > 0 and ry > 0:
                n_pts = 64
                coords = [(cx_v + rx * math.cos(2*math.pi*k/n_pts),
                           cy_v + ry * math.sin(2*math.pi*k/n_pts))
                          for k in range(n_pts)]
                is_closed_shape = True

        elif tag == "path":
            d = elem.get("d", "")
            if d:
                # Delegate to the multi-subpath parser; handle each subpath
                # independently.  coords is set to None to signal that the
                # path branch handles feature creation internally.
                subpaths = _parse_svg_path_d(d)
                for sp_coords, sp_closed in subpaths:
                    if not sp_coords:
                        continue

                    # A subpath is treated as a closed polygon if ANY of:
                    #   1. The path data contained an explicit Z/z command
                    #      (sp_closed=True from the parser; Z is processed
                    #      immediately when the letter token is consumed).
                    #   2. The computed fill is non-transparent.  SVG renderers
                    #      visually fill open paths by drawing an implicit line
                    #      from the last point back to the first; we mirror this
                    #      behaviour and treat any filled path as a polygon.
                    #      This is safe because fill-classification is now driven
                    #      by CSS class rules (parsed from <style> elements), so
                    #      is_filled is accurate: stroke-only paths carry
                    #      fill:none or fill:#fff and correctly yield
                    #      is_filled=False, while genuinely solid polygon paths
                    #      carry a dark fill and yield is_filled=True.
                    is_closed_shape = sp_closed
                    if not is_closed_shape and is_filled and len(sp_coords) >= 3:
                        is_closed_shape = True  # filled open path → treat as polygon

                    a2, b2, c2, d2, e2, f2 = ctm
                    tx_sp = [(a2*px + c2*py + e2, b2*px + d2*py + f2)
                             for px, py in sp_coords]
                    # Single-point subpath → 1-world-unit diameter circle
                    if len(tx_sp) == 1:
                        px0, py0 = tx_sp[0]
                        n_pts = 32
                        r = 0.5
                        tx_sp = [(px0 + r * math.cos(2*math.pi*k/n_pts),
                                  py0 + r * math.sin(2*math.pi*k/n_pts))
                                 for k in range(n_pts)]
                        is_closed_shape = True
                    if is_closed_shape and is_filled and len(tx_sp) >= 3:
                        # Closed + filled → Polygon, unless the path is
                        # self-intersecting (buffer(0) → MultiPolygon).
                        # Self-intersecting closed paths typically arise from
                        # Illustrator variable-width stroke outlines, which
                        # trace both edges of a stroke as a single closed path.
                        # These represent a line, not a filled region, so we
                        # fall back to a closed LineString in that case.
                        try:
                            poly = Polygon(tx_sp)
                            if not poly.is_valid:
                                fixed = poly.buffer(0)
                                if isinstance(fixed, MultiPolygon):
                                    # Self-intersecting compound path → LineString
                                    ring = list(tx_sp)
                                    if ring[0] != ring[-1]:
                                        ring.append(ring[0])
                                    line_geom = LineString(ring)
                                    if not line_geom.is_empty:
                                        features.append(Feature(id=elem_id, polygon=line_geom))
                                        fid[0] += 1
                                    continue
                                poly = fixed
                            if poly.is_valid and not poly.is_empty and poly.area > 0:
                                features.append(Feature(id=elem_id, polygon=poly))
                                fid[0] += 1
                        except Exception:
                            pass
                    elif is_closed_shape and not is_filled and len(tx_sp) >= 2:
                        # Closed + unfilled → closed LineString (outline / border)
                        try:
                            ring = list(tx_sp)
                            if ring[0] != ring[-1]:
                                ring.append(ring[0])  # close the ring
                            line_geom = LineString(ring)
                            if not line_geom.is_empty:
                                features.append(Feature(id=elem_id, polygon=line_geom))
                                fid[0] += 1
                        except Exception:
                            pass
                    elif not is_closed_shape and len(tx_sp) >= 2:
                        try:
                            line_geom = LineString(tx_sp)
                            if not line_geom.is_empty:
                                features.append(Feature(id=elem_id, polygon=line_geom))
                                fid[0] += 1
                        except Exception:
                            pass
                coords = None  # signal that path was already handled

        elif tag == "rect":
            # A rectangle has no explicit vertex list in SVG, so we synthesise
            # a five-point closed ring from its position and dimensions. The
            # fifth point repeats the first to satisfy Shapely's closed-ring
            # requirement.
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            coords = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
            is_closed_shape = True

        elif tag == "line":
            # A <line> element defines a two-point open line segment.
            x1 = float(elem.get("x1", 0)); y1 = float(elem.get("y1", 0))
            x2 = float(elem.get("x2", 0)); y2 = float(elem.get("y2", 0))
            coords = [(x1, y1), (x2, y2)]
            is_closed_shape = False

        elif tag == "polyline":
            # A <polyline> element defines an open multi-segment line via a
            # space/comma-separated list of coordinate pairs.
            nums   = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
                                 elem.get("points", ""))
            coords = [(float(nums[j]), float(nums[j + 1]))
                      for j in range(0, len(nums) - 1, 2)]
            is_closed_shape = False

        if coords:
            # Apply the accumulated CTM to every coordinate pair. The 2-D
            # affine transform is: x' = a*x + c*y + e,  y' = b*x + d*y + f,
            # where (a, b, c, d, e, f) are the six independent components of
            # the 3×3 homogeneous matrix stored in column-major order.
            a, b, c, d, e, f = ctm
            tx = [(a*px + c*py + e, b*px + d*py + f) for px, py in coords]

            # Single-point → represent as a 1-world-unit diameter circle polygon
            if len(tx) == 1:
                px0, py0 = tx[0]
                n_pts = 32
                r = 0.5  # radius = 0.5 → diameter = 1 world unit
                tx = [(px0 + r * math.cos(2*math.pi*k/n_pts),
                       py0 + r * math.sin(2*math.pi*k/n_pts))
                      for k in range(n_pts)]
                is_closed_shape = True

            if is_closed_shape and is_filled and len(tx) >= 3:
                # Closed + filled → Polygon
                try:
                    poly = Polygon(tx)

                    # Self-intersecting rings (a "bow-tie" polygon) are technically
                    # invalid but common in exported SVG files. buffer(0) is a
                    # standard Shapely idiom that resolves most self-intersections
                    # by computing a zero-distance outward offset, returning one or
                    # more valid, non-self-intersecting polygons.
                    if not poly.is_valid:
                        poly = poly.buffer(0)

                    # Discard anything that still failed validation, collapsed to a
                    # line or point (empty), or has no area (e.g. a degenerate
                    # triangle with collinear vertices). Such shapes cannot be
                    # matched or rendered meaningfully.
                    if poly.is_valid and not poly.is_empty and poly.area > 0:
                        # flip_y=False because SVG's Y axis already increases
                        # downward, matching screen/canvas coordinate conventions,
                        # so no inversion is needed before rendering.
                        features.append(Feature(id=elem_id, polygon=poly))
                        fid[0] += 1
                except Exception:
                    # Shapely can raise on pathological inputs (too few unique
                    # points, NaN coordinates, etc.). Silently discard those
                    # shapes so a single bad element doesn't abort the whole parse.
                    pass

            elif is_closed_shape and not is_filled and len(tx) >= 2:
                # Closed + unfilled → closed LineString (e.g. a border rectangle
                # or an outlined circle with fill:none).  The shape is an outline,
                # not a filled polygon, so it should be treated as a line feature.
                try:
                    ring = list(tx)
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])  # ensure the ring is explicitly closed
                    line_geom = LineString(ring)
                    if not line_geom.is_empty:
                        features.append(Feature(id=elem_id, polygon=line_geom))
                        fid[0] += 1
                except Exception:
                    pass

            elif not is_closed_shape and len(tx) >= 2:
                try:
                    line_geom = LineString(tx)
                    if not line_geom.is_empty:
                        features.append(Feature(id=elem_id, polygon=line_geom))
                        fid[0] += 1
                except Exception:
                    pass

        # Recurse into child elements unless this is a non-rendered container.
        # Skipping _SKIP_TAGS here (rather than at the top of _walk) ensures
        # that the container element itself is also never processed for geometry.
        # Pass the effective fill down so children inherit it correctly.
        if tag not in _SKIP_TAGS:
            for child in elem:
                _walk(child, ctm, eff_fill, eff_fill_opacity)

    _walk(root, root_ctm)

    # ── Artboard / canvas rectangle filter ───────────────────────────────────
    # SVG editors (Illustrator, Inkscape, Affinity Designer, etc.) often emit
    # a filled rectangle that covers the entire canvas as the first or last
    # drawn element — the document background or artboard border.  This is
    # not a real feature and would produce a misleading skeleton if processed.
    #
    # To detect it we need the viewport dimensions in world space.  After the
    # root_ctm is applied, the page runs from (0, 0) to (viewport_w, viewport_h)
    # where viewport_w/h are the SVG root element's width/height attributes
    # (falling back to the viewBox dimensions if width/height are absent).
    vb_raw = root.get("viewBox") or root.get("viewbox", "")
    vb_nums = [float(v) for v in _SVG_FLOAT_RE.findall(vb_raw)] if vb_raw.strip() else []
    if len(vb_nums) >= 4:
        _vb_x, _vb_y, _vb_w, _vb_h = vb_nums[:4]

        def _vp_dim(attr, default):
            s = root.get(attr)
            if not s:
                return default
            n = _SVG_FLOAT_RE.findall(s)
            return float(n[0]) if n else default

        _vp_w = _vp_dim("width",  _vb_w)
        _vp_h = _vp_dim("height", _vb_h)

        if _vp_w > 0 and _vp_h > 0:
            # Scale the viewBox dimensions to world space (same arithmetic as
            # _svg_viewport_transform so the bounds are consistent with how all
            # feature coordinates were transformed).
            _sx = _vp_w / _vb_w if _vb_w > 0 else 1.0
            _sy = _vp_h / _vb_h if _vb_h > 0 else 1.0
            page_w = _vb_w * _sx   # == _vp_w
            page_h = _vb_h * _sy   # == _vp_h

            filtered_features = []
            for feat in features:
                # _is_artboard_rect handles all geometry types: Polygon,
                # MultiPolygon, and LineString (closed outlines / white-filled
                # rects).  Any feature that covers the artboard area is dropped
                # regardless of whether it is filled, unfilled, or white-filled.
                if _is_artboard_rect(feat.polygon, 0.0, 0.0, page_w, page_h):
                    print(f"  [svg] Skipping artboard/canvas rectangle "
                          f"(feature {feat.id!r}).")
                else:
                    filtered_features.append(feat)
            features = filtered_features

    return features

def _svg_arc_to_points(x1, y1, rx, ry, x_rot_deg, large_arc, sweep, x2, y2, n=32):
    """
    Convert an SVG elliptical arc to a polyline approximation.

    Implements the SVG arc parameterisation described in the SVG 1.1 spec
    (§F.6.5), converting from endpoint parameterisation to centre
    parameterisation, then sampling n points uniformly in angle.

    Returns a list of (x, y) tuples starting at (x1, y1) and ending at
    (x2, y2), including both endpoints.
    """
    if abs(x2 - x1) < 1e-9 and abs(y2 - y1) < 1e-9:
        return [(x1, y1)]
    if rx < 1e-9 or ry < 1e-9:
        return [(x1, y1), (x2, y2)]

    phi   = math.radians(x_rot_deg)
    cos_p = math.cos(phi)
    sin_p = math.sin(phi)

    # Step 1: rotate to ellipse-aligned frame
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p =  cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy

    # Step 2: ensure radii are large enough
    lam = (x1p / rx)**2 + (y1p / ry)**2
    if lam > 1.0:
        s  = math.sqrt(lam)
        rx *= s
        ry *= s

    # Step 3: compute centre (in rotated frame)
    num_sq = max(0.0,
                 (rx * ry)**2 - (rx * y1p)**2 - (ry * x1p)**2)
    den_sq = (rx * y1p)**2 + (ry * x1p)**2
    sq     = math.sqrt(num_sq / den_sq) if den_sq > 0 else 0.0
    if large_arc == sweep:
        sq = -sq
    cxp =  sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx

    # Step 4: transform centre back to original frame
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    cx = cos_p * cxp - sin_p * cyp + mx
    cy = sin_p * cxp + cos_p * cyp + my

    # Step 5: compute angles
    def angle(ux, uy, vx, vy):
        d = math.sqrt(ux*ux + uy*uy) * math.sqrt(vx*vx + vy*vy)
        if d < 1e-12:
            return 0.0
        c = max(-1.0, min(1.0, (ux*vx + uy*vy) / d))
        a = math.acos(c)
        if ux*vy - uy*vx < 0:
            a = -a
        return a

    theta1 = angle(1, 0, (x1p - cxp)/rx, (y1p - cyp)/ry)
    dtheta = angle((x1p - cxp)/rx, (y1p - cyp)/ry,
                   (-x1p - cxp)/rx, (-y1p - cyp)/ry)

    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    # Step 6: sample
    pts = []
    for k in range(n + 1):
        t   = theta1 + dtheta * k / n
        xr  = rx * math.cos(t)
        yr  = ry * math.sin(t)
        px  = cos_p * xr - sin_p * yr + cx
        py  = sin_p * xr + cos_p * yr + cy
        pts.append((px, py))

    return pts


def _parse_svg_path_d(d):
    """
    Parse an SVG path ``d`` attribute string into a list of subpaths, each
    being a (coords, is_closed) tuple.

    Each subpath starts with a M/m command and may end with Z/z (closed) or
    another M/m (open). The function returns one tuple per subpath encountered.

    All standard SVG 1.1 path commands are handled including proper elliptical
    arc sampling via :func:`_svg_arc_to_points`.

    Args:
        d (str): The raw value of an SVG ``<path>`` element's ``d`` attribute.

    Returns:
        list[tuple[list[tuple[float, float]], bool]]: A list of
        ``(coords, is_closed)`` tuples, one per subpath.
    """
    # The regex alternation captures either a single command letter or a
    # standalone float (including optional sign and scientific notation).
    tok = re.findall(
        r"[MmLlHhVvZzCcSsQqTtAa]"
        r"|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
        d,
    )

    subpaths   = []          # list of (coords, is_closed)
    cur_coords = []          # coords for the current subpath
    cur_closed = False       # whether current subpath has a Z
    cx = cy    = 0.0
    sx = sy    = 0.0
    last_ctrl  = None
    cmd        = None
    i          = 0

    def flush():
        nonlocal cur_coords, cur_closed
        if cur_coords:
            subpaths.append((cur_coords, cur_closed))
        cur_coords = []
        cur_closed = False

    def num():
        nonlocal i
        v = float(tok[i]); i += 1; return v

    def pair():
        nonlocal i
        x, y = float(tok[i]), float(tok[i+1]); i += 2; return x, y

    while i < len(tok):
        t = tok[i]
        if re.fullmatch(r"[MmLlHhVvZzCcSsQqTtAa]", t):
            cmd = t; i += 1; last_ctrl = None
            # Z/z has no parameters and must be executed immediately when the
            # command letter is consumed.  If we let the loop continue as usual
            # (waiting for a parameter token to trigger the elif-chain), Z at
            # end-of-path or immediately before another letter (e.g. "...Z M")
            # would never reach the elif handler and would be silently ignored —
            # leaving sp_closed=False and the closing coordinate unadded.
            if cmd in ("Z", "z"):
                cur_coords.append((sx, sy))
                cx, cy = sx, sy
                cur_closed = True
                flush()
                cmd = None
            continue

        if cmd in ("M", "m"):
            x, y = pair()
            # Starting a new subpath: flush the old one first
            if cur_coords:
                flush()
            cx, cy = (x, y) if cmd == "M" else (cx+x, cy+y)
            sx, sy = cx, cy
            cur_coords.append((cx, cy))
            cmd = "L" if cmd == "M" else "l"

        elif cmd in ("L", "l"):
            x, y = pair()
            cx, cy = (x, y) if cmd == "L" else (cx+x, cy+y)
            cur_coords.append((cx, cy))

        elif cmd in ("H", "h"):
            x = num()
            cx = x if cmd == "H" else cx+x
            cur_coords.append((cx, cy))

        elif cmd in ("V", "v"):
            y = num()
            cy = y if cmd == "V" else cy+y
            cur_coords.append((cx, cy))

        elif cmd in ("C", "c"):
            x1, y1 = pair(); x2, y2 = pair(); x, y = pair()
            if cmd == "c":
                x1,y1 = cx+x1,cy+y1; x2,y2 = cx+x2,cy+y2; x,y = cx+x,cy+y
            pts = _cubic_bezier((cx,cy),(x1,y1),(x2,y2),(x,y))
            cur_coords.extend(pts[1:])
            last_ctrl = (x2, y2); cx, cy = x, y

        elif cmd in ("S", "s"):
            x2, y2 = pair(); x, y = pair()
            if cmd == "s":
                x2,y2 = cx+x2,cy+y2; x,y = cx+x,cy+y
            x1,y1 = (2*cx-last_ctrl[0], 2*cy-last_ctrl[1]) if last_ctrl else (cx,cy)
            pts = _cubic_bezier((cx,cy),(x1,y1),(x2,y2),(x,y))
            cur_coords.extend(pts[1:])
            last_ctrl = (x2, y2); cx, cy = x, y

        elif cmd in ("Q", "q"):
            x1, y1 = pair(); x, y = pair()
            if cmd == "q":
                x1,y1 = cx+x1,cy+y1; x,y = cx+x,cy+y
            pts = _quadratic_bezier((cx,cy),(x1,y1),(x,y))
            cur_coords.extend(pts[1:])
            last_ctrl = (x1, y1); cx, cy = x, y

        elif cmd in ("T", "t"):
            x, y = pair()
            if cmd == "t": x, y = cx+x, cy+y
            x1,y1 = (2*cx-last_ctrl[0], 2*cy-last_ctrl[1]) if last_ctrl else (cx,cy)
            pts = _quadratic_bezier((cx,cy),(x1,y1),(x,y))
            cur_coords.extend(pts[1:])
            last_ctrl = (x1, y1); cx, cy = x, y

        elif cmd in ("A", "a"):
            rx_raw = abs(num()); ry_raw = abs(num()); x_rot = num()
            large = int(num()); sweep = int(num())
            x, y = pair()
            if cmd == "a": x, y = cx+x, cy+y
            arc_pts = _svg_arc_to_points(cx, cy, rx_raw, ry_raw, x_rot, large, sweep, x, y)
            cur_coords.extend(arc_pts[1:])
            cx, cy = x, y

        else:
            i += 1

    flush()  # flush final subpath
    return subpaths


def _compose_svg_matrix(m1, m2):
    """
    Multiply two 2-D affine transformation matrices and return the result.

    Both matrices are represented as 6-element tuples ``(a, b, c, d, e, f)``
    corresponding to the six independent values of a 3×3 homogeneous matrix
    stored in column-major order::

        | a  c  e |
        | b  d  f |
        | 0  0  1 |

    The multiplication order follows the SVG convention: ``m1 * m2`` means
    *m2 is applied first* (to the coordinates), then *m1 is applied to the
    result*. This matches the left-to-right reading order of a ``transform``
    attribute chain, where the rightmost function is applied closest to the
    geometry.

    The bottom row ``[0, 0, 1]`` is implicit and never stored or computed,
    because it is invariant under affine multiplication.

    Args:
        m1 (tuple[float, float, float, float, float, float]): The outer
            (left-hand) matrix, applied second. Typically the accumulated
            CTM from the parent element hierarchy.
        m2 (tuple[float, float, float, float, float, float]): The inner
            (right-hand) matrix, applied first. Typically the transform
            parsed from the current element's ``transform`` attribute.

    Returns:
        tuple[float, float, float, float, float, float]: The composed
        affine matrix ``m1 * m2`` in the same ``(a, b, c, d, e, f)``
        column-major format.
    """
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2

    # Full 3×3 affine multiplication, with the implicit bottom row [0,0,1]
    # omitted. Each output component is the dot product of one row of m1 with
    # one column of m2:
    #   new_a = a1*a2 + c1*b2 + e1*0   (third term always zero)
    #   new_e = a1*e2 + c1*f2 + e1*1   (translation column picks up e1)
    return (
        a1*a2 + c1*b2,
        b1*a2 + d1*b2,
        a1*c2 + c1*d2,
        b1*c2 + d1*d2,
        a1*e2 + c1*f2 + e1,
        b1*e2 + d1*f2 + f1,
    )

def _parse_svg_transform(t_str):
    """
    Parse an SVG ``transform`` attribute string into a single 2-D affine matrix.

    The SVG ``transform`` attribute can contain a whitespace-separated list of
    transform functions, each with a parenthesised argument list, e.g.
    ``"translate(10 20) rotate(45) scale(2)"``. This function parses every
    recognised function in order and composes them into one combined matrix,
    so that callers can apply a single multiplication rather than chaining
    individual operations.

    Composition follows the SVG specification: functions are applied
    right-to-left to coordinates, which corresponds to multiplying the
    accumulated result on the *right* as we scan left-to-right through the
    string.

    The following transform functions are supported:

    - ``matrix(a b c d e f)`` — explicit 2-D affine matrix.
    - ``translate(tx[, ty])`` — translation; ``ty`` defaults to ``0``.
    - ``scale(sx[, sy])`` — scaling; ``sy`` defaults to ``sx`` (uniform scale).
    - ``rotate(angle)`` — rotation about the origin, angle in degrees.
    - ``rotate(angle, cx, cy)`` — rotation about an arbitrary centre point,
      decomposed as translate(cx,cy) · rotate(angle) · translate(-cx,-cy).
    - ``skewX(angle)`` — horizontal shear by ``angle`` degrees.
    - ``skewY(angle)`` — vertical shear by ``angle`` degrees.

    Unrecognised function names are silently skipped so that vendor-prefixed
    or future SVG functions do not abort parsing of an otherwise valid string.

    Args:
        t_str (str): The raw value of an SVG ``transform`` attribute, e.g.
            ``"translate(30,10) rotate(-90)"``. May be empty or ``None``.

    Returns:
        tuple[float, float, float, float, float, float]: The composed affine
        matrix in ``(a, b, c, d, e, f)`` column-major format. Returns the
        identity matrix ``(1, 0, 0, 1, 0, 0)`` if ``t_str`` is empty,
        whitespace-only, or contains no recognised transform functions.
    """
    # An empty or whitespace-only string means "no transform"; return the
    # identity matrix so callers can always multiply without a special case.
    if not t_str or not t_str.strip():
        return _SVG_IDENTITY

    # Start with the identity so the first function's matrix is returned as-is
    # when only one function is present, and so the loop can unconditionally
    # right-compose each new matrix without a special first-iteration case.
    result = _SVG_IDENTITY

    # Match each 'functionName(args)' token in document order. The regex
    # intentionally does not match across parentheses so that malformed or
    # nested constructs don't corrupt the argument list of a later function.
    for m in re.finditer(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]+)\)", t_str):
        fn   = m.group(1)
        # Extract all numeric values from the argument string as a float list.
        # SVG allows comma-or-whitespace-separated arguments, so splitting on
        # a separator character is fragile; the float regex handles both.
        args = [float(v) for v in _SVG_FLOAT_RE.findall(m.group(2))]

        if fn == "matrix" and len(args) >= 6:
            # Direct 6-value affine matrix — unpack straight into the tuple.
            m_fn = tuple(args[:6])

        elif fn == "translate":
            tx = args[0] if args else 0.0
            # SVG spec: if ty is omitted it defaults to zero, not to tx.
            ty = args[1] if len(args) > 1 else 0.0
            # A pure translation has no rotation or scaling; a=d=1, b=c=0.
            m_fn = (1.0, 0.0, 0.0, 1.0, tx, ty)

        elif fn == "scale":
            sx = args[0] if args else 1.0
            # SVG spec: if sy is omitted the scale is uniform (sy = sx).
            sy = args[1] if len(args) > 1 else sx
            m_fn = (sx, 0.0, 0.0, sy, 0.0, 0.0)

        elif fn == "rotate":
            # SVG rotate angles are in degrees; math trig functions need radians.
            ang = math.radians(args[0]) if args else 0.0
            ca, sa = math.cos(ang), math.sin(ang)

            if len(args) >= 3:
                # rotate(angle, cx, cy) is equivalent to:
                #   translate(cx, cy) · rotate(angle) · translate(-cx, -cy)
                # This is built by composing three matrices so that the centre
                # point acts as a temporary origin for the rotation.
                cx, cy = args[1], args[2]
                m_fn = _compose_svg_matrix(
                    _compose_svg_matrix(
                        (1, 0, 0, 1, cx, cy),    # Step 1: shift origin to (cx,cy)
                        (ca, sa, -sa, ca, 0, 0), # Step 2: rotate about the new origin
                    ),
                    (1, 0, 0, 1, -cx, -cy),      # Step 3: shift origin back
                )
            else:
                # Simple rotation about the coordinate origin.
                m_fn = (ca, sa, -sa, ca, 0.0, 0.0)

        elif fn == "skewX":
            # Horizontal shear: x' = x + y*tan(angle), y' = y (unchanged).
            # In matrix form: c = tan(angle), all other off-diagonal terms zero.
            m_fn = (1.0, 0.0, math.tan(math.radians(args[0] if args else 0.0)), 1.0, 0.0, 0.0)

        elif fn == "skewY":
            # Vertical shear: x' = x (unchanged), y' = x*tan(angle) + y.
            # In matrix form: b = tan(angle), all other off-diagonal terms zero.
            m_fn = (1.0, math.tan(math.radians(args[0] if args else 0.0)), 0.0, 1.0, 0.0, 0.0)

        else:
            # Unrecognised function name — skip without aborting the loop so
            # subsequent valid functions in the same attribute are still parsed.
            continue

        # Right-compose the new function matrix into the running result.
        # Because SVG applies functions right-to-left to coordinates, scanning
        # the string left-to-right and right-composing produces the correct
        # combined transform.
        result = _compose_svg_matrix(result, m_fn)

    return result

def _svg_viewport_transform(svg_elem):
    """
    Derive the implicit coordinate transform for an ``<svg>`` element from
    its ``viewBox`` and ``width`` / ``height`` attributes.

    In SVG, the ``viewBox`` attribute defines an internal user-coordinate
    rectangle that is mapped ("fit") onto the element's viewport — the
    rectangular region defined by ``width`` and ``height``. When the two
    regions differ in size, all child coordinates must be scaled so that the
    viewBox fills the viewport exactly (assuming uniform, non-preserving
    aspect-ratio behaviour, which is the most common case).

    The resulting transform is a scale-and-translate matrix:
    - The scale factors are ``viewport_width / viewBox_width`` and
      ``viewport_height / viewBox_height``.
    - The translation offsets cancel out any non-zero ``viewBox`` origin
      ``(vb_x, vb_y)`` so that the top-left of the viewBox maps to the
      top-left of the viewport.

    Args:
        svg_elem (xml.etree.ElementTree.Element): The parsed ``<svg>`` root
            element. Must have a ``viewBox`` attribute for a non-identity
            matrix to be returned; ``width`` and ``height`` are optional and
            default to the viewBox dimensions (i.e. a 1:1 scale) if absent.

    Returns:
        tuple[float, float, float, float, float, float]: A 2-D affine matrix
        in ``(a, b, c, d, e, f)`` column-major format representing the
        viewport-to-viewBox mapping. Returns the identity matrix if:

        - the ``viewBox`` attribute is absent or empty,
        - the ``viewBox`` string contains fewer than four numeric values, or
        - the viewBox width or height is zero or negative.
    """
    # Try both capitalisation variants; some SVG generators emit 'viewbox'
    # in lower case even though the spec requires 'viewBox'.
    vb_str = svg_elem.get("viewBox") or svg_elem.get("viewbox", "")
    if not vb_str.strip():
        return _SVG_IDENTITY

    nums = [float(v) for v in _SVG_FLOAT_RE.findall(vb_str)]

    # A valid viewBox requires exactly four values: min-x, min-y, width, height.
    if len(nums) < 4:
        return _SVG_IDENTITY

    vb_x, vb_y, vb_w, vb_h = nums[:4]

    # A zero or negative viewBox dimension would produce a division-by-zero
    # or a sign-inverting scale; neither is meaningful, so return identity.
    if vb_w <= 0 or vb_h <= 0:
        return _SVG_IDENTITY

    def _len(attr, default):
        """Return the numeric value of a dimensional SVG attribute (e.g. '400px' → 400.0)."""
        s = svg_elem.get(attr)
        if not s:
            return default
        n = _SVG_FLOAT_RE.findall(s)
        return float(n[0]) if n else default

    # Compute the scale factors needed to map viewBox units onto viewport pixels.
    # If width/height are absent we fall back to the viewBox dimensions, giving
    # a 1:1 (identity) scale while still applying any viewBox origin offset.
    sx = _len("width",  vb_w) / vb_w
    sy = _len("height", vb_h) / vb_h

    # The translation terms cancel out the viewBox origin (vb_x, vb_y) so that
    # a non-zero viewBox minimum maps to coordinate (0, 0) in output space.
    return (sx, 0.0, 0.0, sy, -vb_x * sx, -vb_y * sy)

def _cubic_bezier(p0, p1, p2, p3, n=20):
    """
    Sample a cubic Bézier curve at ``n + 1`` evenly-spaced parameter values.

    A cubic Bézier is defined by four control points ``p0``–``p3``. The curve
    starts at ``p0`` (t=0), ends at ``p3`` (t=1), and is pulled towards the
    intermediate control points ``p1`` and ``p2`` without necessarily passing
    through them. The parametric equation is the standard degree-3 Bernstein
    polynomial::

        B(t) = (1-t)³·p0 + 3(1-t)²t·p1 + 3(1-t)t²·p2 + t³·p3

    The default of ``n=20`` produces 21 points, giving a smooth visual
    approximation for typical SVG path curves while keeping the polygon
    vertex count reasonable.

    Args:
        p0 (tuple[float, float]): Start point of the curve (t=0).
        p1 (tuple[float, float]): First control point.
        p2 (tuple[float, float]): Second control point.
        p3 (tuple[float, float]): End point of the curve (t=1).
        n  (int): Number of segments to sample. The returned list will
            contain ``n + 1`` points. Must be >= 1. Defaults to ``20``.

    Returns:
        list[tuple[float, float]]: An ordered list of ``n + 1`` ``(x, y)``
        coordinate pairs, starting at ``p0`` and ending at ``p3``, tracing
        the curve at uniformly-spaced parameter values
        ``t = 0, 1/n, 2/n, …, 1``.
    """
    pts = []
    for i in range(n + 1):
        t  = i / n       # Parameter value in [0, 1]; t=0 gives p0, t=1 gives p3.
        mt = 1.0 - t     # Complement of t, pre-computed to avoid repeated subtraction.

        x  = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y  = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts

def _quadratic_bezier(p0, p1, p2, n=12):
    """
    Sample a quadratic Bézier curve at ``n + 1`` evenly-spaced parameter values.

    A quadratic Bézier is defined by three control points ``p0``–``p2``. The
    curve starts at ``p0`` (t=0), ends at ``p2`` (t=1), and is drawn towards
    the single intermediate control point ``p1``. The parametric equation is
    the degree-2 Bernstein polynomial::

        B(t) = (1-t)²·p0 + 2(1-t)t·p1 + t²·p2

    The default of ``n=12`` produces 13 points. Quadratic curves have less
    curvature freedom than cubics, so a lower sample count still yields an
    accurate approximation.

    Args:
        p0 (tuple[float, float]): Start point of the curve (t=0).
        p1 (tuple[float, float]): Control point.
        p2 (tuple[float, float]): End point of the curve (t=1).
        n  (int): Number of segments to sample. The returned list will
            contain ``n + 1`` points. Must be >= 1. Defaults to ``12``.

    Returns:
        list[tuple[float, float]]: An ordered list of ``n + 1`` ``(x, y)``
        coordinate pairs, starting at ``p0`` and ending at ``p2``, tracing
        the curve at uniformly-spaced parameter values
        ``t = 0, 1/n, 2/n, …, 1``.
    """
    pts = []
    for i in range(n + 1):
        t  = i / n       # Parameter value in [0, 1].
        mt = 1.0 - t     # Complement, pre-computed for clarity and slight efficiency.

        x  = mt**2*p0[0] + 2*mt*t*p1[0] + t**2*p2[0]
        y  = mt**2*p0[1] + 2*mt*t*p1[1] + t**2*p2[1]
        pts.append((x, y))
    return pts

def parse_pdf(filepath):
    """
    Extract polygon shapes from every page of a PDF file.

    Each page is rendered to an SVG string by PyMuPDF, written to a temporary
    file on disk, and then parsed by :func:`parse_svg`. This round-trip
    approach reuses the full SVG parser (including transform handling and
    Bézier curve sampling) without duplicating that logic for the PDF case.

    Page-spanning background rectangles — the "artboard border" that PDF
    renderers emit as a filled rect covering the entire page — are detected by
    :func:`_is_artboard_rect` and silently discarded, since they represent the
    canvas boundary rather than a real shape.

    Feature IDs are prefixed with the page index (``p0_``, ``p1_``, …) so
    that features from different pages remain uniquely identifiable in the
    combined output list even if the underlying SVG elements share IDs.

    Args:
        filepath (str | os.PathLike): Path to the PDF file to parse.

    Returns:
        list[Feature]: Feature dataclass instances from all pages,
        concatenated in page order. Each instance contains:

        - ``id`` (*str*): Page-prefixed feature ID, e.g. ``"p0_rect1"``.
        - ``polygon`` (*shapely.geometry.Polygon*): The extracted geometry
          in the page's SVG coordinate space.
        - ``flip_y`` (*bool*): Inherited from :func:`parse_svg`; always
          ``False`` for SVG-derived features.

        Pages that fail to parse are skipped with a printed warning rather
        than raising an exception, so a single corrupt page does not abort
        processing of the rest of the document.

    Raises:
        ImportError: If PyMuPDF (``fitz``) is not installed.
        fitz.FileNotFoundError: If ``filepath`` does not exist or is not a
            valid PDF.
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required for PDF parsing:  pip install PyMuPDF")

    doc      = fitz.open(filepath)
    features = []

    for page_num, page in enumerate(doc):
        # fitz.Rect stores the page bounding box in PDF points (1/72 inch).
        # This is passed to _is_artboard_rect so it can compare polygon bounds
        # against the full page dimensions.
        pr      = page.rect
        # Render the page's vector content to an SVG string using PyMuPDF's
        # built-in SVG exporter. fitz.Identity means no additional scaling is
        # applied; coordinates are in the page's native PDF point units.
        svg_str = page.get_svg_image(matrix=fitz.Identity)

        # parse_svg requires a file path, not a string, so the SVG content is
        # written to a named temporary file. delete=False is necessary on
        # Windows, where a file cannot be opened by another process while it
        # is still held open by the NamedTemporaryFile context manager.
        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w",
                                         encoding="utf-8", delete=False) as tf:
            tf.write(svg_str)
            tmp_path = tf.name

        try:
            page_features = parse_svg(tmp_path)
            kept = 0

            for feat in page_features:
                # Discard the full-page background rectangle that PyMuPDF
                # typically emits as the first shape on every page. Keeping it
                # would produce a polygon that covers the entire canvas and
                # interferes with any subsequent spatial analysis.
                if _is_artboard_rect(feat.polygon, pr.x0, pr.y0, pr.x1, pr.y1):
                    print(f"  [pdf] Skipping page-border rectangle on page {page_num}.")
                    continue

                # Prepend the page index to the feature ID so that IDs remain
                # unique across all pages even when the SVG exporter reuses
                # element IDs (which it commonly does).
                feat.id = f"p{page_num}_{feat.id}"
                features.append(feat)
                kept += 1

        except Exception as exc:
            # A parse failure on one page (e.g. malformed SVG output from an
            # unusual PDF) should not prevent the remaining pages from being
            # processed.
            print(f"  Warning: could not parse PDF page {page_num}: {exc}")

        finally:
            # Always remove the temporary file, even if parsing raised an
            # exception. OSError is suppressed in case the file was already
            # cleaned up or the OS refuses deletion (e.g. antivirus lock).
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    doc.close()
    return features

def _is_artboard_rect(geom, page_x0, page_y0, page_x1, page_y1):
    """
    Determine whether a geometry is a page-spanning background shape that
    should be discarded as an artboard or canvas rectangle.

    SVG and PDF editors (Illustrator, Inkscape, Affinity Designer, PyMuPDF,
    etc.) commonly emit a shape covering the entire page as the first or last
    drawn element — the document background or artboard border.  This shape
    is not a real feature and must be filtered out before the extracted
    polygons are used for spatial analysis.

    The test handles every geometry type that the parsers can produce:

    * **Polygon / MultiPolygon** — solid filled background: uses the actual
      fill area.
    * **LineString / MultiLineString** — unfilled artboard border or white-
      filled rect (converted to a closed LineString): uses the bounding-box
      area as a proxy, since a closed outline has zero Shapely fill area.

    Two conditions must **both** hold for the geometry to be classified as an
    artboard rectangle (both thresholds are configurable in the settings):

    1. **Rectangular**: ``effective_area / bbox_area >= ARTBOARD_MIN_RECT``.
       The shape must fill at least ``ARTBOARD_MIN_RECT`` of its own
       axis-aligned bounding box.  A perfect rectangle scores 1.0; shapes
       such as triangles, circles, or irregular outlines score much lower.

    2. **Large enough**: ``effective_area / page_area >= ARTBOARD_MIN_SIZE``.
       The shape must cover at least ``ARTBOARD_MIN_SIZE`` of the total
       page area.

    Args:
        geom: Any Shapely geometry (Polygon, MultiPolygon, LineString, …).
        page_x0 (float): Left edge of the page in world coordinates.
        page_y0 (float): Top edge of the page in world coordinates.
        page_x1 (float): Right edge of the page in world coordinates.
        page_y1 (float): Bottom edge of the page in world coordinates.

    Returns:
        bool: ``True`` if the geometry is judged to be a background artboard
        rectangle and should be discarded; ``False`` if it should be kept.
    """
    page_area = (page_x1 - page_x0) * (page_y1 - page_y0)
    if page_area <= 0:
        return False

    try:
        minx, miny, maxx, maxy = geom.bounds
    except Exception:
        return False

    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area <= 0:
        return False

    # For filled polygons use the true fill area; for outlines (LineString /
    # closed unfilled border) the Shapely area is 0, so fall back to the
    # bounding-box area — a closed rectangular outline perfectly tiles its
    # bbox and should score 1.0 on the rectangularity test.
    fill_area = geom.area
    if fill_area <= 0:
        fill_area = bbox_area   # outline proxy

    rect_fill     = fill_area / bbox_area   # 1.0 = perfect rectangle
    page_fraction = fill_area / page_area   # fraction of page covered

    return rect_fill >= config.ARTBOARD_MIN_RECT and page_fraction >= config.ARTBOARD_MIN_SIZE

def parse_image(filepath):
    """
    Extract polygon contours from a raster image file (JPEG, PNG, TIFF, etc.).

    The image is converted to greyscale and binarised using a fixed global
    threshold (``IMAGE_THRESHOLD``). Connected dark regions in the binary image
    are each labelled as a separate region by ``skimage.measure.label``, and the
    outer contour of each sufficiently large region is traced and converted to a
    Shapely ``Polygon``.

    The function assumes that features of interest are dark on a light
    background. If the binary image is majority-dark (more than 50% of pixels
    fall below the threshold), the polarity is inverted automatically so that
    light-on-dark images are handled correctly without manual intervention.

    Args:
        filepath (str | os.PathLike): Path to the raster image file. Any
            format supported by Pillow's ``Image.open`` is accepted.

    Returns:
        list[Feature]: Feature dataclass instances, one per detected region.
        Regions with fewer than 10 pixels or with no traceable contour are
        discarded. Each instance contains:

        - ``id`` (*str*): String representation of the region's integer
          label (``"1"``, ``"2"``, …), assigned by ``skimage.measure.label``
          in raster-scan order.
        - ``polygon`` (*shapely.geometry.Polygon*): The outer boundary of
          the region in image pixel coordinates, with x = column index and
          y = row index.
        - ``flip_y`` (*bool*): Always ``False``; image coordinates already
          use a top-left origin with Y increasing downward, matching
          screen-space conventions.

    Raises:
        ImportError: If Pillow is not installed.
        ImportError: If scikit-image is not installed.
        PIL.UnidentifiedImageError: If ``filepath`` is not a recognised image
            format.
    """
    if not HAS_PIL:
        raise ImportError("Pillow is required for image parsing:  pip install pillow")
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for image parsing:  pip install scikit-image")

    # Convert to greyscale ('L' mode = 8-bit luminance) so that colour images,
    # RGBA PNGs, and single-channel bitmaps all produce a consistent 2-D array.
    img = Image.open(filepath).convert("L")
    arr = np.array(img)   # Shape: (rows, cols), dtype uint8, values 0–255.

    # Binarise: pixels strictly below IMAGE_THRESHOLD are considered "dark"
    # (foreground features); all others are "light" (background).
    binary = arr < config.IMAGE_THRESHOLD

    # Polarity check: if more than half the pixels are dark, the image is
    # likely light-on-dark (e.g. white lines on a black background). Inverting
    # restores the dark-on-light assumption without requiring the caller to
    # pre-process the image.
    if binary.sum() > binary.size * 0.5:
        binary = ~binary

    # Label each connected component of dark pixels with a unique integer ID.
    # Connectivity defaults to the full 8-neighbourhood (diagonal adjacency),
    # which prevents thin diagonal lines from fragmenting into many small regions.
    labeled_arr = label(binary)
    features    = []

    for region_id in range(1, int(labeled_arr.max()) + 1):
        mask = labeled_arr == region_id

        # Discard specks and noise: regions smaller than 10 pixels cannot form
        # a meaningful polygon and are almost certainly compression artefacts.
        if mask.sum() < 10:
            continue

        # find_contours traces iso-contours at the given level (0.5 sits exactly
        # between the 0 = background and 1 = foreground pixel values). Each
        # contour is an (N, 2) array of (row, col) coordinates.
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            continue

        # A region may have multiple contours if it contains holes; the longest
        # one traces the outer boundary, which is the only contour we need for
        # a simple filled polygon.
        outer  = max(contours, key=len)

        # find_contours returns (row, col) pairs; convert to (x, y) = (col, row)
        # to match the conventional right-hand spatial coordinate system expected
        # by Shapely and the rest of the pipeline.
        coords = [(float(c[1]), float(c[0])) for c in outer]

        if len(coords) < 3:
            continue

        # Crop the mask to its bounding box to avoid storing full-image arrays
        rows, cols = np.where(mask)
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        mask_crop = mask[r_min : r_max + 1, c_min : c_max + 1]

        try:
            poly = Polygon(coords)
            # Repair self-intersecting contours with the buffer(0) idiom, as in
            # parse_svg. Contour-tracing can occasionally produce a figure-eight
            # ring for regions with a one-pixel-wide neck.
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                # _flip_y=False: image (row, col) → (x, y) is already in a
                # downward-Y system consistent with screen/canvas rendering.
                features.append(Feature(
                    id=str(region_id),
                    polygon=poly,
                    raw_mask=mask_crop,
                    raw_mask_origin=(c_min, r_min),
                ))
        except Exception:
            # Silently discard any polygon Shapely cannot construct (e.g. a
            # contour that reduced to fewer than three unique points after
            # coordinate conversion).
            pass

    return features


def _dispatch_parser(input_path):
    """
    Select and invoke the appropriate file parser based on the input file's
    extension, returning the parsed feature list.

    The dispatch table maps every supported extension to its parser function.
    Multiple extensions can map to the same parser (e.g. ``.jpg``, ``.jpeg``,
    ``.tif``, and ``.tiff`` all route to :func:`parse_image`), so adding
    support for a new format requires only a single new entry in the table
    rather than a change to the branching logic.

    Args:
        input_path (pathlib.Path): Path to the input file. The extension is
            extracted and lower-cased before lookup so that files with
            upper-case extensions (e.g. ``.SHP``, ``.SVG``) are handled
            correctly on case-sensitive file systems.

    Returns:
        list[Feature]: Feature dataclass instances as returned by the
            selected parser. See the individual parser functions for details.

    Raises:
        ValueError: If the file extension is not present in the dispatch
            table, with a message listing all supported extensions.
    """
    ext = input_path.suffix.lower()

    # A plain dict used as a dispatch table is preferred over a chain of
    # if/elif branches because it makes the full set of supported formats
    # immediately visible and trivially extensible — adding a new format
    # requires inserting one key-value pair, nothing else.
    dispatch = {
        ".shp":  parse_shapefile,
        ".svg":  parse_svg,
        ".pdf":  parse_pdf,
        ".jpg":  parse_image,
        ".jpeg": parse_image,
        ".png":  parse_image,
        ".tif":  parse_image,
        ".tiff": parse_image,
    }

    if ext not in dispatch:
        # Include the full list of supported extensions in the error message
        # so the user can self-diagnose without reading the source code.
        raise ValueError(
            f"Unsupported file type: '{ext}'\n"
            f"Supported: {', '.join(dispatch)}"
        )

    # Pass the path as a string because all parser functions accept str
    # (rather than Path) for compatibility with libraries that pre-date
    # pathlib (e.g. Fiona, PyMuPDF, Pillow).
    return dispatch[ext](str(input_path))
