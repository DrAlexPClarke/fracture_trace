"""
data_models.py — Core dataclasses for the skeletonisation pipeline.

Defines Branch (one skeleton branch within a feature) and Feature (a single
polygon feature at any stage of the analysis pipeline).
"""

from dataclasses import dataclass, field


@dataclass
class Branch:
    """One skeleton branch within a feature."""
    id:            int
    centerline:    list                                  # list[tuple[float,float]] — Gaussian-smoothed world coords
    sample_points: list = field(default_factory=list)   # SAMPLING_INTERVAL-spaced points → width measurement
    output_coords: list = field(default_factory=list)   # decimated (uniform + RDP) coords written to skeleton.svg
    profile:       list = field(default_factory=list)   # populated by _find_partial_thickness()


@dataclass
class Feature:
    """A single polygon feature at any stage of the analysis pipeline."""
    id:         str
    polygon:    object   # shapely.geometry.Polygon
    flip_y:     bool   = False
    branches:   list   = field(default_factory=list)   # list[Branch], populated by _dispatch_skeletoniser()
    stats:      object = field(default=None, repr=False)  # dict, populated by _calculate_statistics()
    fft_data:   object = field(default=None, repr=False)  # dict, populated by _calculate_statistics()
    skeleton_overlay_data: object = field(default=None, repr=False)  # dict, populated by _dispatch_skeletoniser()
    raw_mask:        object = field(default=None, repr=False)   # numpy bool array (cropped), raster features only
    raw_mask_origin: tuple  = field(default=(0, 0))             # (col_min, row_min) pixel offset of crop
