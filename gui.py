"""
gui.py — PyQt6 graphical front-end for the Fracture Width Analyser pipeline.

Launch directly:
    python gui.py

Tabs (left → right):
    1. Basic Settings
    2. Input Parsing
    3. Skeletoniser
    4. Post-processing
    5. Export
    6. Advanced
"""

import sys
import traceback

import config
import main as main_module

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Stdout redirect
# ---------------------------------------------------------------------------

class _LogEmitter(QObject):
    """QObject that carries the text_written signal.

    Kept separate from _LogStream so the signal can be emitted from any
    thread and Qt's queued-connection mechanism delivers it safely to the
    main thread for widget updates.
    """
    text_written = pyqtSignal(str)


class _LogStream:
    """File-like object that emits text via a Qt signal instead of touching
    the QTextEdit directly.  Safe to use from any thread."""

    def __init__(self, emitter: _LogEmitter):
        self._emitter = emitter

    def write(self, text: str):
        if text:
            self._emitter.text_written.emit(text)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _Worker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            main_module.main()
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skeletoniser & Width Analyser")
        self.setMinimumSize(600, 700)

        self._worker: _Worker | None = None
        self._real_stdout = sys.stdout

        # Emitter for thread-safe log output: worker thread emits the signal,
        # Qt delivers it to _append_log in the main thread via queued connection.
        self._log_emitter = _LogEmitter()
        self._log_stream   = _LogStream(self._log_emitter)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Tab bar
        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs, stretch=1)

        # Tabs in display order
        self._build_basic_tab()
        self._build_input_parsing_tab()
        self._build_skeletoniser_tab()
        self._build_postprocessing_tab()
        self._build_export_tab()
        self._build_advanced_tab()

        # Log panel
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(150)
        mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._log.setFont(mono)
        root_layout.addWidget(self._log)

        # Connect log emitter → append slot (queued: always runs in main thread)
        self._log_emitter.text_written.connect(self._append_log)

        # Bottom bar: Run / Stop / Status
        bottom = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setDefault(True)
        self._btn_run.clicked.connect(self._on_run)

        self._btn_stop = QPushButton("●  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        self._status_label = QLabel("Status: Idle")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        bottom.addWidget(self._btn_run)
        bottom.addWidget(self._btn_stop)
        bottom.addStretch()
        bottom.addWidget(self._status_label)
        root_layout.addLayout(bottom)

        # Status bar (bottom of window)
        self.setStatusBar(QStatusBar())

        # Populate from config
        self._load_from_config()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scroll_wrap(widget: QWidget) -> QScrollArea:
        """Wrap *widget* in a QScrollArea."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    @staticmethod
    def _form_container() -> tuple[QWidget, QFormLayout]:
        """Return a container widget + its QFormLayout."""
        container = QWidget()
        layout = QFormLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        return container, layout

    @staticmethod
    def _labelled(text: str, tooltip: str = "") -> QLabel:
        lbl = QLabel(text)
        if tooltip:
            lbl.setToolTip(tooltip)
        return lbl

    @staticmethod
    def _note(text: str) -> QLabel:
        """Italic note label spanning the full form width."""
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #555; font-style: italic; margin-bottom: 4px;")
        return lbl

    @staticmethod
    def _section(text: str) -> QLabel:
        """Bold section separator within a tab."""
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; margin-top: 8px;")
        return lbl

    @staticmethod
    def _browse_line() -> tuple[QLineEdit, QPushButton, QWidget]:
        """Return (line_edit, button, row_widget) for a file/dir picker."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        le = QLineEdit()
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        h.addWidget(le)
        h.addWidget(btn)
        return le, btn, row

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_basic_tab(self):
        container, form = self._form_container()

        # INPUT_FILE
        tt = ".shp | .svg | .pdf | .jpg | .png | .tif | .tiff — path to the input dataset"
        self._le_input_file, btn_input, row_input = self._browse_line()
        self._le_input_file.setToolTip(tt)
        btn_input.clicked.connect(self._browse_input_file)
        form.addRow(self._labelled("Input file", tt), row_input)

        # OUTPUT_DIRECTORY
        tt = "All output files are written to this directory"
        self._le_output_dir, btn_out, row_out = self._browse_line()
        self._le_output_dir.setToolTip(tt)
        btn_out.clicked.connect(self._browse_output_dir)
        form.addRow(self._labelled("Output directory", tt), row_out)

        form.addRow(self._section("Feature handling"))

        # SEPARATE_MULTIPOLYGONS
        tt = (
            "If checked, MultiPolygon features are split into one feature per component "
            "(classic behaviour). If unchecked, all components are kept together and a single "
            "connected skeleton is traced across them."
        )
        self._chk_separate = QCheckBox()
        self._chk_separate.setToolTip(tt)
        form.addRow(self._labelled("Separate multipolygons", tt), self._chk_separate)

        # PRESERVE_TOPOLOGY
        tt = (
            "Extend branch endpoints that abut another feature's polygon so they snap "
            "onto that feature's skeleton, preserving connectivity across the dataset."
        )
        self._chk_topology = QCheckBox()
        self._chk_topology.setToolTip(tt)
        form.addRow(self._labelled("Preserve topology", tt), self._chk_topology)

        form.addRow(self._section("Raster input"))

        # IMAGE_THRESHOLD
        tt = "Greyscale threshold for raster image parsing (0–255)"
        self._sb_image_threshold = QSpinBox()
        self._sb_image_threshold.setRange(0, 255)
        self._sb_image_threshold.setToolTip(tt)
        form.addRow(self._labelled("Image threshold", tt), self._sb_image_threshold)

        self._tabs.addTab(self._scroll_wrap(container), "Basic")

    def _build_input_parsing_tab(self):
        container, form = self._form_container()

        form.addRow(self._note(
            "These settings control detection and removal of artboard/canvas rectangles "
            "from SVG and PDF files. A rectangle covering most of the canvas is assumed "
            "to be a background frame and is excluded from analysis."
        ))

        # ARTBOARD_MIN_SIZE
        tt = (
            "Artboard detection: polygon must cover at least this fraction of the page/canvas area "
            "to be considered an artboard rectangle. Default 0.50 (50%). "
            "Decrease to catch smaller background frames; set to 1.0 to disable size-based filtering."
        )
        self._sb_artboard_size = QDoubleSpinBox()
        self._sb_artboard_size.setRange(0.0, 1.0)
        self._sb_artboard_size.setSingleStep(0.01)
        self._sb_artboard_size.setDecimals(2)
        self._sb_artboard_size.setToolTip(tt)
        form.addRow(self._labelled("Min canvas coverage", tt), self._sb_artboard_size)

        # ARTBOARD_MIN_RECT
        tt = (
            "Artboard detection: polygon must fill at least this fraction of its own bounding box "
            "to be considered rectangular. Default 0.95. "
            "Decrease to also catch non-axis-aligned frames or rectangles with heavily rounded corners."
        )
        self._sb_artboard_rect = QDoubleSpinBox()
        self._sb_artboard_rect.setRange(0.0, 1.0)
        self._sb_artboard_rect.setSingleStep(0.01)
        self._sb_artboard_rect.setDecimals(2)
        self._sb_artboard_rect.setToolTip(tt)
        form.addRow(self._labelled("Min rectangularity", tt), self._sb_artboard_rect)

        self._tabs.addTab(self._scroll_wrap(container), "Input Parsing")

    def _build_skeletoniser_tab(self):
        container, form = self._form_container()

        # MINIMUM_FEATURE_SIZE
        tt = (
            "% of extent — features smaller than this are dropped before processing "
            "(0 = keep all). For polygons: bounding-box side length must exceed this. "
            "For line features: arc length must exceed this."
        )
        self._sb_min_feature = QDoubleSpinBox()
        self._sb_min_feature.setRange(0.0, 100.0)
        self._sb_min_feature.setSingleStep(0.001)
        self._sb_min_feature.setDecimals(4)
        self._sb_min_feature.setSuffix(" %")
        self._sb_min_feature.setToolTip(tt)
        form.addRow(self._labelled("Minimum feature size", tt), self._sb_min_feature)

        form.addRow(self._section("Method"))

        # SKELETONISATION_METHOD
        tt = (
            "auto: full automatic decision tree (recommended).\n"
            "directional: cross-sections only, no Lee fallback.\n"
            "directional_and_single_branch: directional with Lee single-branch fallback.\n"
            "single_or_multi_branch: Lee thinning, decides via branching threshold.\n"
            "single_branch: Lee thinning + graph diameter, single path.\n"
            "multi_branch: full medial-axis skeleton with stub pruning."
        )
        self._cb_skel_method = QComboBox()
        self._cb_skel_method.addItems([
            "auto",
            "directional",
            "directional_and_single_branch",
            "single_or_multi_branch",
            "single_branch",
            "multi_branch",
        ])
        self._cb_skel_method.setToolTip(tt)
        form.addRow(self._labelled("Skeletonisation method", tt), self._cb_skel_method)

        form.addRow(self._section("Rasterisation"))

        # RASTER_RESOLUTION
        tt = (
            "% of extent — world units per pixel when rasterising (~20 px across the image). "
            "Auto-scaling via pixel budget limits refines this. Ignored for raster images."
        )
        self._sb_raster_res = QDoubleSpinBox()
        self._sb_raster_res.setRange(0.0, 100.0)
        self._sb_raster_res.setSingleStep(0.0001)
        self._sb_raster_res.setDecimals(4)
        self._sb_raster_res.setSuffix(" %")
        self._sb_raster_res.setToolTip(tt)
        form.addRow(self._labelled("Raster resolution", tt), self._sb_raster_res)

        # RASTER_BUFFER
        tt = (
            "% of extent — outward buffer applied to every polygon before rasterisation. "
            "Ensures thin features survive morphological thinning. "
            "Applied to the local rasterisation polygon only; feature.polygon is never modified."
        )
        self._sb_raster_buf = QDoubleSpinBox()
        self._sb_raster_buf.setRange(0.0, 100.0)
        self._sb_raster_buf.setSingleStep(0.0001)
        self._sb_raster_buf.setDecimals(4)
        self._sb_raster_buf.setSuffix(" %")
        self._sb_raster_buf.setToolTip(tt)
        form.addRow(self._labelled("Raster buffer", tt), self._sb_raster_buf)

        # MAX_RASTER_PIXELS
        tt = "Pixel budget per polygon; resolution is auto-coarsened if this would be exceeded"
        self._sb_max_raster_px = QSpinBox()
        self._sb_max_raster_px.setRange(0, 500_000_000)
        self._sb_max_raster_px.setSingleStep(1_000_000)
        self._sb_max_raster_px.setToolTip(tt)
        form.addRow(self._labelled("Max raster pixels", tt), self._sb_max_raster_px)

        # MIN_RASTER_PIXELS
        tt = "Minimum pixel count per polygon; resolution is refined upward if below this (0 = off)"
        self._sb_min_raster_px = QSpinBox()
        self._sb_min_raster_px.setRange(0, 500_000_000)
        self._sb_min_raster_px.setSingleStep(100_000)
        self._sb_min_raster_px.setToolTip(tt)
        form.addRow(self._labelled("Min raster pixels", tt), self._sb_min_raster_px)

        form.addRow(self._section("Branch pruning"))

        # MIN_BRANCH_PIXELS
        tt = "A branch is treated as a stub and pruned if it has fewer than this many pixels"
        self._sb_min_branch_px = QSpinBox()
        self._sb_min_branch_px.setRange(0, 1000)
        self._sb_min_branch_px.setToolTip(tt)
        form.addRow(self._labelled("Min branch pixels", tt), self._sb_min_branch_px)

        # MIN_BRANCH_PERCENT
        tt = "A branch is also pruned if it accounts for fewer than this % of total skeleton pixels (0 = off)"
        self._sb_min_branch_pct = QDoubleSpinBox()
        self._sb_min_branch_pct.setRange(0.0, 100.0)
        self._sb_min_branch_pct.setSingleStep(0.1)
        self._sb_min_branch_pct.setDecimals(1)
        self._sb_min_branch_pct.setSuffix(" %")
        self._sb_min_branch_pct.setToolTip(tt)
        form.addRow(self._labelled("Min branch percent", tt), self._sb_min_branch_pct)

        form.addRow(self._section("Method selection thresholds"))
        form.addRow(self._note(
            "The following thresholds control which skeletonisation method is automatically "
            "chosen in 'auto' and directional modes. Most users will not need to change these."
        ))

        # CURVATURE_THRESHOLD
        tt = (
            "Sinuosity (arc / chord) above which a feature is treated as too curved for the "
            "directional method and Lee is used instead. Set to 0 to always attempt directional."
        )
        self._sb_curvature = QDoubleSpinBox()
        self._sb_curvature.setRange(0.0, 20.0)
        self._sb_curvature.setSingleStep(0.05)
        self._sb_curvature.setDecimals(2)
        self._sb_curvature.setToolTip(tt)
        form.addRow(self._labelled("Curvature threshold", tt), self._sb_curvature)

        # SOLIDITY_THRESHOLD
        tt = (
            "Polygon area / convex-hull area. Below this ratio the feature is considered "
            "too non-convex for directional skeletonisation."
        )
        self._sb_solidity = QDoubleSpinBox()
        self._sb_solidity.setRange(0.0, 1.0)
        self._sb_solidity.setSingleStep(0.01)
        self._sb_solidity.setDecimals(2)
        self._sb_solidity.setToolTip(tt)
        form.addRow(self._labelled("Solidity threshold", tt), self._sb_solidity)

        # ASPECT_RATIO_THRESHOLD
        tt = (
            "Long / short axis ratio of the minimum bounding rectangle. Below this the feature "
            "is considered too compact for directional skeletonisation."
        )
        self._sb_aspect = QDoubleSpinBox()
        self._sb_aspect.setRange(0.0, 50.0)
        self._sb_aspect.setSingleStep(0.1)
        self._sb_aspect.setDecimals(1)
        self._sb_aspect.setToolTip(tt)
        form.addRow(self._labelled("Aspect ratio threshold", tt), self._sb_aspect)

        # ESCAPE_THRESHOLD
        tt = (
            "Maximum fraction of interior directional-skeleton vertices that may fall outside "
            "the polygon before the method falls back to Lee."
        )
        self._sb_escape = QDoubleSpinBox()
        self._sb_escape.setRange(0.0, 1.0)
        self._sb_escape.setSingleStep(0.01)
        self._sb_escape.setDecimals(2)
        self._sb_escape.setToolTip(tt)
        form.addRow(self._labelled("Escape threshold", tt), self._sb_escape)

        # BRANCHING_THRESHOLD
        tt = (
            "Fraction of post-pruning skeleton pixels that lie off the main (diameter) branch. "
            "Above this → multi_branch skeleton is used."
        )
        self._sb_branching = QDoubleSpinBox()
        self._sb_branching.setRange(0.0, 1.0)
        self._sb_branching.setSingleStep(0.01)
        self._sb_branching.setDecimals(2)
        self._sb_branching.setToolTip(tt)
        form.addRow(self._labelled("Branching threshold", tt), self._sb_branching)

        self._tabs.addTab(self._scroll_wrap(container), "Skeletoniser")

    def _build_postprocessing_tab(self):
        container, form = self._form_container()

        form.addRow(self._section("Width measurement"))

        # SAMPLING_INTERVAL
        tt = "% of extent — spacing between width measurement points along the centreline"
        self._sb_sampling = QDoubleSpinBox()
        self._sb_sampling.setRange(0.0, 100.0)
        self._sb_sampling.setSingleStep(0.001)
        self._sb_sampling.setDecimals(4)
        self._sb_sampling.setSuffix(" %")
        self._sb_sampling.setToolTip(tt)
        form.addRow(self._labelled("Sampling interval", tt), self._sb_sampling)

        # MAX_WIDTH_RAY_DISTANCE
        tt = (
            "% of extent — maximum length of the perpendicular rays fired from the centreline "
            "to locate fracture walls. Caps spuriously large widths in open or branching features. "
            "Set to 0 for automatic (2 × polygon bounding-box diagonal)."
        )
        self._sb_max_ray = QDoubleSpinBox()
        self._sb_max_ray.setRange(0.0, 100.0)
        self._sb_max_ray.setSingleStep(0.001)
        self._sb_max_ray.setDecimals(4)
        self._sb_max_ray.setSuffix(" %")
        self._sb_max_ray.setToolTip(tt)
        form.addRow(self._labelled("Max width ray distance", tt), self._sb_max_ray)

        form.addRow(self._section("Centreline quality"))

        # SMOOTHING
        tt = "% of extent — Gaussian sigma for centreline smoothing (0 = none)"
        self._sb_smoothing = QDoubleSpinBox()
        self._sb_smoothing.setRange(0.0, 100.0)
        self._sb_smoothing.setSingleStep(0.001)
        self._sb_smoothing.setDecimals(4)
        self._sb_smoothing.setSuffix(" %")
        self._sb_smoothing.setToolTip(tt)
        form.addRow(self._labelled("Smoothing (σ)", tt), self._sb_smoothing)

        # RDP_EPSILON
        tt = (
            "% of extent — Ramer–Douglas–Peucker epsilon: "
            "points within this distance of the simplified line are discarded"
        )
        self._sb_rdp = QDoubleSpinBox()
        self._sb_rdp.setRange(0.0, 100.0)
        self._sb_rdp.setSingleStep(0.001)
        self._sb_rdp.setDecimals(4)
        self._sb_rdp.setSuffix(" %")
        self._sb_rdp.setToolTip(tt)
        form.addRow(self._labelled("RDP epsilon", tt), self._sb_rdp)

        self._tabs.addTab(self._scroll_wrap(container), "Post-processing")

    def _build_export_tab(self):
        container, form = self._form_container()

        form.addRow(self._section("Output files"))

        # EXPORT_SKELETON_OVERLAY
        tt = "Save a skeleton overlay plot for every feature"
        self._chk_skel_overlay = QCheckBox()
        self._chk_skel_overlay.setToolTip(tt)
        form.addRow(self._labelled("Skeleton overlay plot", tt), self._chk_skel_overlay)

        # EXPORT_PROFILE_PLOT
        tt = "Save width-profile graphs"
        self._chk_profile_plot = QCheckBox()
        self._chk_profile_plot.setToolTip(tt)
        form.addRow(self._labelled("Width profile plot", tt), self._chk_profile_plot)

        # EXPORT_PROFILE_DATA
        tt = "Save per-branch width measurements as CSV files"
        self._chk_profile_data = QCheckBox()
        self._chk_profile_data.setToolTip(tt)
        form.addRow(self._labelled("Width profile CSV", tt), self._chk_profile_data)

        # EXPORT_PROFILE_FFT
        tt = "Save a tortuosity FFT (fast Fourier transform) spectrum plot for every feature"
        self._chk_profile_fft = QCheckBox()
        self._chk_profile_fft.setToolTip(tt)
        form.addRow(self._labelled("Tortuosity FFT plot", tt), self._chk_profile_fft)

        # EXPORT_RAW_TRACES
        tt = (
            "Save skeleton_raw.svg using the full-density smoothed centreline instead of the "
            "decimated output coordinates. Useful for inspecting the pre-simplification path geometry."
        )
        self._chk_raw_traces = QCheckBox()
        self._chk_raw_traces.setToolTip(tt)
        form.addRow(self._labelled("Raw traces SVG", tt), self._chk_raw_traces)

        form.addRow(self._section("SVG output"))

        # OUTPUT_SIZE
        tt = "Display size (pixels) of the longer SVG dimension in skeleton.svg"
        self._sb_output_size = QSpinBox()
        self._sb_output_size.setRange(100, 10000)
        self._sb_output_size.setSingleStep(100)
        self._sb_output_size.setSuffix(" px")
        self._sb_output_size.setToolTip(tt)
        form.addRow(self._labelled("Output size", tt), self._sb_output_size)

        # OUTPUT_RESOLUTION
        tt = "% of extent — minimum vertex spacing written to skeleton.svg (0 = keep all vertices)"
        self._sb_output_res = QDoubleSpinBox()
        self._sb_output_res.setRange(0.0, 100.0)
        self._sb_output_res.setSingleStep(0.001)
        self._sb_output_res.setDecimals(4)
        self._sb_output_res.setSuffix(" %")
        self._sb_output_res.setToolTip(tt)
        form.addRow(self._labelled("Output resolution", tt), self._sb_output_res)

        self._tabs.addTab(self._scroll_wrap(container), "Export")

    def _build_advanced_tab(self):
        container, form = self._form_container()

        form.addRow(self._note(
            "Settings that rarely need changing for typical workflows."
        ))

        form.addRow(self._section("Performance"))

        # N_WORKERS
        tt = "Number of parallel worker threads. 0 = use all logical CPU cores."
        self._sb_n_workers = QSpinBox()
        self._sb_n_workers.setRange(0, 256)
        self._sb_n_workers.setSpecialValueText("All cores (0)")
        self._sb_n_workers.setToolTip(tt)
        form.addRow(self._labelled("Worker threads", tt), self._sb_n_workers)

        self._tabs.addTab(self._scroll_wrap(container), "Advanced")

    # ------------------------------------------------------------------
    # Config load / save
    # ------------------------------------------------------------------

    def _load_from_config(self):
        """Populate all widgets from the current config module values."""
        self._le_input_file.setText(str(config.INPUT_FILE))
        self._le_output_dir.setText(str(config.OUTPUT_DIRECTORY))
        self._chk_separate.setChecked(bool(config.SEPARATE_MULTIPOLYGONS))
        self._chk_topology.setChecked(bool(config.PRESERVE_TOPOLOGY))
        self._sb_image_threshold.setValue(int(config.IMAGE_THRESHOLD))

        self._sb_artboard_size.setValue(config.ARTBOARD_MIN_SIZE)
        self._sb_artboard_rect.setValue(config.ARTBOARD_MIN_RECT)

        self._sb_min_feature.setValue(config.MINIMUM_FEATURE_SIZE)
        idx = self._cb_skel_method.findText(config.SKELETONISATION_METHOD)
        if idx >= 0:
            self._cb_skel_method.setCurrentIndex(idx)
        self._sb_raster_res.setValue(config.RASTER_RESOLUTION)
        self._sb_raster_buf.setValue(config.RASTER_BUFFER)
        self._sb_max_raster_px.setValue(int(config.MAX_RASTER_PIXELS))
        self._sb_min_raster_px.setValue(int(config.MIN_RASTER_PIXELS))
        self._sb_min_branch_px.setValue(int(config.MIN_BRANCH_PIXELS))
        self._sb_min_branch_pct.setValue(config.MIN_BRANCH_PERCENT)
        self._sb_curvature.setValue(config.CURVATURE_THRESHOLD)
        self._sb_solidity.setValue(config.SOLIDITY_THRESHOLD)
        self._sb_aspect.setValue(config.ASPECT_RATIO_THRESHOLD)
        self._sb_escape.setValue(config.ESCAPE_THRESHOLD)
        self._sb_branching.setValue(config.BRANCHING_THRESHOLD)

        self._sb_sampling.setValue(config.SAMPLING_INTERVAL)
        self._sb_max_ray.setValue(config.MAX_WIDTH_RAY_DISTANCE)
        self._sb_smoothing.setValue(config.SMOOTHING)
        self._sb_rdp.setValue(config.RDP_EPSILON)

        self._chk_skel_overlay.setChecked(bool(config.EXPORT_SKELETON_OVERLAY))
        self._chk_profile_plot.setChecked(bool(config.EXPORT_PROFILE_PLOT))
        self._chk_profile_data.setChecked(bool(config.EXPORT_PROFILE_DATA))
        self._chk_profile_fft.setChecked(bool(config.EXPORT_PROFILE_FFT))
        self._chk_raw_traces.setChecked(bool(config.EXPORT_RAW_TRACES))
        self._sb_output_size.setValue(int(config.OUTPUT_SIZE))
        self._sb_output_res.setValue(config.OUTPUT_RESOLUTION)

        # N_WORKERS: None → 0 (special "all cores" value)
        self._sb_n_workers.setValue(0 if config.N_WORKERS is None else int(config.N_WORKERS))

    def _apply_to_config(self):
        """Write all widget values back to the config module globals."""
        setattr(config, "INPUT_FILE",            self._le_input_file.text())
        setattr(config, "OUTPUT_DIRECTORY",       self._le_output_dir.text())
        setattr(config, "SEPARATE_MULTIPOLYGONS", self._chk_separate.isChecked())
        setattr(config, "PRESERVE_TOPOLOGY",      self._chk_topology.isChecked())
        setattr(config, "IMAGE_THRESHOLD",        self._sb_image_threshold.value())

        setattr(config, "ARTBOARD_MIN_SIZE", self._sb_artboard_size.value())
        setattr(config, "ARTBOARD_MIN_RECT", self._sb_artboard_rect.value())

        setattr(config, "MINIMUM_FEATURE_SIZE",   self._sb_min_feature.value())
        setattr(config, "SKELETONISATION_METHOD", self._cb_skel_method.currentText())
        setattr(config, "RASTER_RESOLUTION",      self._sb_raster_res.value())
        setattr(config, "RASTER_BUFFER",          self._sb_raster_buf.value())
        setattr(config, "MAX_RASTER_PIXELS",      self._sb_max_raster_px.value())
        setattr(config, "MIN_RASTER_PIXELS",      self._sb_min_raster_px.value())
        setattr(config, "MIN_BRANCH_PIXELS",      self._sb_min_branch_px.value())
        setattr(config, "MIN_BRANCH_PERCENT",     self._sb_min_branch_pct.value())
        setattr(config, "CURVATURE_THRESHOLD",    self._sb_curvature.value())
        setattr(config, "SOLIDITY_THRESHOLD",     self._sb_solidity.value())
        setattr(config, "ASPECT_RATIO_THRESHOLD", self._sb_aspect.value())
        setattr(config, "ESCAPE_THRESHOLD",       self._sb_escape.value())
        setattr(config, "BRANCHING_THRESHOLD",    self._sb_branching.value())

        setattr(config, "SAMPLING_INTERVAL",       self._sb_sampling.value())
        setattr(config, "MAX_WIDTH_RAY_DISTANCE",  self._sb_max_ray.value())
        setattr(config, "SMOOTHING",               self._sb_smoothing.value())
        setattr(config, "RDP_EPSILON",             self._sb_rdp.value())

        setattr(config, "EXPORT_SKELETON_OVERLAY", self._chk_skel_overlay.isChecked())
        setattr(config, "EXPORT_PROFILE_PLOT",     self._chk_profile_plot.isChecked())
        setattr(config, "EXPORT_PROFILE_DATA",     self._chk_profile_data.isChecked())
        setattr(config, "EXPORT_PROFILE_FFT",      self._chk_profile_fft.isChecked())
        setattr(config, "EXPORT_RAW_TRACES",       self._chk_raw_traces.isChecked())
        setattr(config, "OUTPUT_SIZE",             self._sb_output_size.value())
        setattr(config, "OUTPUT_RESOLUTION",       self._sb_output_res.value())

        # N_WORKERS: 0 → None (all cores)
        n = self._sb_n_workers.value()
        setattr(config, "N_WORKERS", None if n == 0 else n)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input file",
            self._le_input_file.text(),
            "Supported files (*.shp *.svg *.pdf *.jpg *.jpeg *.png *.tif *.tiff);;"
            "All files (*)",
        )
        if path:
            self._le_input_file.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self._le_output_dir.text(),
        )
        if path:
            self._le_output_dir.setText(path)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _on_run(self):
        self._apply_to_config()
        self._log.clear()

        # Redirect stdout through the signal-based stream so the worker thread
        # never touches Qt widgets directly (would crash on any UI interaction).
        sys.stdout = self._log_stream

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status_label.setText("Status: Running…")

        self._worker = _Worker()
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._restore_stdout()
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText("Status: Stopped")

    def _on_finished(self):
        self._restore_stdout()
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText("Status: Finished")

    def _on_error(self, tb: str):
        # Signal is delivered to main thread via queued connection — safe to
        # call _append_log directly here.
        self._append_log("\n--- ERROR ---\n" + tb)

    def _append_log(self, text: str):
        """Append *text* to the log panel and to the original terminal stdout."""
        # Mirror to terminal so output survives a GUI crash or early exit
        self._real_stdout.write(text)
        self._real_stdout.flush()
        # Update the in-window log panel (main thread only)
        cursor = self._log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._log.setTextCursor(cursor)
        self._log.insertPlainText(text)
        self._log.ensureCursorVisible()

    def _restore_stdout(self):
        sys.stdout = self._real_stdout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
