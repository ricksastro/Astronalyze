#!/usr/bin/env python3
"""
Astronalyze - Astronomical Image Viewer for FITS and XISF files

Usage:
    python astronalyze.py [directory]
"""

import json
import os
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path

if getattr(sys, 'frozen', False):
    CONFIG_PATH = Path(sys.executable).parent / '.astronalyze_config.json'
else:
    CONFIG_PATH = Path(__file__).with_name('.astronalyze_config.json')


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(cfg: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg))
    except Exception:
        pass


def _load_last_dir() -> str | None:
    d = _load_config().get('last_dir')
    return d if d and Path(d).is_dir() else None


def _save_last_dir(path: str):
    cfg = _load_config()
    cfg['last_dir'] = path
    _save_config(cfg)

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

try:
    from photutils.detection import DAOStarFinder
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from xisf import XISF
    HAS_XISF = True
except ImportError:
    HAS_XISF = False

try:
    from send2trash import send2trash as _send2trash
    HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# Use numpy-style (row, col) array indexing throughout
pg.setConfigOption('imageAxisOrder', 'row-major')

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG        = '#1a1a2e'
TOOLBAR   = '#16213e'
BTN       = '#0f3460'
BTN_DEL   = '#922b21'
FG        = '#e0e0e0'
FG_ACCENT = '#64ffda'
FG_DIM    = '#a8b2d8'

# Grayscale LUT for single-channel images
GRAY_LUT = np.repeat(np.arange(256, dtype=np.uint8).reshape(256, 1), 3, axis=1)

# ---------------------------------------------------------------------------
# Image processing helpers  (unchanged from matplotlib version)
# ---------------------------------------------------------------------------

def autostretch(data: np.ndarray) -> np.ndarray:
    """
    PixInsight-style Screen Transfer Function (STF) autostretch.
    Returns float32 array in [0, 1].
    """
    data = data.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    valid = data[np.isfinite(data)].ravel()
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    if valid.size > 100_000:
        rng = np.random.default_rng(0)
        sample = rng.choice(valid, 100_000, replace=False)
    else:
        sample = valid

    vmin = np.percentile(sample, 0.05)
    vmax = np.percentile(sample, 99.95)
    span = vmax - vmin
    if span < 1e-10:
        return np.zeros_like(data, dtype=np.float32)

    data = np.clip((data - vmin) / span, 0.0, 1.0)

    flat = data.ravel()
    if flat.size > 100_000:
        rng = np.random.default_rng(0)
        sample2 = rng.choice(flat, 100_000, replace=False)
    else:
        sample2 = flat
    median = np.median(sample2)
    mad = np.median(np.abs(sample2 - median))
    c0 = float(np.clip(median - 2.8 * mad, 0.0, 1.0))

    data = np.clip((data - c0) / (1.0 - c0 + 1e-10), 0.0, 1.0)

    m = float(np.median(data))
    target = 0.25

    if 0.0 < m < 1.0:
        denom = m * (2.0 * target - 1.0) - target
        if abs(denom) > 1e-10:
            mtf_m = m * (target - 1.0) / denom
            if mtf_m > 0.0:
                result = (mtf_m - 1.0) * data / ((2.0 * mtf_m - 1.0) * data - mtf_m)
                result = np.where(data == 0.0, 0.0,
                         np.where(data == 1.0, 1.0, result))
                return np.clip(result, 0.0, 1.0).astype(np.float32)

    return data.astype(np.float32)


def _bit_depth_max(data_max: float) -> float:
    """Return the natural full-scale value inferred from the data maximum."""
    if data_max <= 1.0:
        return 1.0
    if data_max <= 255.0:
        return 255.0
    if data_max <= 65535.0:
        return 65535.0
    return float(data_max)


def linear_stretch(data: np.ndarray) -> np.ndarray:
    """Linear map from 0 (or data min if negative) to the natural bit-depth max."""
    data = data.astype(np.float64)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    vmin = min(0.0, float(valid.min()))
    vmax = _bit_depth_max(float(valid.max()))
    span = vmax - vmin
    if span < 1e-10:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - vmin) / span, 0.0, 1.0).astype(np.float32)


def compute_stretch_params(data: np.ndarray, auto: bool) -> dict | None:
    """
    Compute stretch parameters from image statistics using a fast sample.
    Returns a params dict that apply_stretch can use on any crop of the image.
    """
    d = data
    if d.ndim == 3:
        d = d[0] if (d.shape[0] in (1, 3) and d.shape[0] < d.shape[1]) else d[:, :, 0]
    flat = d.ravel().astype(np.float64)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    rng = np.random.default_rng(0)
    sample = rng.choice(flat, min(flat.size, 100_000), replace=False)

    if not auto:
        vmin = min(0.0, float(sample.min()))
        vmax = _bit_depth_max(float(sample.max()))
        span = vmax - vmin
        return None if span < 1e-10 else {'auto': False, 'vmin': vmin, 'span': span}

    vmin = float(np.percentile(sample, 0.05))
    vmax = float(np.percentile(sample, 99.95))
    span = vmax - vmin
    if span < 1e-10:
        return None
    norm = np.clip((sample - vmin) / span, 0.0, 1.0)
    median = float(np.median(norm))
    mad = float(np.median(np.abs(norm - median)))
    c0 = float(np.clip(median - 2.8 * mad, 0.0, 1.0))
    adj = np.clip((norm - c0) / (1.0 - c0 + 1e-10), 0.0, 1.0)
    m = float(np.median(adj))
    mtf_m = None
    if 0.0 < m < 1.0:
        denom = m * (2.0 * 0.25 - 1.0) - 0.25
        if abs(denom) > 1e-10:
            v = m * (0.25 - 1.0) / denom
            if v > 0.0:
                mtf_m = float(v)
    return {'auto': True, 'vmin': vmin, 'span': span, 'c0': c0, 'mtf_m': mtf_m}


def apply_stretch(data: np.ndarray, params: dict) -> np.ndarray:
    """Apply pre-computed stretch parameters to a 2-D array. Fast — no stat recomputation."""
    d = data.astype(np.float64)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip((d - params['vmin']) / params['span'], 0.0, 1.0)
    if not params['auto']:
        return d.astype(np.float32)
    c0 = params['c0']
    d = np.clip((d - c0) / (1.0 - c0 + 1e-10), 0.0, 1.0)
    mtf_m = params['mtf_m']
    if mtf_m is not None:
        result = (mtf_m - 1.0) * d / ((2.0 * mtf_m - 1.0) * d - mtf_m)
        result = np.where(d == 0.0, 0.0, np.where(d == 1.0, 1.0, result))
        return np.clip(result, 0.0, 1.0).astype(np.float32)
    return d.astype(np.float32)


def prepare_display(data: np.ndarray, stretch: bool,
                    params: dict | None = None) -> tuple:
    """
    Returns (display_array, is_gray).
    display_array is float32 in [0,1], shaped (H,W) or (H,W,3).
    When params is supplied the stretch is applied without recomputing statistics.
    """
    if params is not None:
        fn = lambda d: apply_stretch(d, params)
    else:
        fn = autostretch if stretch else linear_stretch

    if data.ndim == 2:
        return fn(data), True

    if data.ndim == 3:
        if data.shape[0] in (1, 3) and data.shape[0] < data.shape[1]:
            data = np.moveaxis(data, 0, -1)
        c = data.shape[2]
        if c == 1:
            return fn(data[:, :, 0]), True
        if c >= 3:
            channels = [fn(data[:, :, i]) for i in range(3)]
            return np.stack(channels, axis=-1), False

    return fn(data.squeeze()), True

# ---------------------------------------------------------------------------
# FITS / XISF loaders
# ---------------------------------------------------------------------------

def load_fits(path: str) -> tuple[np.ndarray, object]:
    """Return (data float32, primary image header)."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.data.astype(np.float32), hdu.header
    raise ValueError("No image data found in FITS file")


def extract_plate_scale(hdr) -> float | None:
    """
    Try to read arcsec/pixel plate scale from a FITS header object.
    Returns arcsec/pixel or None if not found.
    """
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        try:
            wcs = WCS(hdr, naxis=2)
            if wcs.has_celestial:
                scales = proj_plane_pixel_scales(wcs)
                return float(np.mean(np.abs(scales))) * 3600.0
        except Exception:
            pass
        for key in ('CDELT2', 'CDELT1'):
            if key in hdr:
                val = abs(float(hdr[key]))
                if val < 1.0:
                    return val * 3600.0
                if val < 3600.0:
                    return val
        for key in ('PIXSCALE', 'PIXSCAL1', 'PIXSCAL2', 'SCALE', 'PLATESCL'):
            if key in hdr:
                return abs(float(hdr[key]))
        focal_mm = None
        for key in ('FOCALLEN', 'FOCAL', 'FL'):
            if key in hdr:
                focal_mm = float(hdr[key])
                break
        pix_um = None
        for key in ('XPIXSZ', 'PIXSIZE1', 'PIXELSX'):
            if key in hdr:
                pix_um = float(hdr[key])
                break
        if focal_mm and pix_um and focal_mm > 0:
            return 206.265 * pix_um / focal_mm
    except Exception:
        pass
    return None


def load_xisf(path: str) -> tuple[np.ndarray, dict]:
    """Return (data float32, flat keyword dict) for the first image in an XISF file."""
    if not HAS_XISF:
        raise ImportError("xisf library not installed.\nRun:  pip install xisf")
    xisf_obj = XISF(path)
    data = xisf_obj.read_image(0).astype(np.float32)
    kw: dict = {}
    try:
        meta = xisf_obj.get_images_metadata()
        if meta:
            for key, entries in meta[0].get('FITSKeywords', {}).items():
                if entries:
                    kw[key] = entries[0].get('value', '')
    except Exception:
        pass
    return data, kw


def extract_plate_scale_xisf(kw: dict) -> float | None:
    """Extract arcsec/pixel plate scale from an XISF flat keyword dict."""
    try:
        focal_mm = None
        for key in ('FOCALLEN', 'FOCAL', 'FL'):
            if key in kw:
                focal_mm = float(kw[key])
                break
        pix_um = None
        for key in ('XPIXSZ', 'PIXSIZE1', 'PIXELSX'):
            if key in kw:
                pix_um = float(kw[key])
                break
        if focal_mm and pix_um and focal_mm > 0:
            return 206.265 * pix_um / focal_mm
        for key in ('PIXSCALE', 'PIXSCAL1', 'PLATESCL', 'SCALE'):
            if key in kw:
                return abs(float(kw[key]))
    except Exception:
        pass
    return None


def extract_telescope_info(kw) -> tuple:
    """Extract (name, focal_mm, fratio) from a FITS header or XISF keyword dict.
    Returns (str|None, float|None, float|None)."""
    def _get(k):
        try:
            v = kw[k]
            return v if v not in (None, '', 'N/A') else None
        except (KeyError, TypeError):
            return None

    name = None
    for key in ('TELESCOP', 'SCOPE', 'TELESCOPE', 'INSTRUME'):
        v = _get(key)
        if v:
            name = str(v).strip()
            break

    focal_mm = None
    for key in ('FOCALLEN', 'FOCAL', 'FL'):
        v = _get(key)
        if v is not None:
            try:
                focal_mm = float(v)
                break
            except (ValueError, TypeError):
                pass

    fratio = None
    for key in ('FOCRATIO', 'FRATIO'):
        v = _get(key)
        if v is not None:
            try:
                fratio = float(v)
                break
            except (ValueError, TypeError):
                pass
    if fratio is None and focal_mm:
        for key in ('APTDIA', 'APERTURE', 'APTDIAM'):
            v = _get(key)
            if v is not None:
                try:
                    apt = float(v)
                    if apt > 0:
                        fratio = focal_mm / apt
                    break
                except (ValueError, TypeError):
                    pass

    return name, focal_mm, fratio

# ---------------------------------------------------------------------------
# 2-D Moffat (beta=4) for star fitting
# ---------------------------------------------------------------------------

_MOFFAT_BETA = 4.0
# FWHM = 2 * alpha * sqrt(2^(1/beta) - 1); pre-compute the factor for beta=4
_MOFFAT_FWHM_FACTOR = 2.0 * float(np.sqrt(2.0 ** (1.0 / _MOFFAT_BETA) - 1.0))  # ≈ 1.287
# For initial alpha guess: half-max radius ≈ alpha * sqrt(2^(1/beta)-1)
_MOFFAT_ALPHA_SCALE = float(np.sqrt(2.0 ** (1.0 / _MOFFAT_BETA) - 1.0))         # ≈ 0.644


def _moffat4_2d(xy, amplitude, xo, yo, alpha_x, alpha_y, theta, offset):
    """Elliptical 2-D Moffat profile with beta fixed at 4."""
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = x - xo, y - yo
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    rr = (u / alpha_x) ** 2 + (v / alpha_y) ** 2
    return (offset + amplitude * (1.0 + rr) ** (-_MOFFAT_BETA)).ravel()


def fit_star_fwhm(cutout: np.ndarray, cx_hint: float = None, cy_hint: float = None):
    """Fit 2-D Moffat4 to a star cutout; return FWHM in pixels or None.

    cx_hint / cy_hint: expected star centre within the cutout (pixels).
    When supplied the fit is anchored near that position and amplitude is
    estimated from that pixel rather than from argmax — critical for stars
    embedded in extended emission (nebulae) where the brightest pixel in
    the cutout may not be the star itself.
    """
    if not HAS_SCIPY or cutout.size == 0:
        return None
    h, w = cutout.shape
    if h < 5 or w < 5:
        return None
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    bg = float(np.percentile(cutout, 10))

    if cx_hint is not None and cy_hint is not None:
        xo = float(np.clip(cx_hint, 0, w - 1))
        yo = float(np.clip(cy_hint, 0, h - 1))
        amp = float(cutout[int(round(yo)), int(round(xo))]) - bg
        # Restrict position to ±5 px of the hint so the fit cannot wander
        # to a brighter unrelated feature (nebula knot, neighbouring star).
        x_lo, x_hi = max(0.0, xo - 5), min(float(w), xo + 5)
        y_lo, y_hi = max(0.0, yo - 5), min(float(h), yo + 5)
    else:
        yo, xo = np.unravel_index(np.argmax(cutout), cutout.shape)
        xo, yo = float(xo), float(yo)
        amp = float(cutout.max()) - bg
        x_lo, x_hi = 0.0, float(w)
        y_lo, y_hi = 0.0, float(h)

    if amp <= 0:
        return None
    # Estimate initial alpha from pixels above half-maximum.
    # For Moffat4, half-max radius ≈ alpha * _MOFFAT_ALPHA_SCALE, so
    # alpha ≈ sqrt(above/pi) / _MOFFAT_ALPHA_SCALE.
    half_max = bg + amp * 0.5
    above = max(1, int((cutout >= half_max).sum()))
    alpha0 = float(np.clip(
        np.sqrt(above / np.pi) / _MOFFAT_ALPHA_SCALE,
        0.5, min(h, w) / 3.0,
    ))
    alpha_max = min(h, w) / 2.0
    try:
        popt, _ = curve_fit(
            _moffat4_2d,
            (x_idx, y_idx),
            cutout.ravel(),
            p0=[amp, xo, yo, alpha0, alpha0, 0.0, bg],
            bounds=(
                [0, x_lo, y_lo, 0.3, 0.3, -np.pi, -np.inf],
                [np.inf, x_hi, y_hi, alpha_max, alpha_max, np.pi, np.inf],
            ),
            maxfev=2000,
        )
        ax, ay = abs(popt[3]), abs(popt[4])
        if max(ax, ay) / max(min(ax, ay), 0.1) > 2.0:   # too elongated → double/blend
            return None
        fwhm_x = _MOFFAT_FWHM_FACTOR * ax
        fwhm_y = _MOFFAT_FWHM_FACTOR * ay
        return float(np.sqrt((fwhm_x ** 2 + fwhm_y ** 2) / 2.0))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

SUPPORTED_EXT = {'.fits', '.fit', '.fts', '.xisf'}


def _make_precomp(data: np.ndarray, auto: bool) -> dict:
    """Pre-render a display-ready uint8 image from raw data (runs in background).
    Returns a dict consumed by FITSViewer._fast_display()."""
    # Determine native dimensions
    if data.ndim == 2:
        h, w = data.shape
    elif data.shape[0] in (1, 3) and data.shape[0] < data.shape[1]:
        h, w = data.shape[1], data.shape[2]
    else:
        h, w = data.shape[0], data.shape[1]
    # Build downsampled overview (mirrors _overview logic)
    stride = max(1, max(w // 4096, h // 4096))
    if data.ndim == 2:
        ov = data[::stride, ::stride]
    elif data.shape[0] in (1, 3) and data.shape[0] < data.shape[1]:
        ov = data[:, ::stride, ::stride]
    else:
        ov = data[::stride, ::stride]
    stretch_params = compute_stretch_params(data, auto)
    display, is_gray = prepare_display(ov, auto, stretch_params)
    img_u8 = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)
    valid_max = float(data[np.isfinite(data)].max()) if data is not None else 1.0
    pixel_scale = 65535.0 if valid_max <= 1.0 else 1.0
    return {'stretch_params': stretch_params, 'img_u8': img_u8,
            'is_gray': is_gray, 'h': h, 'w': w, 'auto': auto,
            'pixel_scale': pixel_scale}


# ---------------------------------------------------------------------------
# Corner inspector widget
# ---------------------------------------------------------------------------

_INSPECTOR_CELLS = [
    (0, 0, 'TL'),    (0, 1, 'Top'),    (0, 2, 'TR'),
    (1, 0, 'Left'),  (1, 1, 'Center'), (1, 2, 'Right'),
    (2, 0, 'BL'),    (2, 1, 'Bottom'), (2, 2, 'BR'),
]


class _InspectorCell(QtWidgets.QLabel):
    """One panel in the 3×3 inspector grid.

    Stars are stored as (img_x, img_y, label_text) in full-image pixel
    coordinates.  paintEvent converts them to cell-local screen coordinates
    using the stored crop origin and actual crop size, which means the
    overlay stays correct even if the window is resized after FWHM is computed.
    """

    _CIRCLE_R  = 6          # pixels — matches scatter size=12 in the map
    _ACCENT    = QtGui.QColor(0x64, 0xff, 0xda)   # FG_ACCENT

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label   = label
        self._stars:  list[tuple] = []   # (img_x, img_y, label_text)
        self._crop_x0 = 0
        self._crop_y0 = 0
        self._act_w   = 0   # actual crop width in image pixels
        self._act_h   = 0
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(50, 50)
        self.setStyleSheet('background: #000;')

    def set_stars(self, stars: list, crop_x0: int, crop_y0: int,
                  act_w: int, act_h: int):
        self._stars   = stars
        self._crop_x0 = crop_x0
        self._crop_y0 = crop_y0
        self._act_w   = act_w
        self._act_h   = act_h
        self.update()

    def clear_stars(self):
        self._stars = []
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        # Position label (TL / Top / Center …)
        painter.setPen(QtGui.QColor(200, 200, 200, 200))
        painter.drawText(4, 14, self._label)
        # Star circles + labels
        if self._stars:
            cw, ch = self.width(), self.height()
            pad_x = max(0, (cw - self._act_w) // 2) if self._act_w else 0
            pad_y = max(0, (ch - self._act_h) // 2) if self._act_h else 0
            r = self._CIRCLE_R
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            fm = painter.fontMetrics()
            for img_x, img_y, lbl_text in self._stars:
                sx = int(img_x - self._crop_x0) + pad_x
                sy = int(img_y - self._crop_y0) + pad_y
                painter.drawEllipse(QtCore.QPoint(sx, sy), r, r)
                painter.setPen(QtGui.QPen(self._ACCENT))
                lbl_w = fm.horizontalAdvance(lbl_text)
                tx = (sx - r - 2 - lbl_w
                      if sx + r + 2 + lbl_w > cw
                      else sx + r + 2)
                painter.drawText(tx, sy - r + 2, lbl_text)
                painter.setPen(pen)
        painter.end()


class _InspectorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(2)
        grid.setContentsMargins(2, 2, 2, 2)
        self._cells: dict[tuple[int, int], _InspectorCell] = {}
        for row, col, lbl in _INSPECTOR_CELLS:
            cell = _InspectorCell(lbl)
            grid.addWidget(cell, row, col)
            grid.setRowStretch(row, 1)
            grid.setColumnStretch(col, 1)
            self._cells[(row, col)] = cell
        self._data: np.ndarray | None = None
        self._stretch_params: dict | None = None
        self._on_rendered = None   # optional callback after cells are repainted

    def load(self, data: np.ndarray, stretch_params):
        self._data = data
        self._stretch_params = stretch_params
        self._render_all()

    # ------------------------------------------------------------------
    def _img_dims(self) -> tuple[int, int]:
        d = self._data
        if d.ndim == 3:
            if d.shape[0] in (1, 3) and d.shape[0] < d.shape[1]:
                return d.shape[1], d.shape[2]
            return d.shape[0], d.shape[1]
        return d.shape[0], d.shape[1]

    def _crop_origin(self, row: int, col: int,
                     img_w: int, img_h: int, cw: int, ch: int) -> tuple[int, int]:
        x0 = (0 if col == 0 else
               max(0, img_w // 2 - cw // 2) if col == 1 else
               max(0, img_w - cw))
        y0 = (0 if row == 0 else
               max(0, img_h // 2 - ch // 2) if row == 1 else
               max(0, img_h - ch))
        return x0, y0

    def crop_region(self, row: int, col: int) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, x1, y1) in full-image pixels for this cell."""
        if self._data is None:
            return None
        img_h, img_w = self._img_dims()
        cell = self._cells[(row, col)]
        cw, ch = max(cell.width(), 1), max(cell.height(), 1)
        x0, y0 = self._crop_origin(row, col, img_w, img_h, cw, ch)
        return x0, y0, min(x0 + cw, img_w), min(y0 + ch, img_h)

    def _get_raw_crop(self, row: int, col: int):
        """Return (raw_crop, cw, ch) or None. crop is a view/slice of self._data."""
        cell = self._cells[(row, col)]
        cw, ch = cell.width(), cell.height()
        if cw < 10 or ch < 10:
            return None
        img_h, img_w = self._img_dims()
        x0, y0 = self._crop_origin(row, col, img_w, img_h, cw, ch)
        x1, y1 = min(x0 + cw, img_w), min(y0 + ch, img_h)
        d = self._data
        if d.ndim == 3:
            crop = (d[:, y0:y1, x0:x1] if d.shape[0] in (1, 3) and d.shape[0] < d.shape[1]
                    else d[y0:y1, x0:x1, :])
        else:
            crop = d[y0:y1, x0:x1]
        return crop, cw, ch

    @staticmethod
    def _raw_sky(crop: np.ndarray) -> float:
        """5th-percentile of luminance in raw data — sky background estimate."""
        if crop.ndim == 3:
            lum = (crop.mean(axis=0)
                   if crop.shape[0] in (1, 3) and crop.shape[0] < crop.shape[1]
                   else crop.mean(axis=-1))
        else:
            lum = crop
        flat = lum.ravel().astype(np.float64)
        flat = flat[np.isfinite(flat)]
        return float(np.percentile(flat, 5)) if flat.size > 0 else 0.0

    @staticmethod
    def _bg_level(display: np.ndarray) -> float:
        """10th-percentile of luminance in display space — sky background estimate."""
        lum = display if display.ndim == 2 else display.mean(axis=-1)
        return float(np.percentile(lum.ravel(), 10))

    @staticmethod
    def _display_to_pixmap(display: np.ndarray, is_gray: bool,
                           cw: int, ch: int) -> 'QtGui.QPixmap':
        img_u8 = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)
        act_h, act_w = img_u8.shape[:2]
        img_rgb = np.stack([img_u8] * 3, axis=-1) if is_gray else img_u8
        img_rgb = np.ascontiguousarray(img_rgb)
        qimg = QtGui.QImage(img_rgb.data, act_w, act_h, 3 * act_w,
                            QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        if act_w < cw or act_h < ch:
            full = QtGui.QPixmap(cw, ch)
            full.fill(QtGui.QColor(0, 0, 0))
            p = QtGui.QPainter(full)
            p.drawPixmap((cw - act_w) // 2, (ch - act_h) // 2, pix)
            p.end()
            pix = full
        return pix

    def _render_all(self):
        if self._data is None:
            return

        # Pass 1: extract raw crops and measure sky in linear (raw) space
        raw_crops: dict = {}
        for row, col, _ in _INSPECTOR_CELLS:
            result = self._get_raw_crop(row, col)
            if result is not None:
                raw_crops[(row, col)] = result

        if not raw_crops:
            return

        skies = {k: self._raw_sky(v[0]) for k, v in raw_crops.items()}
        ref_sky = float(np.median(list(skies.values())))

        # Pass 2: equalize raw sky → stretch → nudge display to TARGET_BG
        TARGET_BG = 0.10
        for (row, col), (crop, cw, ch) in raw_crops.items():
            raw_offset = ref_sky - skies[(row, col)]
            adjusted = crop.astype(np.float32) + raw_offset
            display, is_gray = prepare_display(adjusted, True, self._stretch_params)
            disp_offset = TARGET_BG - self._bg_level(display)
            display = np.clip(display + disp_offset, 0.0, 1.0).astype(np.float32)
            self._cells[(row, col)].setPixmap(
                self._display_to_pixmap(display, is_gray, cw, ch)
            )

    def set_cell_stars(self, row: int, col: int, stars: list,
                       crop_x0: int, crop_y0: int, act_w: int, act_h: int):
        self._cells[(row, col)].set_stars(stars, crop_x0, crop_y0, act_w, act_h)

    def clear_stars(self):
        for cell in self._cells.values():
            cell.clear_stars()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_all()
        if self._on_rendered:
            self._on_rendered()


class Astronalyze(QtWidgets.QMainWindow):

    # Signals for thread-safe result delivery
    _sig_map_done        = QtCore.pyqtSignal(object, object, object, object)  # (rgba, contour paths, fwhm_pts, avg_fwhm)
    _sig_map_msg         = QtCore.pyqtSignal(str)
    _sig_display         = QtCore.pyqtSignal(int, int, object)  # (gen, index, path) — queued load ready
    _sig_inspector_done  = QtCore.pyqtSignal(dict)   # {(row,col): fwhm_px or None}

    def __init__(self, start_path: str | None = None):
        super().__init__()
        self.setWindowTitle('Astronalyze')
        self.resize(1280, 920)

        self._directory: Path | None = None
        self._files: list[Path] = []
        self._index: int = 0
        self._data: np.ndarray | None = None
        self._plate_scale: float | None = None
        self._scope_info: tuple = (None, None, None)   # (name, focal_mm, fratio)
        self._pixel_scale: float = 1.0   # 65535 for normalized [0,1] float data
        self._stretch_params: dict | None = None
        self._last_map_fwhm: tuple | None = None  # (avg_fwhm, center_fwhm)
        self._fwhm_label_items: list = []
        self._contour_label_items: list = []
        self._map_visible:       bool = False
        self._map_computing:     bool = False
        self._map_color_visible:   bool = True
        self._map_contour_visible: bool = True
        self._contour_items: list = []
        self._fwhm_unit: str = 'arcsec'   # 'arcsec' or 'px'
        self._fwhm_pts: list = []
        self._contour_level_segs: dict = {}
        self._inspector_active: bool = False
        self._inspector_computing: bool = False
        self._inspector_fwhm_result: dict = {}   # {(row,col): [(img_x,img_y,fwhm_px),...]}
        self._inspector_regions: dict = {}        # {(row,col): (x0,y0,x1,y1)}

        # Pre-fetch cache: keeps up to 3 images (current + neighbours) in RAM
        # so Next/Prev is instant after the first load.
        self._cache: OrderedDict[Path, tuple] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._prefetching: set[Path] = set()
        self._prefetch_lock = threading.Lock()

        self._files_default: list[Path] = []   # original directory sort order
        self._active_sort: str = 'default'   # 'default' | 'fwhm' | 'stars'
        self._sort_reversed: bool = False

        # Sequential navigation queue: one background thread loads in order,
        # each result is displayed via signal — no main-thread blocking ever.
        import queue as _q
        self._nav_q: _q.Queue = _q.Queue()
        self._nav_loader_active: bool = False
        self._nav_gen: int = 0   # incremented on direct jumps to discard stale queue results
        self._nav_index: int = 0  # tracks the last index enqueued (may be ahead of _index)
        self._playing: bool = False

        self._sig_map_done.connect(self._show_fwhm_map)
        self._sig_map_msg.connect(self._on_map_msg)
        self._sig_display.connect(self._on_display_ready)
        self._sig_inspector_done.connect(self._show_inspector_fwhm)

        self._build_ui()
        self._setup_shortcuts()
        self._apply_style()
        self._restore_geometry()

        if start_path:
            p = Path(start_path)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
                QtCore.QTimer.singleShot(100, lambda: self._open_dir(str(p.parent), select_file=p))
            elif p.is_dir():
                QtCore.QTimer.singleShot(100, lambda: self._open_dir(str(p)))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _mk_btn(self, text: str, slot) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _vsep() -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet(f'color: {FG_DIM};')
        return f

    def _restore_geometry(self):
        geom_hex = _load_config().get('geometry')
        if geom_hex:
            try:
                self.restoreGeometry(QtCore.QByteArray(bytes.fromhex(geom_hex)))
            except Exception:
                pass

    def closeEvent(self, event):
        cfg = _load_config()
        cfg['geometry'] = bytes(self.saveGeometry()).hex()
        _save_config(cfg)
        super().closeEvent(event)

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── top toolbar ──
        top = QtWidgets.QWidget()
        trow = QtWidgets.QHBoxLayout(top)
        trow.setContentsMargins(4, 2, 4, 2)
        trow.addWidget(self._mk_btn('Open Directory', self._choose_dir))
        self._lbl_file = QtWidgets.QLabel('No directory open')
        trow.addWidget(self._lbl_file, 1)
        self._lbl_count = QtWidgets.QLabel('')
        trow.addWidget(self._lbl_count)
        layout.addWidget(top)

        # ── centre: image canvas + zoom bar (left) | file list (right) ──
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._splitter = splitter
        splitter.setChildrenCollapsible(False)

        # Left panel: canvas + zoom bar
        left = QtWidgets.QWidget()
        lcol = QtWidgets.QVBoxLayout(left)
        lcol.setSpacing(2)
        lcol.setContentsMargins(0, 0, 0, 0)

        # PyQtGraph ViewBox handles pan (left-drag) and zoom (scroll) natively via GPU.
        # No re-rendering is needed for viewport changes — only for load/stretch toggle.
        self._gview = pg.GraphicsLayoutWidget()
        self._vb    = self._gview.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)          # y=0 at top, matches image row convention

        self._img_item = pg.ImageItem()
        self._img_item.setLookupTable(GRAY_LUT)
        self._vb.addItem(self._img_item)

        self._overlay_item = pg.ImageItem()
        self._overlay_item.setZValue(1)
        self._overlay_item.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        self._vb.addItem(self._overlay_item)

        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(FG_ACCENT, width=1.5),
            brush=pg.mkBrush(None),
            size=12, symbol='o',
        )
        self._scatter.setZValue(3)
        self._vb.addItem(self._scatter)

        self._vb.sigRangeChanged.connect(self._update_zoom_label)

        self._inspector_widget = _InspectorWidget()
        self._inspector_widget._on_rendered = self._on_inspector_cells_rendered
        self._view_stack = QtWidgets.QStackedWidget()
        self._view_stack.addWidget(self._gview)
        self._view_stack.addWidget(self._inspector_widget)
        lcol.addWidget(self._view_stack, 1)

        # Zoom toolbar
        zbar = QtWidgets.QWidget()
        zrow = QtWidgets.QHBoxLayout(zbar)
        zrow.setContentsMargins(8, 2, 8, 2)
        zrow.addWidget(QtWidgets.QLabel('Zoom:'))
        zrow.addWidget(self._mk_btn('−', lambda: self._zoom_by(1.5)))
        zrow.addWidget(self._mk_btn('+', lambda: self._zoom_by(1 / 1.5)))
        zrow.addWidget(self._vsep())
        zrow.addWidget(self._mk_btn('50%',  lambda: self._zoom_to_ratio(2.0)))
        zrow.addWidget(self._mk_btn('100%', lambda: self._zoom_to_ratio(1.0)))
        zrow.addWidget(self._mk_btn('200%', lambda: self._zoom_to_ratio(0.5)))
        zrow.addWidget(self._vsep())
        zrow.addWidget(self._mk_btn('Fit', self._zoom_fit))
        self._lbl_zoom = QtWidgets.QLabel('')
        zrow.addWidget(self._lbl_zoom)
        zrow.addStretch()
        self._lbl_scope = QtWidgets.QLabel('')
        zrow.addWidget(self._lbl_scope)
        zrow.addWidget(self._vsep())
        zrow.addWidget(QtWidgets.QLabel('Scale:'))
        scale_box = QtWidgets.QWidget()
        scale_box_row = QtWidgets.QHBoxLayout(scale_box)
        scale_box_row.setContentsMargins(0, 0, 0, 0)
        scale_box_row.setSpacing(0)
        self._entry_scale = QtWidgets.QLineEdit()
        self._entry_scale.setFixedWidth(38)
        self._entry_scale.setStyleSheet('border: none; background: transparent; padding: 0; margin: 0;')
        self._entry_scale.editingFinished.connect(self._on_scale_changed)
        scale_box_row.addWidget(self._entry_scale)
        scale_box_row.addWidget(QtWidgets.QLabel('\u2033/px'))
        zrow.addWidget(scale_box)
        zrow.addSpacing(12)
        self._btn_fits_header = self._mk_btn('FITS Header', self._show_fits_header)
        self._btn_fits_header.setEnabled(False)
        zrow.addWidget(self._btn_fits_header)
        lcol.addWidget(zbar)

        splitter.addWidget(left)

        # Right panel: sort controls + file list
        right = QtWidgets.QWidget()
        right.setMinimumWidth(160)
        rcol = QtWidgets.QVBoxLayout(right)
        rcol.setSpacing(2)
        rcol.setContentsMargins(0, 0, 0, 0)

        sort_hdr = QtWidgets.QLabel('Sort')
        sort_hdr.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rcol.addWidget(sort_hdr)
        sort_row = QtWidgets.QHBoxLayout()
        sort_row.setSpacing(2)
        sort_row.setContentsMargins(0, 0, 0, 0)
        self._btn_sort_default = self._mk_btn('Default', self._sort_default)
        self._btn_sort_fwhm    = self._mk_btn('FWHM',    self._sort_fwhm)
        self._btn_sort_stars   = self._mk_btn('Stars',   self._sort_stars)
        sort_row.addWidget(self._btn_sort_default)
        sort_row.addWidget(self._btn_sort_fwhm)
        sort_row.addWidget(self._btn_sort_stars)
        rcol.addLayout(sort_row)

        self._lw_files = QtWidgets.QListWidget()
        self._lw_files.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._lw_files.itemClicked.connect(self._on_file_item_clicked)
        self._lw_files.currentRowChanged.connect(self._on_file_row_changed)
        self._lw_files.installEventFilter(self)
        rcol.addWidget(self._lw_files, 1)

        nav_row = QtWidgets.QHBoxLayout()
        nav_row.setContentsMargins(0, 2, 0, 0)
        nav_row.setSpacing(2)
        nav_row.addWidget(self._mk_btn('◀', self.prev_image))
        nav_row.addWidget(self._mk_btn('▶', self.next_image))
        nav_row.addStretch()
        nav_row.addSpacing(20)
        del_btn = self._mk_btn('Delete', self._delete_image)
        del_btn.setObjectName('del_btn')
        nav_row.addWidget(del_btn)
        rcol.addLayout(nav_row)

        self._btn_play = self._mk_btn('Play', self._play_start)
        self._btn_stop = self._mk_btn('Stop', self._play_stop)
        self._btn_stop.setEnabled(False)

        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)   # image panel stretches
        splitter.setStretchFactor(1, 0)   # file list keeps its width
        splitter.setSizes([1060, 220])
        self._splitter = splitter
        splitter.splitterMoved.connect(self._sync_bottom_bar)

        layout.addWidget(splitter, 1)

        # ── bottom controls ──
        # Two halves so Play/Stop align under the file-list column
        bot = QtWidgets.QWidget()
        bot_outer = QtWidgets.QHBoxLayout(bot)
        bot_outer.setContentsMargins(0, 0, 0, 0)
        bot_outer.setSpacing(0)

        bot_left = QtWidgets.QWidget()
        brow = QtWidgets.QHBoxLayout(bot_left)
        brow.setContentsMargins(8, 4, 8, 4)
        bot_outer.addWidget(bot_left, 1)

        bot_right = QtWidgets.QWidget()
        self._bot_right = bot_right
        brow_right = QtWidgets.QHBoxLayout(bot_right)
        brow_right.setContentsMargins(0, 4, 8, 4)
        brow_right.setSpacing(2)
        bot_outer.addWidget(bot_right, 0)

        self._cb_stretch = QtWidgets.QCheckBox('Auto Stretch')
        self._cb_stretch.setChecked(True)
        self._cb_stretch.toggled.connect(self._on_stretch_toggle)
        brow.addWidget(self._cb_stretch)
        brow.addSpacing(8)

        self._btn_inspector = self._mk_btn('Inspector', self._toggle_inspector)
        brow.addWidget(self._btn_inspector)
        self._btn_map = self._mk_btn('FWHM Map', self._toggle_fwhm_map)
        brow.addWidget(self._btn_map)
        self._btn_map_color = self._mk_btn('Color', self._toggle_map_color)
        self._btn_map_color.hide()
        brow.addWidget(self._btn_map_color)
        self._btn_map_contour = self._mk_btn('Contours', self._toggle_map_contour)
        self._btn_map_contour.hide()
        brow.addWidget(self._btn_map_contour)
        self._btn_fwhm_unit = self._mk_btn('\u2033', self._toggle_fwhm_unit)
        self._btn_fwhm_unit.setEnabled(False)
        brow.addWidget(self._btn_fwhm_unit)
        self._lbl_fwhm = QtWidgets.QLabel('FWHM: —')
        brow.addWidget(self._lbl_fwhm)
        brow.addWidget(self._vsep())
        self._lbl_pixel = QtWidgets.QLabel('')
        brow.addWidget(self._lbl_pixel)
        brow.addStretch()

        brow_right.addWidget(self._btn_play)
        brow_right.addWidget(self._btn_stop)
        brow_right.addStretch()

        layout.addWidget(bot)

        self._gview.scene().sigMouseClicked.connect(self._on_scene_clicked)
        QtCore.QTimer.singleShot(0, self._sync_bottom_bar)

    def _on_scene_clicked(self, event):
        if self._data is None:
            return
        pt = self._vb.mapSceneToView(event.scenePos())
        col, row = int(pt.x()), int(pt.y())
        h, w = self._get_image_hw()
        if not (0 <= col < w and 0 <= row < h):
            self._lbl_pixel.setText('')
            return
        d = self._data
        s = self._pixel_scale
        if d.ndim == 2:
            self._lbl_pixel.setText(f'({col}, {row}): {int(round(d[row, col] * s))}')
        elif d.ndim == 3:
            if d.shape[0] in (1, 3) and d.shape[0] < d.shape[1]:  # channel-first
                if d.shape[0] == 1:
                    self._lbl_pixel.setText(f'({col}, {row}): {int(round(d[0, row, col] * s))}')
                else:
                    r, g, b = (int(round(d[0, row, col] * s)),
                               int(round(d[1, row, col] * s)),
                               int(round(d[2, row, col] * s)))
                    self._lbl_pixel.setText(f'({col}, {row}): R={r} G={g} B={b}')
            else:                                                   # channel-last
                if d.shape[2] == 1:
                    self._lbl_pixel.setText(f'({col}, {row}): {int(round(d[row, col, 0] * s))}')
                else:
                    r, g, b = (int(round(d[row, col, 0] * s)),
                               int(round(d[row, col, 1] * s)),
                               int(round(d[row, col, 2] * s)))
                    self._lbl_pixel.setText(f'({col}, {row}): R={r} G={g} B={b}')

    def _apply_style(self):
        self.setStyleSheet(f"""
            QWidget           {{ background: {BG}; color: {FG};
                                 font-family: Consolas; font-size: 10pt; }}
            QPushButton       {{ background: {BTN}; border: none;
                                 padding: 4px 10px; color: {FG}; }}
            QPushButton:hover    {{ background: #1a4a80; }}
            QPushButton#sort_active {{ background: #1a4a80; color: {FG_ACCENT}; }}
            QCheckBox         {{ spacing: 6px; }}
            QCheckBox::indicator          {{ width: 14px; height: 14px; }}
            QCheckBox::indicator:unchecked {{ background: {BTN}; border: 1px solid {FG_DIM}; }}
            QCheckBox::indicator:checked   {{ background: {FG_ACCENT}; border: 1px solid {FG_ACCENT}; }}
            QLineEdit         {{ background: {BTN}; border: none;
                                 padding: 2px 4px; color: {FG}; }}
            QLabel            {{ color: {FG}; }}
            #del_btn          {{ background: {BTN_DEL}; }}
            #del_btn:hover    {{ background: #c0392b; }}
            QListWidget       {{ background: {TOOLBAR}; border: none;
                                 font-size: 9pt; outline: 0; }}
            QListWidget::item {{ padding: 3px 6px; color: {FG_DIM}; }}
            QListWidget::item:selected {{
                background: {BTN}; color: {FG_ACCENT};
            }}
            QListWidget::item:hover:!selected {{ background: #1e2a4a; }}
            QSplitter::handle {{ background: {TOOLBAR}; width: 3px; }}
        """)
        self._gview.setBackground(BG)

    def _setup_shortcuts(self):
        pairs = [
            (QtCore.Qt.Key.Key_Left,   self.prev_image),
            (QtCore.Qt.Key.Key_Right,  self.next_image),
            (QtCore.Qt.Key.Key_Delete, self._delete_image),
            (QtCore.Qt.Key.Key_Plus,   lambda: self._zoom_by(1 / 1.5)),
            (QtCore.Qt.Key.Key_Equal,  lambda: self._zoom_by(1 / 1.5)),
            (QtCore.Qt.Key.Key_Minus,  lambda: self._zoom_by(1.5)),
            (QtCore.Qt.Key.Key_F,      self._zoom_fit),
        ]
        for key, fn in pairs:
            QtGui.QShortcut(QtGui.QKeySequence(key), self, fn)

    # ------------------------------------------------------------------
    # Zoom  (PyQtGraph ViewBox handles the actual pan/zoom via GPU;
    #        these methods just set the viewport rectangle.)
    # ------------------------------------------------------------------

    def _get_image_hw(self) -> tuple[int, int]:
        d = self._data
        if d is None:
            return 0, 0
        if d.ndim == 2:
            return d.shape[0], d.shape[1]
        if d.shape[0] in (1, 3) and d.shape[0] < d.shape[1]:
            return d.shape[1], d.shape[2]
        return d.shape[0], d.shape[1]

    def _zoom_by(self, factor: float):
        if self._data is None:
            return
        vr = self._vb.viewRect()
        cx = vr.x() + vr.width() / 2
        cy = vr.y() + vr.height() / 2
        hw = vr.width()  * factor / 2
        hh = vr.height() * factor / 2
        self._vb.setRange(xRange=(cx - hw, cx + hw),
                          yRange=(cy - hh, cy + hh), padding=0)

    def _zoom_fit(self):
        if self._data is None:
            return
        h, w = self._get_image_hw()
        self._vb.setRange(xRange=(0, w), yRange=(0, h), padding=0.01)

    def _zoom_to_ratio(self, img_px_per_screen_px: float):
        """img_px_per_screen_px: 0.5=200%, 1.0=100%, 2.0=50%"""
        if self._data is None:
            return
        vr  = self._vb.viewRect()
        geo = self._gview.geometry()
        cx  = vr.x() + vr.width()  / 2
        cy  = vr.y() + vr.height() / 2
        hw  = geo.width()  * img_px_per_screen_px / 2
        hh  = geo.height() * img_px_per_screen_px / 2
        self._vb.setRange(xRange=(cx - hw, cx + hw),
                          yRange=(cy - hh, cy + hh), padding=0)

    def _update_zoom_label(self):
        if self._data is None:
            return
        vr  = self._vb.viewRect()
        geo = self._gview.geometry()
        if vr.width() > 0:
            self._lbl_zoom.setText(f'{geo.width() / vr.width() * 100:.0f}%')

    # ------------------------------------------------------------------
    # Directory / file management
    # ------------------------------------------------------------------

    def _choose_dir(self):
        start = _load_last_dir() or ''
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select image directory', start)
        if d:
            self._open_dir(d)

    def _open_dir(self, path: str, select_file: Path | None = None):
        self._directory = Path(path)
        self._files = sorted(
            f for f in self._directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
        )
        if not self._files:
            QtWidgets.QMessageBox.information(
                self, 'No images', f'No FITS or XISF files found in:\n{path}')
            return
        _save_last_dir(path)
        self._nav_gen += 1   # discard any queued navigation loads from previous directory
        self._playing = False
        self._btn_play.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._index = 0
        self._nav_index = 0
        if select_file is not None:
            try:
                self._index = self._files.index(select_file)
                self._nav_index = self._index
            except ValueError:
                pass
        self._files_default = list(self._files)
        self._active_sort = None          # prevent reverse-on-open
        self._sort_reversed = False
        self._apply_sort('default', list(self._files), navigate=False)
        self._load_current(reset_view=True)

    def _populate_file_list(self):
        """Rebuild the file list widget from self._files."""
        self._lw_files.blockSignals(True)
        self._lw_files.clear()
        for f in self._files:
            self._lw_files.addItem(f.name)
        self._lw_files.setCurrentRow(self._index)
        self._lw_files.blockSignals(False)

        # Resize the right panel to fit the longest filename.
        # 24px covers item padding (6px × 2) + list widget margins.
        if self._files:
            fm = self._lw_files.fontMetrics()
            max_text_w = max(fm.horizontalAdvance(f.name) for f in self._files)
            needed = max(160, max_text_w + 24)
            sizes = self._splitter.sizes()
            total = sum(sizes)
            if total > needed:
                self._splitter.setSizes([total - needed, needed])

        # Scroll horizontally to show the end of filenames (where FWHM values appear)
        QtCore.QTimer.singleShot(
            0, lambda: self._lw_files.horizontalScrollBar().setValue(
                self._lw_files.horizontalScrollBar().maximum()
            )
        )

    def _apply_sort(self, key: str, sorted_files: list, navigate: bool = True):
        """Apply a sort order, highlight the active button, and go to the top."""
        # Re-pressing the active sort toggles direction
        if key == self._active_sort:
            sorted_files = list(reversed(self._files))
            self._sort_reversed = not self._sort_reversed
        else:
            self._sort_reversed = False
        # When called from a sort button (navigate=True) go to top;
        # when called on startup (navigate=False) preserve the selected file.
        current_file = (self._files[self._index] if (not navigate and self._files) else None)
        self._files = sorted_files
        self._active_sort = key
        # Update button labels and highlight
        _labels = {'default': 'Default', 'fwhm': 'FWHM', 'stars': 'Stars'}
        for btn, name in ((self._btn_sort_default, 'default'),
                          (self._btn_sort_fwhm,    'fwhm'),
                          (self._btn_sort_stars,   'stars')):
            if name == key:
                prefix = '\u2212' if self._sort_reversed else '+'
                btn.setText(f'{prefix}{_labels[name]}')
                btn.setObjectName('sort_active')
            else:
                btn.setText(_labels[name])
                btn.setObjectName('')
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        # Restore selection to the same file; fall back to top if not found
        if current_file is not None and current_file in self._files:
            self._index = self._files.index(current_file)
        else:
            self._index = 0
        self._nav_index = self._index
        self._populate_file_list()
        if self._index == 0:
            self._lw_files.scrollToTop()
        if navigate and self._files:
            self._load_current(reset_view=False)

    def _sort_default(self):
        self._apply_sort('default', list(self._files_default))

    def _sort_fwhm(self):
        _PAT = re.compile(r'FWHM([\d.]+)_', re.IGNORECASE)
        matched = [f for f in self._files_default if _PAT.search(f.name)]
        if not matched:
            sample = self._files[0].name if self._files else '(no files)'
            QtWidgets.QMessageBox.information(
                self, 'No FWHM values found',
                f'No filenames contained FWHM<value>_ .\n\n'
                f'Example filename checked:\n  {sample}\n\n'
                f'Expected format:  …FWHM2.35_…'
            )
            return
        def fwhm_key(p: Path) -> float:
            m = _PAT.search(p.name)
            return float(m.group(1)) if m else float('inf')
        self._apply_sort('fwhm', sorted(self._files_default, key=fwhm_key))

    def _sort_stars(self):
        _PAT = re.compile(r'stars(\d+)_', re.IGNORECASE)
        matched = [f for f in self._files_default if _PAT.search(f.name)]
        if not matched:
            sample = self._files[0].name if self._files else '(no files)'
            QtWidgets.QMessageBox.information(
                self, 'No star-count values found',
                f'No filenames contained stars<integer>_ .\n\n'
                f'Example filename checked:\n  {sample}\n\n'
                f'Expected format:  …stars42_…'
            )
            return
        def stars_key(p: Path) -> int:
            m = _PAT.search(p.name)
            return int(m.group(1)) if m else 999999
        self._apply_sort('stars', sorted(self._files_default, key=stars_key))

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Intercept up/down arrow keys in the file list so they use the nav queue."""
        if obj is self._lw_files and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key.Key_Down:
                self.next_image()
                return True   # consume — prevents currentRowChanged from firing
            if key == QtCore.Qt.Key.Key_Up:
                self.prev_image()
                return True
        return super().eventFilter(obj, event)

    def _on_file_row_changed(self, row: int):
        if row >= 0 and row != self._index:
            self._index = row
            self._nav_index = row
            self._nav_gen += 1   # discard any queued navigation loads
            self._load_current()

    def _on_file_item_clicked(self, item: QtWidgets.QListWidgetItem):
        self._on_file_row_changed(self._lw_files.row(item))

    def _load_current(self, reset_view: bool = False):
        if not self._files:
            return
        path = self._files[self._index]
        self._lbl_file.setText(str(path))
        self._lbl_count.setText(f'{self._index + 1} / {len(self._files)}')
        self._lw_files.blockSignals(True)
        self._lw_files.setCurrentRow(self._index)
        self._lw_files.blockSignals(False)
        self._lbl_fwhm.setText('FWHM: —')
        self._scatter.clear()
        for item in self._fwhm_label_items:
            self._vb.removeItem(item)
        self._fwhm_label_items.clear()
        self._clear_fwhm_map()
        self._last_map_fwhm = None

        with self._cache_lock:
            cached = self._cache.get(path)
            if cached is not None:
                self._cache.move_to_end(path)   # mark as recently used

        precomp = None
        if cached is not None:
            self._data, self._plate_scale, self._scope_info, precomp = cached
            self._entry_scale.setText(
                f'{self._plate_scale:.2f}' if self._plate_scale else '')
        else:
            try:
                ext = path.suffix.lower()
                if ext in ('.fits', '.fit', '.fts'):
                    self._data, hdr = load_fits(str(path))
                    self._plate_scale = extract_plate_scale(hdr)
                    self._scope_info = extract_telescope_info(hdr)
                    self._entry_scale.setText(
                        f'{self._plate_scale:.2f}' if self._plate_scale else '')
                elif ext == '.xisf':
                    self._data, xisf_kw = load_xisf(str(path))
                    self._plate_scale = extract_plate_scale_xisf(xisf_kw)
                    self._scope_info = extract_telescope_info(xisf_kw)
                    self._entry_scale.setText(
                        f'{self._plate_scale:.2f}' if self._plate_scale else '')
                else:
                    QtWidgets.QMessageBox.critical(
                        self, 'Unsupported', f'Unknown extension: {ext}')
                    return
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, 'Load error', f'Could not load image:\n{exc}')
                self._data = None
                return
            self._cache_put(path, self._data, self._plate_scale, self._scope_info)
        self._update_scope_label()
        self._btn_fits_header.setEnabled(True)

        # Fast path: prefetch worker already rendered the display image
        if precomp is not None and precomp['auto'] == self._cb_stretch.isChecked():
            self._stretch_params = precomp['stretch_params']
            self._pixel_scale = precomp['pixel_scale']
            self._fast_display(precomp, reset_view)
            self._schedule_prefetch()
            return

        self._stretch_params = compute_stretch_params(
            self._data, self._cb_stretch.isChecked()
        )
        # Normalized float data (XISF [0,1]) — scale clicks to 16-bit range
        d = self._data
        valid_max = float(d[np.isfinite(d)].max()) if d is not None else 1.0
        self._pixel_scale = 65535.0 if valid_max <= 1.0 else 1.0
        self._refresh_display(reset_view=reset_view)
        self._schedule_prefetch()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-fetch cache
    # ------------------------------------------------------------------

    def _cache_put(self, path: Path, data: np.ndarray, scale: float | None,
                   scope_info: tuple = (None, None, None), precomp: dict | None = None):
        """Insert into the LRU cache, evicting the oldest entry if full."""
        with self._cache_lock:
            self._cache[path] = (data, scale, scope_info, precomp)
            self._cache.move_to_end(path)
            while len(self._cache) > 5:
                self._cache.popitem(last=False)

    def _schedule_prefetch(self):
        """Kick off background loads for the immediate neighbours (±2)."""
        for delta in (1, -1, 2, -2):
            idx = self._index + delta
            if not (0 <= idx < len(self._files)):
                continue
            path = self._files[idx]
            with self._cache_lock:
                if path in self._cache:
                    continue
            with self._prefetch_lock:
                if path in self._prefetching:
                    continue
                self._prefetching.add(path)
            threading.Thread(
                target=self._prefetch_worker, args=(path,), daemon=True
            ).start()

    def _prefetch_worker(self, path: Path):
        try:
            ext = path.suffix.lower()
            if ext in ('.fits', '.fit', '.fts'):
                data, hdr = load_fits(str(path))
                scale = extract_plate_scale(hdr)
                scope_info = extract_telescope_info(hdr)
            elif ext == '.xisf':
                data, xisf_kw = load_xisf(str(path))
                scale = extract_plate_scale_xisf(xisf_kw)
                scope_info = extract_telescope_info(xisf_kw)
            else:
                return
            precomp = _make_precomp(data, auto=True)
            self._cache_put(path, data, scale, scope_info, precomp)
        except Exception:
            pass
        finally:
            with self._prefetch_lock:
                self._prefetching.discard(path)

    def _overview(self) -> np.ndarray:
        """Downsample raw data to ≤4096 px on the longest axis for GPU upload."""
        h, w = self._get_image_hw()
        stride = max(1, max(w // 4096, h // 4096))
        d = self._data
        if d.ndim == 2:
            return d[::stride, ::stride]
        if d.shape[0] in (1, 3) and d.shape[0] < d.shape[1]:
            return d[:, ::stride, ::stride]
        return d[::stride, ::stride]

    def _fast_display(self, precomp: dict, reset_view: bool = False):
        """GPU upload only — all CPU work was done by the prefetch thread."""
        if precomp['is_gray']:
            self._img_item.setLookupTable(GRAY_LUT, update=False)
            self._img_item.setImage(precomp['img_u8'], autoLevels=False, levels=(0, 255))
        else:
            self._img_item.setLookupTable(None, update=False)
            self._img_item.setImage(precomp['img_u8'], autoLevels=False)
        self._img_item.setRect(QtCore.QRectF(0, 0, precomp['w'], precomp['h']))
        if reset_view:
            self._zoom_fit()
        if self._inspector_active:
            self._inspector_widget.load(self._data, self._stretch_params)
            self._inspector_widget.clear_stars()
            self._inspector_fwhm_result = {}

    def _refresh_display(self, reset_view: bool = False):
        """
        Stretch the overview and upload it to the GPU once.
        After this, all pan/zoom is handled by PyQtGraph with zero CPU involvement.
        """
        if self._data is None:
            return

        display, is_gray = prepare_display(
            self._overview(), self._cb_stretch.isChecked(), self._stretch_params
        )

        # uint8 halves GPU memory vs float32 with no visible quality difference
        img_u8 = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)

        if is_gray:
            self._img_item.setLookupTable(GRAY_LUT, update=False)
            self._img_item.setImage(img_u8, autoLevels=False, levels=(0, 255))
        else:
            self._img_item.setLookupTable(None, update=False)
            self._img_item.setImage(img_u8, autoLevels=False)

        # Scale the image item to fill original image coordinates so that
        # zoom/pan work in true image pixels regardless of overview stride.
        h, w = self._get_image_hw()
        self._img_item.setRect(QtCore.QRectF(0, 0, w, h))

        if reset_view:
            self._zoom_fit()
        if self._inspector_active:
            self._inspector_widget.load(self._data, self._stretch_params)
            self._inspector_widget.clear_stars()
            self._inspector_fwhm_result = {}

    # ------------------------------------------------------------------
    # FWHM calculation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # FWHM map overlay
    # ------------------------------------------------------------------

    def _toggle_fwhm_map(self):
        if self._inspector_active:
            self._compute_inspector_fwhm()
            return
        if self._map_visible:
            self._clear_fwhm_map()
            return
        if self._map_computing:
            return
        if self._data is None:
            QtWidgets.QMessageBox.information(self, 'No image', 'Load an image first.')
            return
        if not HAS_PHOTUTILS:
            QtWidgets.QMessageBox.critical(self, 'Missing library',
                'photutils is required.\nRun:  pip install photutils')
            return
        if not HAS_SCIPY:
            QtWidgets.QMessageBox.critical(self, 'Missing library',
                'scipy is required.\nRun:  pip install scipy')
            return
        self._map_computing = True
        self._btn_map.setEnabled(False)
        self._btn_map.setText('Computing…')
        threading.Thread(target=self._fwhm_map_worker, daemon=True).start()

    def _fwhm_map_worker(self):
        try:
            data = self._data
            if data.ndim == 3:
                work = (data[0] if data.shape[0] in (1, 3) and data.shape[0] < data.shape[1]
                        else data[:, :, 0]).astype(np.float64)
            else:
                work = data.astype(np.float64)
            work = np.nan_to_num(work)
            image_max = float(work.max())
            sat_level = _bit_depth_max(image_max)

            # Subsample for detection: 4× faster stats + convolution with no
            # meaningful loss — stars are still easily found at half resolution.
            det_stride = 2
            detect = work[::det_stride, ::det_stride]
            _, median, std = sigma_clipped_stats(detect, sigma=3.0)
            daofind = DAOStarFinder(fwhm=max(2.0, 4.0 / det_stride),
                                    threshold=5.0 * std)
            sources = daofind(detect - median)
            # Scale centroid coordinates back to full-resolution pixels
            if sources is not None and len(sources) > 0:
                sources['xcentroid'] = np.array(sources['xcentroid']) * det_stride
                sources['ycentroid'] = np.array(sources['ycentroid']) * det_stride

            if sources is None or len(sources) == 0:
                self._sig_map_msg.emit('no stars found')
                return

            sources.sort('peak', reverse=True)

            # Record positions of saturated stars so we can exclude neighbours.
            est_raw_all = np.array(sources['peak']) + median
            sat_mask = est_raw_all >= sat_level * 0.75
            sat_xs = np.array(sources['xcentroid'])[sat_mask]
            sat_ys = np.array(sources['ycentroid'])[sat_mask]

            # Pre-filter saturated sources before grid selection so each cell
            # gets the brightest *valid* star, not just the brightest overall.
            est_raw = est_raw_all
            sources = sources[est_raw < sat_level * 0.75]
            # Roundness filter: reject elongated sources (double stars, blends)
            round_ok = ((np.abs(sources['roundness1']) < 0.5) &
                        (np.abs(sources['roundness2']) < 0.5))
            sources = sources[round_ok]
            if len(sources) == 0:
                self._sig_map_msg.emit('no unsaturated stars found')
                return

            h, w = work.shape
            xs_all = np.array(sources['xcentroid'])
            ys_all = np.array(sources['ycentroid'])

            # Exclude candidates too close to a saturated star — their PSF
            # is contaminated by scattered light and will skew the FWHM map.
            if len(sat_xs) > 0:
                excl_radius = max(150.0, min(w, h) * 0.03)
                too_close = np.zeros(len(sources), dtype=bool)
                for sx, sy in zip(sat_xs, sat_ys):
                    d2 = (xs_all - sx) ** 2 + (ys_all - sy) ** 2
                    too_close |= d2 < excl_radius ** 2
                sources = sources[~too_close]
                xs_all  = xs_all[~too_close]
                ys_all  = ys_all[~too_close]

            # For sparse fields use every detected star.
            # For dense fields grid-sample (7×7 cells) to bound computation.
            N_GRID = 7
            if len(sources) <= N_GRID * N_GRID:
                candidates = sources
            else:
                cell_w = w / N_GRID
                cell_h = h / N_GRID
                grid_selected = []
                for gr in range(N_GRID):
                    for gc in range(N_GRID):
                        x0, x1c = gc * cell_w, (gc + 1) * cell_w
                        y0, y1c = gr * cell_h, (gr + 1) * cell_h
                        mask = ((xs_all >= x0) & (xs_all < x1c) &
                                (ys_all >= y0) & (ys_all < y1c))
                        cell = sources[mask]
                        if len(cell) > 0:
                            for row in cell[:3]:   # top 3 so saturated brightest doesn't void the cell
                                grid_selected.append(row)
                candidates = grid_selected

            if not candidates:
                self._sig_map_msg.emit('no stars found')
                return

            half = 30
            fwhm_pts = []   # (x, y, fwhm)
            for src in candidates:
                x, y = int(src['xcentroid']), int(src['ycentroid'])
                x1, x2 = max(0, x - half), min(w, x + half)
                y1, y2 = max(0, y - half), min(h, y + half)
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                cutout = work[y1:y2, x1:x2].copy()
                peak = cutout.max()
                if peak > sat_level * 0.75:
                    continue
                if (cutout >= peak * 0.995).sum() > 4:
                    continue
                # Reject hot pixel pairs: a real star has many pixels above
                # quarter-max; a 1–2 pixel defect has almost none.
                bg = float(np.percentile(cutout, 20))
                amp = peak - bg
                if amp <= 0 or int((cutout > bg + amp * 0.25).sum()) < 5:
                    continue
                fwhm = fit_star_fwhm(cutout, cx_hint=x - x1, cy_hint=y - y1)
                if fwhm is not None and 1.0 < fwhm < 30.0:
                    fwhm_pts.append((float(src['xcentroid']),
                                     float(src['ycentroid']), fwhm))

            if len(fwhm_pts) < 3:
                self._sig_map_msg.emit('not enough stars for map')
                return

            # Local spatial outlier rejection: compare each point to its k
            # nearest neighbours rather than the global median.  This keeps
            # gradual FWHM gradients (field curvature, coma) while removing
            # spotty outliers (double stars, cosmic rays, blended sources)
            # whose FWHM is inconsistent with their immediate surroundings.
            if len(fwhm_pts) >= 6:
                from scipy.spatial import cKDTree
                pts_xy  = np.array([(p[0], p[1]) for p in fwhm_pts])
                vals_np = np.array([p[2]          for p in fwhm_pts])
                k       = min(6, len(fwhm_pts) - 1)
                tree    = cKDTree(pts_xy)
                _, idxs = tree.query(pts_xy, k=k + 1)   # col-0 is self
                local_good = np.ones(len(fwhm_pts), dtype=bool)
                for i in range(len(fwhm_pts)):
                    nbr_vals    = vals_np[idxs[i, 1:]]
                    local_med   = np.median(nbr_vals)
                    local_mad   = np.median(np.abs(nbr_vals - local_med))
                    local_sigma = local_mad * 1.4826
                    if abs(vals_np[i] - local_med) > 3.0 * max(local_sigma, 0.3):
                        local_good[i] = False
                fwhm_pts = [p for p, g in zip(fwhm_pts, local_good) if g]

            if len(fwhm_pts) < 3:
                self._sig_map_msg.emit('not enough stars after outlier rejection')
                return

            pts  = np.array([(p[0], p[1]) for p in fwhm_pts])
            vals = np.array([p[2]          for p in fwhm_pts])

            # Gaussian kernel regression (Nadaraya-Watson): each grid point is
            # a distance-weighted average of all measured FWHM values.
            # Bandwidth adapts to star density — roughly 1.5× the typical
            # inter-star spacing — so every grid point draws on several
            # neighbouring measurements rather than being dominated by one.
            from scipy.ndimage import gaussian_filter
            grid_w = min(w, 300)
            grid_h = min(h, 300)
            gx = np.linspace(0, w, grid_w)
            gy = np.linspace(0, h, grid_h)
            GX, GY = np.meshgrid(gx, gy)
            grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

            bandwidth = float(np.sqrt(w * h / max(len(fwhm_pts), 1))) * 1.5
            diff_x = grid_pts[:, 0:1] - pts[:, 0]   # (n_grid, n_data)
            diff_y = grid_pts[:, 1:2] - pts[:, 1]
            weights = np.exp(-0.5 * (diff_x ** 2 + diff_y ** 2) / bandwidth ** 2)
            grid_z = (weights @ vals) / np.maximum(weights.sum(axis=1), 1e-10)
            grid_z = grid_z.reshape(grid_h, grid_w)

            # Light pixel-level blur to remove grid-sampling artefacts
            grid_z = gaussian_filter(grid_z, sigma=3.0)

            fwhm_min = float(np.percentile(vals, 5))
            fwhm_max = float(np.percentile(vals, 95))
            if fwhm_max - fwhm_min < 0.1:
                fwhm_min = fwhm_max - 0.5

            norm = np.clip((grid_z - fwhm_min) / (fwhm_max - fwhm_min), 0.0, 1.0)

            # Green (good/small) → yellow → red (bad/large)
            r = np.clip(norm * 2.0 * 255, 0, 255).astype(np.uint8)
            g = np.clip((1.0 - norm) * 2.0 * 255, 0, 255).astype(np.uint8)
            b = np.zeros(grid_z.shape, dtype=np.uint8)
            a = np.full(grid_z.shape, 70, dtype=np.uint8)
            rgba = np.stack([r, g, b, a], axis=-1)

            # Contour lines via matplotlib (library already required)
            contour_paths = []
            try:
                from matplotlib.figure import Figure
                import warnings
                min_spacing_px = (0.2 / self._plate_scale
                                  if self._plate_scale else 0.2)
                range_px = fwhm_max - fwhm_min
                n_levels = max(1, min(5, int(range_px / min_spacing_px)))
                levels = np.linspace(fwhm_min, fwhm_max, n_levels + 2)[1:-1]
                fig = Figure()
                ax  = fig.add_subplot(111)
                cs  = ax.contour(GX, GY, grid_z, levels=levels)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for lv, level_segs in zip(levels, cs.allsegs):
                        for seg in level_segs:
                            if len(seg) > 1:
                                contour_paths.append((float(lv), np.array(seg)))
            except Exception:
                pass   # overlay still shown without contours

            avg_fwhm = float(np.median(grid_z))
            self._sig_map_done.emit(rgba, contour_paths, fwhm_pts, avg_fwhm)

        except Exception as exc:
            self._sig_map_msg.emit(f'map error — {exc}')

    def _show_fwhm_map(self, rgba, contour_paths, fwhm_pts, avg_fwhm):
        h, w = self._get_image_hw()
        self._overlay_item.setImage(rgba, autoLevels=False)
        self._overlay_item.setRect(QtCore.QRectF(0, 0, w, h))

        for item in self._contour_items:
            self._vb.removeItem(item)
        self._contour_items.clear()
        for item in self._contour_label_items:
            self._vb.removeItem(item)
        self._contour_label_items.clear()

        # Store data for unit toggle redraws
        from collections import defaultdict
        level_segs: dict = defaultdict(list)
        for lv, seg in contour_paths:
            level_segs[lv].append(seg)
        self._contour_level_segs = dict(level_segs)
        self._fwhm_pts = list(fwhm_pts)

        # Enable unit toggle only when plate scale is available
        has_ps = bool(self._plate_scale)
        self._btn_fwhm_unit.setEnabled(has_ps)
        if not has_ps:
            self._fwhm_unit = 'px'
        # Update button to show current unit
        self._btn_fwhm_unit.setText('\u2033' if self._fwhm_unit == 'arcsec' else 'px')

        for lv, segs in level_segs.items():
            for seg in segs:
                curve = pg.PlotCurveItem(
                    x=seg[:, 0], y=seg[:, 1],
                    pen=pg.mkPen(color=(255, 255, 255, 160), width=1.0),
                )
                curve.setZValue(2)
                curve.setVisible(self._map_contour_visible)
                self._vb.addItem(curve)
                self._contour_items.append(curve)

        self._draw_fwhm_labels()

        self._scatter.setData(
            x=[p[0] for p in fwhm_pts],
            y=[p[1] for p in fwhm_pts],
        )

        h, w = self._get_image_hw()
        cx0, cx1 = w * 0.25, w * 0.75
        cy0, cy1 = h * 0.25, h * 0.75
        ctr_pts = [p[2] for p in fwhm_pts if cx0 <= p[0] <= cx1 and cy0 <= p[1] <= cy1]
        ctr_fwhm = float(np.median(ctr_pts)) if ctr_pts else avg_fwhm
        self._last_map_fwhm = (avg_fwhm, ctr_fwhm)
        self._refresh_map_fwhm_label()

        # Dashed box showing the center 50% region used for center FWHM
        dash_pen = pg.mkPen(color=(255, 255, 255, 90), width=1.5,
                            style=QtCore.Qt.PenStyle.DashLine)
        box = pg.PlotCurveItem(
            x=[cx0, cx1, cx1, cx0, cx0],
            y=[cy0, cy0, cy1, cy1, cy0],
            pen=dash_pen,
        )
        box.setZValue(3)
        box.setVisible(self._map_contour_visible)
        self._vb.addItem(box)
        self._contour_items.append(box)

        self._map_visible   = True
        self._map_computing = False
        self._btn_map.setEnabled(True)
        self._btn_map.setText('Hide FWHM Map')
        self._map_color_visible   = True
        self._map_contour_visible = True
        self._btn_map_color.setText('No Color')
        self._btn_map_color.show()
        self._btn_map_contour.setText('No Contours')
        self._btn_map_contour.show()
        self._overlay_item.show()

    # ------------------------------------------------------------------
    # Corner inspector
    # ------------------------------------------------------------------

    def _toggle_inspector(self):
        self._inspector_active = not self._inspector_active
        if self._inspector_active:
            self._clear_fwhm_map()
            self._view_stack.setCurrentIndex(1)
            self._btn_inspector.setText('Close Inspector')
            self._btn_map.setText('FWHM')
            self._inspector_fwhm_result = {}
            self._btn_fwhm_unit.setEnabled(bool(self._plate_scale))
            if self._data is not None:
                self._inspector_widget.load(self._data, self._stretch_params)
        else:
            self._view_stack.setCurrentIndex(0)
            self._btn_inspector.setText('Inspector')
            self._btn_map.setText('FWHM Map')
            self._inspector_widget.clear_stars()
            self._inspector_fwhm_result = {}
            self._btn_fwhm_unit.setEnabled(False)

    def _compute_inspector_fwhm(self):
        # Toggle off if labels already showing
        if self._inspector_fwhm_result:
            self._inspector_fwhm_result = {}
            self._inspector_widget.clear_stars()
            self._btn_map.setText('FWHM')
            return
        if self._inspector_computing:
            return
        if self._data is None:
            QtWidgets.QMessageBox.information(self, 'No image', 'Load an image first.')
            return
        if not HAS_PHOTUTILS or not HAS_SCIPY:
            QtWidgets.QMessageBox.critical(self, 'Missing library',
                'photutils and scipy are required.')
            return
        regions = {}
        for row, col, _ in _INSPECTOR_CELLS:
            r = self._inspector_widget.crop_region(row, col)
            if r is not None:
                regions[(row, col)] = r
        if not regions:
            return
        self._inspector_regions = regions   # saved for label redraws
        self._inspector_computing = True
        self._btn_map.setEnabled(False)
        self._btn_map.setText('Computing…')
        threading.Thread(
            target=self._inspector_fwhm_worker, args=(regions,), daemon=True
        ).start()

    def _inspector_fwhm_worker(self, regions: dict):
        try:
            data = self._data
            if data.ndim == 3:
                work = (data[0] if data.shape[0] in (1, 3) and data.shape[0] < data.shape[1]
                        else data[:, :, 0]).astype(np.float64)
            else:
                work = data.astype(np.float64)
            work = np.nan_to_num(work)
            image_max = float(work.max())
            sat_level = _bit_depth_max(image_max)

            det_stride = 2
            detect = work[::det_stride, ::det_stride]
            _, median, std = sigma_clipped_stats(detect, sigma=3.0)
            daofind = DAOStarFinder(fwhm=max(2.0, 4.0 / det_stride), threshold=5.0 * std)
            sources = daofind(detect - median)
            if sources is not None and len(sources) > 0:
                sources['xcentroid'] = np.array(sources['xcentroid']) * det_stride
                sources['ycentroid'] = np.array(sources['ycentroid']) * det_stride

            if sources is None or len(sources) == 0:
                self._sig_inspector_done.emit({})
                return

            est_raw_all = np.array(sources['peak']) + median
            sat_mask = est_raw_all >= sat_level * 0.75
            sat_xs = np.array(sources['xcentroid'])[sat_mask]
            sat_ys = np.array(sources['ycentroid'])[sat_mask]

            sources = sources[est_raw_all < sat_level * 0.75]
            round_ok = ((np.abs(sources['roundness1']) < 0.5) &
                        (np.abs(sources['roundness2']) < 0.5))
            sources = sources[round_ok]

            h, w = work.shape
            xs_all = np.array(sources['xcentroid'])
            ys_all = np.array(sources['ycentroid'])

            if len(sat_xs) > 0:
                excl_radius = max(150.0, min(w, h) * 0.03)
                too_close = np.zeros(len(sources), dtype=bool)
                for sx, sy in zip(sat_xs, sat_ys):
                    d2 = (xs_all - sx) ** 2 + (ys_all - sy) ** 2
                    too_close |= d2 < excl_radius ** 2
                sources = sources[~too_close]
                xs_all = xs_all[~too_close]
                ys_all = ys_all[~too_close]

            # Select top-N candidates per cell region only — no need to fit
            # every star in the full image (that's why the map uses a grid).
            MAX_PER_CELL = 25
            cell_candidates = {}   # (row,col) -> list of source rows
            for (row, col), (rx0, ry0, rx1, ry1) in regions.items():
                in_cell = ((xs_all >= rx0) & (xs_all < rx1) &
                           (ys_all >= ry0) & (ys_all < ry1))
                cell_src = sources[in_cell]
                if len(cell_src) > 0:
                    cell_src.sort('peak', reverse=True)
                    cell_candidates[(row, col)] = [row for row in cell_src[:MAX_PER_CELL]]

            # Fit FWHM only for the selected per-cell candidates
            half = 30
            star_results = []
            seen = set()   # avoid fitting the same star twice (overlapping regions)
            for cell_rows in cell_candidates.values():
                for src in cell_rows:
                    x, y = int(src['xcentroid']), int(src['ycentroid'])
                    if (x, y) in seen:
                        continue
                    seen.add((x, y))
                    x1, x2 = max(0, x - half), min(w, x + half)
                    y1, y2 = max(0, y - half), min(h, y + half)
                    if (x2 - x1) < 8 or (y2 - y1) < 8:
                        continue
                    cutout = work[y1:y2, x1:x2].copy()
                    peak = cutout.max()
                    if peak > sat_level * 0.75:
                        continue
                    if (cutout >= peak * 0.995).sum() > 4:
                        continue
                    bg = float(np.percentile(cutout, 20))
                    amp = peak - bg
                    if amp <= 0 or int((cutout > bg + amp * 0.25).sum()) < 5:
                        continue
                    fwhm = fit_star_fwhm(cutout, cx_hint=x - x1, cy_hint=y - y1)
                    if fwhm is not None and 1.0 < fwhm < 30.0:
                        star_results.append((float(src['xcentroid']),
                                             float(src['ycentroid']), fwhm))

            # Assign fitted stars back to their cells
            result = {}
            for (row, col), (rx0, ry0, rx1, ry1) in regions.items():
                cell_stars = [(sx, sy, fwhm) for (sx, sy, fwhm) in star_results
                              if rx0 <= sx < rx1 and ry0 <= sy < ry1]
                result[(row, col)] = cell_stars   # list of (img_x, img_y, fwhm_px)

            self._sig_inspector_done.emit(result)
        except Exception:
            self._sig_inspector_done.emit({})

    def _show_inspector_fwhm(self, result: dict):
        self._inspector_computing = False
        self._btn_map.setEnabled(True)
        self._btn_map.setText('Clear FWHM')
        # Store raw per-star data {(row,col): [(img_x, img_y, fwhm_px), ...]}
        self._inspector_fwhm_result = result
        # Also store regions so _redraw can get crop origins
        self._inspector_regions_cache = getattr(self, '_inspector_regions', {})
        self._redraw_inspector_fwhm_labels()

    def _on_inspector_cells_rendered(self):
        """Called by _InspectorWidget after every repaint (including resize)."""
        if self._inspector_active and self._inspector_fwhm_result:
            self._redraw_inspector_fwhm_labels()

    def _redraw_inspector_fwhm_labels(self):
        """Re-draw star circles and labels on each inspector cell using current unit."""
        self._inspector_widget.clear_stars()
        if not self._inspector_fwhm_result:
            return
        ps = self._plate_scale
        use_arcsec = self._fwhm_unit == 'arcsec' and bool(ps)
        for (row, col), stars in self._inspector_fwhm_result.items():
            # Always use current cell dimensions so labels stay aligned after resize
            region = self._inspector_widget.crop_region(row, col)
            if region is None:
                continue
            rx0, ry0, rx1, ry1 = region
            act_w, act_h = rx1 - rx0, ry1 - ry0
            labelled = []
            for img_x, img_y, fwhm_px in stars:
                txt = (f'{fwhm_px * ps:.2f}\u2033' if use_arcsec
                       else f'{fwhm_px:.2f}px')
                labelled.append((img_x, img_y, txt))
            self._inspector_widget.set_cell_stars(
                row, col, labelled, rx0, ry0, act_w, act_h
            )

    def _draw_fwhm_labels(self):
        """Draw contour text labels and per-star labels using current _fwhm_unit."""
        for item in self._contour_label_items:
            self._vb.removeItem(item)
        self._contour_label_items.clear()
        for item in self._fwhm_label_items:
            self._vb.removeItem(item)
        self._fwhm_label_items.clear()

        use_arcsec = self._fwhm_unit == 'arcsec' and bool(self._plate_scale)
        ps = self._plate_scale or 1.0

        for lv, segs in self._contour_level_segs.items():
            longest = max(segs, key=len)
            mid = len(longest) // 2
            lbl_text = f'{lv * ps:.1f}\u2033' if use_arcsec else f'{lv:.1f}px'
            lbl = pg.TextItem(text=lbl_text, color=(255, 255, 255, 220), anchor=(0.5, 0.5))
            font = QtGui.QFont()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 2)
            lbl.setFont(font)
            lbl.setZValue(3)
            lbl.setPos(longest[mid, 0], longest[mid, 1])
            lbl.setVisible(self._map_contour_visible)
            self._vb.addItem(lbl)
            self._contour_label_items.append(lbl)

        for x, y, fwhm_px in self._fwhm_pts:
            txt = f'{fwhm_px * ps:.1f}\u2033' if use_arcsec else f'{fwhm_px:.1f}px'
            lbl = pg.TextItem(text=txt, color=FG_ACCENT, anchor=(0, 1))
            lbl.setZValue(4)
            lbl.setPos(x + 8, y - 8)
            self._vb.addItem(lbl)
            self._fwhm_label_items.append(lbl)

    def _toggle_fwhm_unit(self):
        self._fwhm_unit = 'px' if self._fwhm_unit == 'arcsec' else 'arcsec'
        self._btn_fwhm_unit.setText('px' if self._fwhm_unit == 'arcsec' else '\u2033')
        self._draw_fwhm_labels()
        self._refresh_map_fwhm_label()
        if self._inspector_active:
            self._redraw_inspector_fwhm_labels()

    def _clear_fwhm_map(self):
        self._overlay_item.show()
        self._overlay_item.clear()
        for item in self._contour_items:
            self._vb.removeItem(item)
        self._contour_items.clear()
        for item in self._contour_label_items:
            self._vb.removeItem(item)
        self._contour_label_items.clear()
        self._scatter.clear()
        for item in self._fwhm_label_items:
            self._vb.removeItem(item)
        self._fwhm_label_items.clear()
        self._fwhm_pts = []
        self._contour_level_segs = {}
        self._btn_fwhm_unit.setEnabled(False)
        self._btn_map_color.setText('No Color')
        self._btn_map_color.hide()
        self._btn_map_contour.setText('No Contours')
        self._btn_map_contour.hide()
        self._map_color_visible   = True
        self._map_contour_visible = True
        self._map_visible   = False
        self._map_computing = False
        self._btn_map.setEnabled(True)
        self._btn_map.setText('FWHM Map')

    def _toggle_map_color(self):
        self._map_color_visible = not self._map_color_visible
        if self._map_color_visible:
            self._overlay_item.show()
            self._btn_map_color.setText('No Color')
        else:
            self._overlay_item.hide()
            self._btn_map_color.setText('Color')

    def _toggle_map_contour(self):
        self._map_contour_visible = not self._map_contour_visible
        for item in self._contour_items:
            item.setVisible(self._map_contour_visible)
        for item in self._contour_label_items:
            item.setVisible(self._map_contour_visible)
        self._btn_map_contour.setText('No Contours' if self._map_contour_visible else 'Contours')

    def _on_map_msg(self, msg: str):
        self._map_computing = False
        self._btn_map.setEnabled(True)
        self._btn_map.setText('FWHM Map')
        QtWidgets.QMessageBox.information(self, 'FWHM Map', msg)

    def _show_fits_header(self):
        if not self._files or self._index >= len(self._files):
            return
        path = self._files[self._index]
        rows: list[tuple[str, str]] = []
        try:
            ext = path.suffix.lower()
            if ext in ('.fits', '.fit', '.fts'):
                with fits.open(str(path)) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 2:
                            for key in hdu.header.keys():
                                if key in ('', 'COMMENT', 'HISTORY'):
                                    continue
                                rows.append((key, str(hdu.header[key])))
                            break
            elif ext == '.xisf':
                _, kw = load_xisf(str(path))
                for key, val in kw.items():
                    rows.append((key, str(val)))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'FITS Header', str(exc))
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f'FITS Header \u2014 {path.name}')
        dlg.resize(520, 620)
        vlay = QtWidgets.QVBoxLayout(dlg)
        vlay.setContentsMargins(8, 8, 8, 8)
        vlay.setSpacing(6)

        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(['Keyword', 'Value'])
        mono = QtGui.QFont('Courier New', 8)
        table.setFont(mono)
        hh = table.horizontalHeader()
        hh.setFont(mono)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)

        for i, (key, val) in enumerate(rows):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(val))
        table.resizeRowsToContents()

        vlay.addWidget(table)

        close_btn = QtWidgets.QPushButton('Close')
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(dlg.accept)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        vlay.addLayout(btn_row)

        dlg.exec()

    def _sync_bottom_bar(self):
        """Keep bot_right width equal to the right splitter column."""
        sizes = self._splitter.sizes()
        if len(sizes) >= 2:
            self._bot_right.setFixedWidth(sizes[1])

    def _update_scope_label(self):
        name, focal_mm, fratio = self._scope_info or (None, None, None)
        parts = []
        if name:
            parts.append(name)
        if focal_mm is not None:
            parts.append(f'{focal_mm:.0f}mm')
        if fratio is not None:
            parts.append(f'f/{fratio:.1f}')
        self._lbl_scope.setText('  \u2022  '.join(parts))

    def _on_scale_changed(self):
        try:
            val = float(self._entry_scale.text())
            self._plate_scale = val if val > 0 else None
        except ValueError:
            self._plate_scale = None
        has_ps = bool(self._plate_scale)
        self._btn_fwhm_unit.setEnabled(has_ps and (self._map_visible or self._inspector_active))
        if not has_ps:
            self._fwhm_unit = 'px'
            self._btn_fwhm_unit.setText('\u2033')
        if self._map_visible and self._fwhm_pts:
            self._draw_fwhm_labels()
        self._refresh_map_fwhm_label()

    def _refresh_map_fwhm_label(self):
        if self._last_map_fwhm is None:
            return
        avg, ctr = self._last_map_fwhm
        use_arcsec = self._fwhm_unit == 'arcsec' and bool(self._plate_scale)
        ps = self._plate_scale or 1.0
        if use_arcsec:
            self._lbl_fwhm.setText(
                f'FWHM  ctr: {ctr*ps:.2f}\u2033  median: {avg*ps:.2f}\u2033  '
                f'[arcseconds]')
        else:
            self._lbl_fwhm.setText(
                f'FWHM  ctr: {ctr:.2f}px  median: {avg:.2f}px  [pixels]')

    # ------------------------------------------------------------------
    # Stretch toggle
    # ------------------------------------------------------------------

    def _on_stretch_toggle(self):
        if self._data is not None:
            self._stretch_params = compute_stretch_params(
                self._data, self._cb_stretch.isChecked()
            )
        self._refresh_display()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def prev_image(self):
        if not self._files:
            return
        self._nav_index = (self._nav_index - 1) % len(self._files)
        self._nav_enqueue(self._nav_index)

    def next_image(self):
        if not self._files:
            return
        self._nav_index = (self._nav_index + 1) % len(self._files)
        self._nav_enqueue(self._nav_index)

    def _nav_enqueue(self, index: int):
        """Queue a navigation request. File list updates only when the image actually displays."""
        self._nav_q.put((self._nav_gen, index, self._files[index]))
        if not self._nav_loader_active:
            self._nav_loader_active = True
            threading.Thread(target=self._nav_loader, daemon=True).start()

    def _nav_loader(self):
        """Single background thread: drain the nav queue in order, loading as needed.
        Enforces a minimum display time so every image is visible even if cached."""
        import queue as _q
        import time
        MIN_DISPLAY_S = 0.7
        last_display_t = 0.0  # monotonic time of last _sig_display.emit
        while True:
            try:
                gen, req_index, path = self._nav_q.get(timeout=0.3)
            except _q.Empty:
                self._nav_loader_active = False
                return
            try:
                with self._cache_lock:
                    in_cache = path in self._cache
                if not in_cache:
                    ext = path.suffix.lower()
                    if ext in ('.fits', '.fit', '.fts'):
                        data, hdr = load_fits(str(path))
                        scale = extract_plate_scale(hdr)
                        scope_info = extract_telescope_info(hdr)
                    elif ext == '.xisf':
                        data, xisf_kw = load_xisf(str(path))
                        scale = extract_plate_scale_xisf(xisf_kw)
                        scope_info = extract_telescope_info(xisf_kw)
                    else:
                        continue
                    precomp = _make_precomp(data, auto=True)
                    self._cache_put(path, data, scale, scope_info, precomp)
                # Ensure at least MIN_DISPLAY_S has elapsed since the previous display,
                # regardless of how long loading took (fixes cache-hit images following
                # a slow load with no gap between them).
                wait = MIN_DISPLAY_S - (time.monotonic() - last_display_t)
                if wait > 0:
                    time.sleep(wait)
                self._sig_display.emit(gen, req_index, path)
                last_display_t = time.monotonic()
            except Exception:
                pass

    def _on_display_ready(self, gen: int, req_index: int, path: object):
        """Main-thread handler: display the just-loaded image if still wanted."""
        if gen != self._nav_gen or not self._files:
            return   # user jumped elsewhere via file list or directory open
        self._index = req_index
        self._load_current()   # in cache — instant
        if self._playing:
            self.next_image()

    def _play_start(self):
        if not self._files:
            return
        self._playing = True
        self._btn_play.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self.next_image()

    def _play_stop(self):
        self._playing = False
        self._nav_gen += 1   # discard any already-queued next image
        self._nav_index = self._index
        self._btn_play.setEnabled(True)
        self._btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete_image(self):
        if not self._files:
            return
        path = self._files[self._index]
        try:
            if HAS_SEND2TRASH:
                _send2trash(str(path))
            else:
                os.remove(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Delete failed', str(exc))
            return

        with self._cache_lock:
            self._cache.pop(path, None)
        self._files.pop(self._index)
        self._files_default = [f for f in self._files_default if f != path]

        if not self._files:
            self._data = None
            self._scatter.clear()
            self._img_item.clear()
            self._lbl_file.setText('No images remaining')
            self._lbl_count.setText('0 / 0')
            self._lbl_fwhm.setText('FWHM: —')
            self._lw_files.clear()
            return

        if self._index >= len(self._files):
            self._index = len(self._files) - 1
        self._populate_file_list()
        self._load_current()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    start = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = Astronalyze(start_path=start)
    viewer.show()
    sys.exit(app.exec())
