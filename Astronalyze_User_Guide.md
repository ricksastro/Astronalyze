# Astronalyze User Guide

Astronalyze is a fast astronomical image viewer for reviewing FITS and XISF sub-frames. It is designed for astrophotographers who need to quickly assess focus quality and seeing across many images.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Opening Images](#2-opening-images)
3. [Navigating Images](#3-navigating-images)
4. [Viewing and Stretch](#4-viewing-and-stretch)
5. [Zoom Controls](#5-zoom-controls)
6. [Plate Scale](#6-plate-scale)
7. [Sorting the File List](#7-sorting-the-file-list)
8. [FWHM Map](#8-fwhm-map)
9. [Corner Inspector](#9-corner-inspector)
10. [FITS Header Viewer](#10-fits-header-viewer)
11. [Slideshow / Play Mode](#11-slideshow--play-mode)
12. [Deleting Images](#12-deleting-images)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)
14. [Configuration and Window State](#14-configuration-and-window-state)
15. [File Association](#15-file-association)

---

## 1. Getting Started

### System Requirements
- Windows 10/11 (64-bit)
- No Python installation required when using the `.exe` build

### Installation
1. Copy the entire `Astronalyze\` folder to any location on your machine.
2. Run `register.bat` **as Administrator** once to associate `.fits`, `.fit`, `.fts`, and `.xisf` files with Astronalyze. After registration you can double-click any of those files in Explorer to open them directly.

---

## 2. Opening Images

### From Explorer
Double-click any `.fits`, `.fit`, `.fts`, or `.xisf` file. Astronalyze opens, loads the entire directory, and displays the file you clicked.

### From within Astronalyze
Click the **Open Dir** button in the toolbar and select a folder. All supported image files in that folder are loaded into the file list on the right. The last-used directory is remembered between sessions.

---

## 3. Navigating Images

### File List
The right-hand panel shows all images in the current directory. Click any filename to load it immediately.

### Prev / Next Buttons
The **◀** and **▶** buttons (bottom of the file list) step through images one at a time. Images are pre-loaded in the background so navigation is near-instant.

### Keyboard
| Key | Action |
|-----|--------|
| ← | Previous image |
| → | Next image |

---

## 4. Viewing and Stretch

### Auto Stretch
The **Auto Stretch** checkbox (bottom-left) applies a PixInsight-style Screen Transfer Function (STF) that automatically reveals faint nebulosity while preserving star shape. Uncheck it to view the image with a simple linear stretch from black to the data's natural bit-depth maximum.

### Pixel Value Display
Click anywhere on the image to read the raw pixel value at that point. The value is shown in the bottom bar. For XISF files stored as normalised [0, 1] floats the value is scaled back to the equivalent 16-bit (0–65535) range for consistency.

---

## 5. Zoom Controls

The zoom toolbar sits below the image:

| Control | Action |
|---------|--------|
| **−** / **+** buttons | Zoom out / in by 1.5× steps |
| **50%** | Zoom to half resolution (1 image pixel = 2 screen pixels) |
| **100%** | Zoom to native resolution (1 image pixel = 1 screen pixel) |
| **200%** | Zoom to double resolution |
| **Fit** | Fit the entire image in the window |
| Mouse wheel | Zoom in/out at the cursor position |
| Click and drag | Pan |

### Keyboard
| Key | Action |
|-----|--------|
| + or = | Zoom in |
| − | Zoom out |
| F | Fit image |

---

## 6. Plate Scale

The **Scale** field (zoom toolbar, far right) shows and sets the plate scale in arcseconds per pixel (″/px). Astronalyze tries to read this automatically from the FITS header (WCS, FOCALLEN/XPIXSZ, PIXSCALE, etc.) when an image is loaded. You can also type a value manually.

The plate scale is used to display FWHM measurements in arcseconds rather than pixels. The **″** toggle button switches between the two units wherever FWHM values are displayed.

---

## 7. Sorting the File List

The **Sort** section at the top of the right panel contains three buttons:

| Button | Sort order |
|--------|-----------|
| **Default** | Alphabetical / filesystem order |
| **FWHM** | By measured FWHM value (requires FWHM Map to have been computed) |
| **Stars** | Numerically by the integer following `stars` in the filename (e.g. `light_stars123_001.fits` → 123) |

- The active sort button is highlighted and prefixed with **+** (ascending) or **−** (descending).
- Clicking the active sort button a second time **reverses** the order.
- Clicking any sort button always navigates to the top of the newly sorted list.

---

## 8. FWHM Map

The FWHM Map analyses star shapes across the entire image to produce a spatial map of focus/seeing quality.

### Computing the Map
Click **FWHM Map**. Astronalyze detects stars, fits a Moffat profile (β = 4) to each one, then interpolates the results using Gaussian kernel regression. The computation runs in a background thread; the button shows *Computing…* while it runs.

### Reading the Map
Once complete, a colour overlay appears on the image:

- **Green** = small FWHM (good focus / good seeing)
- **Yellow** = intermediate
- **Red** = large FWHM (poor focus / poor seeing)

White **contour lines** mark iso-FWHM levels. Each contour is labelled with its FWHM value. Labels on contour lines are **bold and larger** to distinguish them from the per-star labels underneath each detected star.

The colour range is set from the 5th to 95th percentile of measured values to prevent single outliers from dominating the scale. Contour levels are spaced at least 0.2″ apart so labels always represent distinct values.

The bottom bar shows:
- **FWHM: median (px or ″)** — median over all detected stars
- **Ctr: median (px or ″)** — median over stars in the central 50% of the image (inside the dashed white box)

### Toggle Controls
While the map is displayed, two extra buttons appear:

| Button | Action |
|--------|--------|
| **No Color** | Hide the colour overlay (contours remain); press again (**Color**) to restore |
| **No Contours** | Hide contour lines and labels; press again (**Contours**) to restore |

These buttons are hidden when no map is displayed. The colour and contour state always reset to *on* when a new map is computed or when you navigate to a different image.

### Unit Toggle
The **″** button switches all FWHM labels between arcseconds (requires plate scale) and pixels. It is enabled whenever a map or inspector FWHM result is displayed.

### Hiding the Map
Click **Hide FWHM Map** to remove the overlay and return the button to **FWHM Map**.

---

## 9. Corner Inspector

The Inspector shows nine 100%-scale crops simultaneously so you can assess sharpness and coma across the full field of view in a single view.

### Opening the Inspector
Click **Inspector**. The main image is replaced by a 3×3 grid:

| Position | Content |
|----------|---------|
| Top-left, Top-right, Bottom-left, Bottom-right | 100% crops of each corner |
| Top-centre, Bottom-centre, Left-centre, Right-centre | 100% crops of each edge midpoint |
| Centre | 100% crop of the image centre |

All nine cells are background-normalised so that the 20th-percentile sky level renders at **10% gray** in every cell, making brightness differences between corners easy to spot. This normalisation is applied in raw (pre-stretch) pixel space so it works correctly with the nonlinear STF.

### FWHM in Inspector Mode
While the inspector is open, the **FWHM Map** button changes to **FWHM**. Click it to measure the FWHM of stars visible in each of the nine cells (up to 25 brightest stars per cell). Stars are circled and labelled with their individual FWHM values, just like the full map. The ″/px toggle works here too.

- When FWHM results are shown the button reads **No FWHM** — click it to clear the labels.
- Labels near the right edge of a cell automatically flip to the left side of the circle so they are never cut off.
- Resizing the window re-aligns circles and labels to match the redrawn crops.

### Closing the Inspector
Click **Close Inspector** to return to the normal image view.

---

## 10. FITS Header Viewer

Click **FITS Header** (right end of the zoom toolbar, just to the left of the ◀ button) to open a scrollable table of all keyword/value pairs from the image header. Works for both FITS and XISF files. The table uses a compact monospace font. Click **Close** at the bottom to dismiss it.

---

## 11. Slideshow / Play Mode

Click **Play** (bottom bar) to step through the image list automatically. Each image is displayed as soon as it is ready; there is a short minimum display time to prevent flicker on fast drives.

Click **Stop** to pause at the current image.

---

## 12. Deleting Images

Click **Delete** (nav row, bottom-right of the file list) or press the **Delete** key to send the current image to the Recycle Bin (using send2trash if available, otherwise permanently deleted). Astronalyze then loads the next image automatically.

---

## 13. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| ← | Previous image |
| → | Next image |
| Delete | Delete current image |
| + or = | Zoom in |
| − | Zoom out |
| F | Fit image in window |

---

## 14. Configuration and Window State

Astronalyze saves the following settings automatically to `.astronalyze_config.json` (located next to the `.exe` or the script):

- **Last opened directory** — restored on next launch
- **Window position, size, and state** — the window reopens exactly where you left it, including maximised or full-screen state

---

## 15. File Association

Run `dist\Astronalyze\register.bat` **as Administrator** to register Astronalyze as the default application for:

| Extension | Type |
|-----------|------|
| `.fits` | FITS Astronomical Image |
| `.fit` | FITS Astronomical Image |
| `.fts` | FITS Astronomical Image |
| `.xisf` | XISF Astronomical Image |

After registration:
- Double-clicking a file in Explorer opens Astronalyze and loads that file directly.
- The file's directory is loaded into the file list so you can browse neighbouring frames.
- Astronalyze also appears in the right-click **Open with** menu for those file types.

You may need to sign out and back in (or restart Explorer) for icon changes to take effect.
