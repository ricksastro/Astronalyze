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
7. [Filtering the File List](#7-filtering-the-file-list)
8. [Sorting the File List](#8-sorting-the-file-list)
9. [FWHM Map](#9-fwhm-map)
10. [Collimation](#10-collimation)
11. [Corner Inspector](#11-corner-inspector)
12. [FITS Header Viewer](#12-fits-header-viewer)
13. [Slideshow / Play Mode](#13-slideshow--play-mode)
14. [Deleting Images](#14-deleting-images)
15. [Keyboard Shortcuts](#15-keyboard-shortcuts)
16. [Configuration and Window State](#16-configuration-and-window-state)
17. [File Association](#17-file-association)

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

## 7. Filtering the File List

The filter controls at the top of the right panel narrow the file list to matching filenames. Navigation, Play, sorting and Delete then work only on the filtered list.

### Quick Filter Buttons
The **L R G B S H O** buttons filter by filter name in one click. Each button matches both its letter and the spelled-out name, as long as the name is surrounded by underscores in the filename:

| Button | Matches |
|--------|---------|
| **L** | `_L_` or `_Lum_` |
| **R** | `_R_` or `_Red_` |
| **G** | `_G_` or `_Green_` |
| **B** | `_B_` or `_Blue_` |
| **S** | `_S_` or `_SII_` |
| **H** | `_H_` or `_Ha_` |
| **O** | `_O_` or `_OIII_` |

The active quick button is highlighted. Clicking it fills the filter box with the pattern it uses (for example `*_R_*|*_Red_*`), so you can see it and edit it.

### Custom Patterns
Type a wildcard pattern in the filter box and press **Enter** or click **Filter**:

- `*` matches any run of characters and `?` matches one character.
- Matching ignores upper/lower case.
- To match any of several patterns, separate them with `|`, for example `*_Ha_*|*_OIII_*`.

The filter box is highlighted while a filter is active. If nothing matches, a message says so and the list is left unchanged.

### Clearing the Filter
Click **Clear** to show all files again. Opening a new directory also clears the filter. The current sort order is kept when you change or clear the filter.

---

## 8. Sorting the File List

The **Sort** section at the top of the right panel contains three buttons:

| Button | Sort order |
|--------|-----------|
| **Default** | Alphabetical / filesystem order |
| **FWHM** | Numerically by the value following `FWHM` in the filename (e.g. `light_FWHM2.35_001.fits` → 2.35) |
| **Stars** | Numerically by the integer following `stars` in the filename (e.g. `light_stars123_001.fits` → 123) |

The FWHM and Stars sorts read values from the filename, not from measurements made in Astronalyze. Capture software such as NINA can add these values to filenames. Files without a value go to the end of the list. If no file in the list has a value, a message says so and the order is left unchanged.

- The active sort button is highlighted and prefixed with **+** (ascending) or **−** (descending).
- Clicking the active sort button a second time **reverses** the order.
- Clicking any sort button always navigates to the top of the newly sorted list.
- When a filter is active, sorting applies to the filtered list.

---

## 9. FWHM Map

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

## 10. Collimation

Collimation analysis measures defocused star images ("donuts") to find where in the field the telescope's optics are best aligned. It is meant for telescopes with a central obstruction (SCT, RC, Newtonian and similar), whose defocused stars show a dark central shadow inside a bright ring.

### Capturing a Suitable Frame
- **Defocus well.** The rings must be at least 12 px in radius, and 20 px or more is better. Results become reliable at about 15 px.
- **Use a field with plenty of stars.** At least 6 usable donuts are needed, and more give a steadier answer.
- Donuts that touch the frame edge, overlap another donut, are vignetted into crescents, or differ a lot in size from the rest are ignored automatically.

### Running the Analysis
Click **Collimation** (bottom bar). The analysis runs in the background and the button shows *Computing…*. Any FWHM Map that is showing is hidden first. Close the Inspector before you start, because Collimation is not available in Inspector mode.

### How It Works
For each donut, Astronalyze measures how far the central shadow sits from the centre of the bright ring, as a percentage of the ring radius. With good collimation the shadow is centred. With miscollimation the shadow is pushed sideways, and the push changes steadily across the field. Astronalyze fits that pattern and finds the point where the offset would be zero. That point is the optical axis, where collimation is best.

### Reading the Result
- **Colour overlay:** green where shadows are well centred and red where they are most offset. White contour lines are labelled with the offset in % of the ring radius.
- **Arrows:** one per donut, pointing toward the optical axis. Each arrow comes from that donut's own measurement, so arrows that converge neatly mean the data is consistent. Arrow length is proportional to the donut's distance from the axis.
- **BEST COLLIMATION marker (+):** the fitted optical axis, with its pixel coordinates. If the axis falls outside the image, the marker sits at the nearest edge and is labelled *(off frame)*, but the coordinates shown are the true ones.

The bottom bar summarises the result, for example:

`Collim: best @ (2310,1580) ±45px — 210px off center   med Δ=6.2% of ring   n=48  R=22px  fit 0.93 (good)`

| Item | Meaning |
|------|---------|
| **best @ (x,y) ±N px** | Position of the optical axis and its estimated uncertainty |
| **N px off center** | Distance from the axis to the image centre. For a well-collimated scope this should be small. |
| **med Δ** | Median shadow offset over all donuts, in % of ring radius |
| **n** | Number of donuts used |
| **R** | Median ring radius in pixels |
| **fit** | How well the measured offsets follow the expected pattern (1.00 = perfect), with a confidence rating: **good** for rings of 15 px or more, **fair** for smaller rings |

The **No Color** / **No Contours** buttons work the same way as for the FWHM Map.

### When No Result Is Shown
Astronalyze would rather show nothing than show a misleading answer. Instead of a result, you get a message explaining why when:

- fewer than 6 donuts were found,
- the donuts are too small (frame too close to focus), or
- the offsets are too inconsistent or too unstable to locate the axis.

In most cases the fix is to defocus further, or to use a field with more stars.

### Hiding the Result
Click **Hide Collimation**. The result is also cleared when you move to another image, open the FWHM Map, or open the Inspector.

---

## 11. Corner Inspector

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

## 12. FITS Header Viewer

Click **FITS Header** (right end of the zoom toolbar, just to the left of the ◀ button) to open a scrollable table of all keyword/value pairs from the image header. Works for both FITS and XISF files. The table uses a compact monospace font. Click **Close** at the bottom to dismiss it.

---

## 13. Slideshow / Play Mode

Click **Play** (bottom bar) to step through the image list automatically. Each image is displayed as soon as it is ready; there is a short minimum display time to prevent flicker on fast drives.

Click **Stop** to pause at the current image.

---

## 14. Deleting Images

Click **Delete** (nav row, bottom-right of the file list) or press the **Delete** key to send the current image to the Recycle Bin (using send2trash if available, otherwise permanently deleted). Astronalyze then loads the next image automatically.

---

## 15. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| ← | Previous image |
| → | Next image |
| Delete | Delete current image |
| + or = | Zoom in |
| − | Zoom out |
| F | Fit image in window |

---

## 16. Configuration and Window State

Astronalyze saves the following settings automatically to `.astronalyze_config.json` (located next to the `.exe` or the script):

- **Last opened directory** — restored on next launch
- **Window position, size, and state** — the window reopens exactly where you left it, including maximised or full-screen state

---

## 17. File Association

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
