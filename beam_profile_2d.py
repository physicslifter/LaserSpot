import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates

def load_tiff():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Interferometer TIFF",
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )
    if not file_path:
        print("No file selected.")
        sys.exit()
    return tifffile.imread(file_path), file_path

def gaussian_1d(x, amp, x0, sigma, offset):
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2)) + offset

def gaussian_2d_rotated(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = coords
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amp * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )
    return g.ravel()

def fit_2d_gaussian(data):
    background = float(np.median(data))
    data_bs = data.astype(float) - background
    peak_y, peak_x = np.unravel_index(np.argmax(data_bs), data_bs.shape)
    peak_val = data_bs[peak_y, peak_x]
    row = data_bs[peak_y, :]
    above_half = np.where(row > 0.5 * peak_val)[0]
    rough_sigma_x = max(3.0, (above_half.max() - above_half.min()) / 2.3548) if len(above_half) > 1 else 10.0
    col = data_bs[:, peak_x]
    above_half_y = np.where(col > 0.5 * peak_val)[0]
    rough_sigma_y = max(3.0, (above_half_y.max() - above_half_y.min()) / 2.3548) if len(above_half_y) > 1 else 10.0
    crop_half = int(5 * max(rough_sigma_x, rough_sigma_y)) + 5
    y_lo = max(0, peak_y - crop_half)
    y_hi = min(data.shape[0], peak_y + crop_half)
    x_lo = max(0, peak_x - crop_half)
    x_hi = min(data.shape[1], peak_x + crop_half)
    crop = data_bs[y_lo:y_hi, x_lo:x_hi]
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    p0 = [peak_val, peak_x - x_lo, peak_y - y_lo, rough_sigma_x, rough_sigma_y, 0.0, 0.0]
    bounds_lo = [0, 0, 0, 1, 1, -np.pi, -np.inf]
    bounds_hi = [np.inf, crop.shape[1], crop.shape[0], crop.shape[1], crop.shape[0], np.pi, np.inf]
    popt, _ = curve_fit(
        gaussian_2d_rotated, (xx, yy), crop.ravel(),
        p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=20000
    )
    amp, x0_local, y0_local, sigma_x, sigma_y, theta, offset = popt

    result = {
        "amp": amp,
        "x0": x0_local + x_lo,       # back to GLOBAL image coordinates
        "y0": y0_local + y_lo,
        "sigma_x": abs(sigma_x),
        "sigma_y": abs(sigma_y),
        "theta_rad": theta,
        "offset": offset,
        "background": background,
        "crop_bounds": (x_lo, x_hi, y_lo, y_hi),
    }
    return result


# ==========================================
# Lineout sampling + 1D fit
# ==========================================
def sample_lineout(data, x0, y0, angle_deg, half_len, n_samples=400):
    theta = np.radians(angle_deg)
    s = np.linspace(-half_len, half_len, n_samples)
    x_coords = x0 + s * np.cos(theta)
    y_coords = y0 + s * np.sin(theta)
    values = map_coordinates(data.astype(float), [y_coords, x_coords], order=1, mode="nearest")
    return s, values


def fit_lineout(s, values):
    background = float(np.median(np.concatenate((values[:10], values[-10:]))))
    values_bs = values - background
    peak = float(values_bs.max())
    if peak <= 0:
        return None

    normalized = values_bs / peak
    sigma_guess = max(2.0, half_width_at_level(s, normalized, 0.5) / 2.3548)

    try:
        popt, _ = curve_fit(
            gaussian_1d, s, normalized,
            p0=[1.0, 0.0, sigma_guess, 0.0], maxfev=20000
        )
        amp, s0, sigma, offset = popt
        sigma = abs(sigma)
        fwhm = 2.3548200450309493 * sigma
        width_e2 = 4.0 * sigma
        fit_curve = gaussian_1d(s, *popt)
        return {"s0": s0, "sigma": sigma, "fwhm": fwhm, "width_e2": width_e2,
                "normalized": normalized, "fit_curve": fit_curve}
    except (RuntimeError, ValueError):
        return None


def half_width_at_level(s, normalized, level):
    above = np.where(normalized > level)[0]
    if len(above) < 2:
        return (s[-1] - s[0]) / 4.0
    return s[above.max()] - s[above.min()]


# ==========================================
# Main
# ==========================================
def main():
    data, filename = load_tiff()
    if data.ndim > 2:
        print(f"Multi-frame TIFF detected with shape {data.shape}. Using first frame.")
        data = data[0]
    print("Fitting 2D Gaussian...")
    fit2d = fit_2d_gaussian(data)
    x0, y0 = fit2d["x0"], fit2d["y0"]
    print(f"Center: ({x0:.2f}, {y0:.2f}) px | sigma_x={fit2d['sigma_x']:.2f}, "
          f"sigma_y={fit2d['sigma_y']:.2f}, theta={np.degrees(fit2d['theta_rad']):.1f} deg")
    half_len = 4.0 * max(fit2d["sigma_x"], fit2d["sigma_y"]) + 10
    half_len = min(half_len, min(data.shape) / 2.0 - 2)
    angles = {"0 deg (H)": 0, "45 deg": 45, "90 deg (V)": 90, "135 deg": 135}
    lineouts = {}
    for label, angle in angles.items():
        s, values = sample_lineout(data, x0, y0, angle, half_len)
        fit1d = fit_lineout(s, values)
        lineouts[label] = (s, values, fit1d)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    ax_img, ax_fit2d = axs[0]
    ax_lineouts, ax_report = axs[1]
    fig.suptitle(f"Beam Profile: {filename.split('/')[-1]}")
    ax_img.imshow(data, cmap="jet", origin="upper")
    ax_img.plot(x0, y0, "w+", markersize=14, markeredgewidth=2)
    for label, angle in angles.items():
        theta = np.radians(angle)
        dx, dy = half_len * np.cos(theta), half_len * np.sin(theta)
        ax_img.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], "--", linewidth=1, label=label)
    ax_img.set_title("Raw Image + Lineout Directions")
    ax_img.legend(loc="upper right", fontsize=7)
    colors = {"0 deg (H)": "tab:blue", "45 deg": "tab:orange", "90 deg (V)": "tab:green", "135 deg": "tab:red"}
    for label, (s, values, fit1d) in lineouts.items():
        if fit1d is None:
            continue
        c = colors[label]
        ax_lineouts.plot(s, fit1d["normalized"], "o", ms=2, alpha=0.4, color=c)
        ax_lineouts.plot(s, fit1d["fit_curve"], "-", linewidth=1.5, color=c, label=label)
    ax_lineouts.set_xlabel("Distance from center (px)")
    ax_lineouts.set_ylabel("Normalized intensity")
    ax_lineouts.set_title("Lineouts + 1D Gaussian Fits")
    ax_lineouts.legend(fontsize=8)
    ax_lineouts.grid(alpha=0.2)
    x_lo, x_hi, y_lo, y_hi = fit2d["crop_bounds"]
    crop = data[y_lo:y_hi, x_lo:x_hi].astype(float)
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    model = gaussian_2d_rotated(
        (xx, yy), fit2d["amp"], x0 - x_lo, y0 - y_lo,
        fit2d["sigma_x"], fit2d["sigma_y"], fit2d["theta_rad"], fit2d["offset"]
    ).reshape(crop.shape)
    ax_fit2d.imshow(crop, cmap="jet", origin="upper",
                     extent=[x_lo, x_hi, y_hi, y_lo])
    levels = fit2d["offset"] + fit2d["amp"] * np.array([np.exp(-2.0), 0.5, 0.9])
    cs = ax_fit2d.contour(np.arange(x_lo, x_hi), np.arange(y_lo, y_hi), model,
                           levels=sorted(levels), colors="white", linewidths=1.2)
    ax_fit2d.plot(x0, y0, "w+", markersize=12, markeredgewidth=2)
    ax_fit2d.set_title("2D Gaussian Fit (contours: 1/e^2, 50%, 90%)")
    ax_report.axis("off")
    fwhm_x = 2.3548200450309493 * fit2d["sigma_x"]
    fwhm_y = 2.3548200450309493 * fit2d["sigma_y"]
    e2_x = 4.0 * fit2d["sigma_x"]
    e2_y = 4.0 * fit2d["sigma_y"]
    lines = [
        "2D Gaussian Fit",
        f"  Center: ({x0:.2f}, {y0:.2f}) px",
        f"  sigma_x = {fit2d['sigma_x']:.2f} px   sigma_y = {fit2d['sigma_y']:.2f} px",
        f"  FWHM_x  = {fwhm_x:.2f} px   FWHM_y  = {fwhm_y:.2f} px",
        f"  1/e^2_x = {e2_x:.2f} px    1/e^2_y = {e2_y:.2f} px",
        f"  Rotation (theta) = {np.degrees(fit2d['theta_rad']):.1f} deg",
        f"  Background = {fit2d['background']:.1f}",
        "",
        "1D Lineout Fits (cross-check)",
    ]
    for label, (s, values, fit1d) in lineouts.items():
        if fit1d is None:
            lines.append(f"  {label:12s}: fit failed")
        else:
            lines.append(
                f"  {label:12s}: FWHM={fit1d['fwhm']:.2f} px   1/e^2={fit1d['width_e2']:.2f} px"
            )
    ax_report.text(0.02, 0.98, "\n".join(lines), transform=ax_report.transAxes,
                    fontsize=10, va="top", family="monospace")
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
