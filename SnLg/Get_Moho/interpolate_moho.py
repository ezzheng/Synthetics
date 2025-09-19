import numpy as np
from functools import lru_cache

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and cache Moho grid
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_gravity_moho_grid(filepath):
    """
    Load and cache the gravity-derived Moho grid from a text file.

    Parameters:
        filepath (str): Path to gravity Moho file

    Returns:
        tuple:
            zmo (np.ndarray): Moho depths (2D)
            xm  (np.ndarray): Longitude grid (2D)
            ym  (np.ndarray): Latitude grid (2D)
    """
    try:
        data = np.loadtxt(filepath, usecols=2)
    except Exception as e:
        raise RuntimeError(f"Failed to load Moho file '{filepath}': {e}")

    zmo = data.reshape(109, 189).astype(float)
    zmo[zmo >= 1e4] = np.nan  # Mask invalid or fill values

    xmo = np.linspace(65, 112, 189)
    ymo = np.linspace(20, 47, 109)
    xm, ym = np.meshgrid(xmo, ymo)

    return zmo, xm, ym

@lru_cache(maxsize=1)
def load_rf_moho_grid(filepath):
    """
    Load and cache the RF-derived Moho grid from a text file.

    Parameters:
        filepath (str): Path to RF Moho file

    Returns:
        tuple:
            zmo (np.ndarray): Moho depths (2D)
            xm  (np.ndarray): Longitude grid (2D)
            ym  (np.ndarray): Latitude grid (2D)
    """
    try:
        data = np.loadtxt(filepath, usecols=2)
    except Exception as e:
        raise RuntimeError(f"Failed to load Moho file '{filepath}': {e}")

    zmo = data.reshape(65, 145).astype(float)
    zmo[zmo >= 1e4] = np.nan  # Mask invalid or fill values

    xmo = np.linspace(74, 110, 145)
    ymo = np.linspace(26, 42, 65)
    xm, ym = np.meshgrid(xmo, ymo)

    return zmo, xm, ym

# ─────────────────────────────────────────────────────────────────────────────
# 2. Vectorized Haversine distance
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distances (km) between a scalar point and 2D grid.
    """
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Weighted average Moho depth from elliptical Gaussian
# ─────────────────────────────────────────────────────────────────────────────

def calculate_weighted_average(zmo, xm, ym, point, d1=100, d2=100, azm=0, alpha=0.5):
    """
    Compute weighted average Moho depth around (lon, lat) using elliptical Gaussian.
    """
    evlo, evla = point
    dist_km = haversine_km(evla, evlo, ym, xm)

    azimuth_rad = np.arctan2(
        np.deg2rad(xm - evlo) * np.cos(np.deg2rad((ym + evla) / 2)),
        np.deg2rad(ym - evla)
    )

    azm_rad = np.deg2rad(azm)
    d_x = dist_km * np.cos(azimuth_rad - azm_rad)
    d_y = dist_km * np.sin(azimuth_rad - azm_rad)

    mask = (d_x**2 / d1**2 + d_y**2 / d2**2) < 1
    valid = mask & ~np.isnan(zmo)

    if not np.any(valid):
        return np.nan

    sigma_x = alpha * d1
    sigma_y = alpha * d2
    w = np.exp(-0.5 * ((d_x / sigma_x) ** 2 + (d_y / sigma_y) ** 2)) * valid

    return np.nansum(w * zmo) / np.sum(w)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main access function
# ─────────────────────────────────────────────────────────────────────────────

def get_moho(filepath, evlo, evla, load_moho_grid_func, is_plot=False):
    """
    Return interpolated Moho depth at event location (evlo, evla).
    """
    try:
        zmo, xm, ym = load_moho_grid_func(filepath)
    except Exception as e:
        print(f"Error in reading {filepath} using function {load_moho_grid_func}, because {e}")
        return None, None

    moho_depth = calculate_weighted_average(
        zmo, xm, ym, point=(evlo, evla),
        d1=100, d2=100, azm=0, alpha=0.5
    )

    fig = None
    if is_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        cf = ax.contourf(xm, ym, zmo, levels=np.arange(30, 85, 5), cmap='jet', alpha=0.6)
        ax.contour(xm, ym, zmo, levels=[40, 50, 60, 70], colors='k', linestyles='--')
        plt.colorbar(cf, label='Moho Depth (km)', ax=ax)
        ax.set_xlim(80, 95)
        ax.set_ylim(25, 33)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Moho Depth Map')
        plt.tight_layout()

    return moho_depth, fig
