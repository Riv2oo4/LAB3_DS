import os, glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# === Carpetas  ===
DIR20 = "./DATA2020"
DIR24 = "./DATA2024"
OUT_DIR = "./salidas_tif_2024"
os.makedirs(OUT_DIR, exist_ok=True)

def find_band(dirpath, band_tag):
    patt = os.path.join(dirpath, f"*{band_tag}*.tif*")
    files = sorted(glob.glob(patt))
    if not files:
        raise FileNotFoundError(f"No encontré {band_tag} en {dirpath}")
    return files[0]

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return arr, profile, transform, crs, nodata

def reproject_to_match(src_path, match_profile):
    with rasterio.open(src_path) as src:
        dst = np.empty((match_profile["height"], match_profile["width"]), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=match_profile["transform"],
            dst_crs=match_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return dst

def compute_ndvi(nir, red):
    denom = (nir + red)
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi

def save_tif(path, arr, profile):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)

def show(arr, title, vmin=None, vmax=None):
    plt.figure()
    plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# --- localizar bandas ---
B04_2020 = find_band(DIR20, "B04")
B08_2020 = find_band(DIR20, "B08")
B04_2024 = find_band(DIR24, "B04")
B08_2024 = find_band(DIR24, "B08")

print("2020:", os.path.basename(B04_2020), "|", os.path.basename(B08_2020))
print("2024:", os.path.basename(B04_2024), "|", os.path.basename(B08_2024))

# --- leer 2020 ---
red20, prof20, tfm20, crs20, _ = read_band(B04_2020)
nir20, _, _, _, _ = read_band(B08_2020)

# Asegurar que B04 y B08 de 2020 estén en el mismo grid
if red20.shape != nir20.shape:
    nir20 = reproject_to_match(B08_2020, prof20)

ndvi20 = compute_ndvi(nir20, red20)
save_tif(os.path.join(OUT_DIR, "NDVI_2020.tif"), ndvi20, prof20)

# --- leer 2024 ---
red24, prof24, tfm24, crs24, _ = read_band(B04_2024)
nir24, _, _, _, _ = read_band(B08_2024)

if red24.shape != nir24.shape:
    nir24 = reproject_to_match(B08_2024, prof24)

ndvi24 = compute_ndvi(nir24, red24)
save_tif(os.path.join(OUT_DIR, "NDVI_2024.tif"), ndvi24, prof24)

# --- reprojectar NDVI 2020 al grid de 2024  ---
if ndvi20.shape != ndvi24.shape or (tfm20 != tfm24):
    tmp_path = os.path.join(OUT_DIR, "NDVI_2020_tmp.tif")
    save_tif(tmp_path, ndvi20, prof20)
    ndvi20 = reproject_to_match(tmp_path, prof24)

# --- diferencia (2024 - 2020) ---
diff = ndvi24 - ndvi20
save_tif(os.path.join(OUT_DIR, "NDVI_diff_24_minus_20.tif"), diff, prof24)

# --- visualizaciones ---
show(ndvi20, "NDVI 2020", vmin=-1, vmax=1)
show(ndvi24, "NDVI 2024", vmin=-1, vmax=1)
show(diff, "Diferencia NDVI (2024 - 2020)", vmin=-1, vmax=1)

# --- máscara de pérdida significativa ---
THRESH = -0.2  
valid = (~np.isnan(ndvi20)) & (~np.isnan(ndvi24))
deforest_mask = (diff < THRESH) & valid
save_tif(os.path.join(OUT_DIR, "deforest_mask.tif"), deforest_mask.astype("float32"), prof24)
show(deforest_mask.astype(int), f"Máscara deforestación (diff < {THRESH})", vmin=0, vmax=1)

# --- hectáreas y porcentaje ---
px_w = abs(prof24["transform"][0])
px_h = abs(prof24["transform"][4])
m2_per_pixel = px_w * px_h
ha_per_pixel = m2_per_pixel / 10_000.0

defor_pixels = np.count_nonzero(deforest_mask)
valid_pixels = np.count_nonzero(valid)

defor_ha = defor_pixels * ha_per_pixel
total_ha = valid_pixels * ha_per_pixel
defor_pct = (defor_ha / total_ha * 100) if total_ha > 0 else np.nan

print(f"Área analizada (sin NaN): {total_ha:,.2f} ha")
print(f"Hectáreas con pérdida (umbral {THRESH}): {defor_ha:,.2f} ha")
print(f"Porcentaje de deforestación: {defor_pct:.2f}%")
