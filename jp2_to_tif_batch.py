#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jp2_to_tif_batch.py
Convierte Sentinel-2 JP2 -> GeoTIFF y (opcional) recorta al AOI.
- Busca automáticamente B04_10m.jp2, B08_10m.jp2 y SCL_20m.jp2 (o QA60_60m.jp2)
  dentro de una carpeta (por defecto: ./DATA), sin importar nombres exactos.

Uso:
  python jp2_to_tif_batch.py --data-dir ./DATA --prefix 2020 --aoi aoi.geojson
  python jp2_to_tif_batch.py --data-dir ./DATA_2024 --prefix 2024 --aoi aoi.geojson

Requisitos: pip install rasterio
"""
import os
import sys
import json
import argparse
import glob
import rasterio
from rasterio.mask import mask

def read_aoi(geojson_path):
    if not geojson_path or not os.path.exists(geojson_path):
        return None
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        return [feat["geometry"] for feat in gj["features"]]
    if gj.get("type") == "Feature":
        return [gj["geometry"]]
    return [gj]  # geometry directa

def find_one(data_dir, patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None

def transcode(in_path, out_path, aoi_geom=None):
    if in_path is None:
        return None
    with rasterio.open(in_path) as src:
        profile = src.profile
        profile.update(driver="GTiff")
        if aoi_geom:
            arr, transform = mask(src, aoi_geom, crop=True)
            profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr)
        else:
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())
    print(f"[OK] {os.path.basename(out_path)}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./DATA", help="Carpeta con los JP2 (búsqueda recursiva).")
    ap.add_argument("--prefix",   default="2020",   help="Sufijo para nombres de salida (2020/2024).")
    ap.add_argument("--aoi",      default="",       help="GeoJSON de AOI para recortar (opcional).")
    ap.add_argument("--out-dir",  default="",       help="Carpeta de salida (default: ./salidas_tif_<prefix>).")
    args = ap.parse_args()

    aoi_geom = read_aoi(args.aoi)
    if aoi_geom:
        print("[INFO] AOI cargado. Se recortará al polígono.")
    out_dir = args.out_dir or f"./salidas_tif_{args.prefix}"
    os.makedirs(out_dir, exist_ok=True)

    # Buscar archivos
    b04 = find_one(args.data_dir, ["*B04*10m.jp2", "*B04*.jp2"])
    b08 = find_one(args.data_dir, ["*B08*10m.jp2", "*B08*.jp2"])
    scl = find_one(args.data_dir, ["*SCL*20m.jp2", "*SCL*.jp2"])
    qa60= find_one(args.data_dir, ["*QA60*60m.jp2", "*QA60*.jp2"])

    if not b04 or not b08:
        print("[ERROR] No se encontraron B04/B08 en", args.data_dir)
        print("       Asegúrate de que tus JP2 están en la carpeta o subcarpetas.")
        sys.exit(2)

    if not scl and not qa60:
        print("[WARN] No se encontró SCL ni QA60. Continuaré con B04/B08 solamente.")

    # Convertir (y recortar si AOI)
    b04_tif = os.path.join(out_dir, f"B04_{args.prefix}.tif")
    b08_tif = os.path.join(out_dir, f"B08_{args.prefix}.tif")
    transcode(b04, b04_tif, aoi_geom)
    transcode(b08, b08_tif, aoi_geom)

    if scl:
        scl_tif = os.path.join(out_dir, f"SCL_{args.prefix}.tif")
        transcode(scl, scl_tif, aoi_geom)
    elif qa60:
        qa60_tif = os.path.join(out_dir, f"QA60_{args.prefix}.tif")
        transcode(qa60, qa60_tif, aoi_geom)

    print("\n[LISTO] GeoTIFFs en:", os.path.abspath(out_dir))
    print("Usa estas rutas en tu notebook para NDVI y cambio.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
