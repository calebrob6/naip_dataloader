import multiprocessing
import os
import random

import geopandas
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from affine import Affine
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from tqdm import tqdm

NUM_PROCESSES = 24
MIN_YEAR = 2018
NUM_POINTS = 100_000


def random_crop_to_file(row):
    """Reads a random 256 x 256 meter crop from an input GeoTIFF and saves the result as
    a 256 x 256 pixel GeoTIFF (i.e. 1m/px resolution) resampling with bilinear
    interpolation as necessary. It is necessary to resample because sometimes NAIP is
    0.6m/px resolution.

    Args:
        url: input URL (preferably a Cloud Optimized GeoTIFF)
        output_fn: local file in which to save the resulting crop

    Returns:
        lat, lon centroid of the resulting crop
    """
    with rasterio.open(row["naip:url"]) as f:
        xmin, ymin, xmax, ymax = f.bounds
        src_crs = f.crs.to_string()
        width = xmax - xmin
        height = ymax - ymin

        xoffset = random.random() * width - 256
        yoffset = random.random() * height - 256

        window = f.window(
            xmin + xoffset,
            ymin + yoffset,
            xmin + xoffset + 256,
            ymin + yoffset + 256,
        )
        data = f.read(
            window=window, out_shape=(256, 256), resampling=Resampling.bilinear
        )

        dst_transform = Affine(
            1.0, 0.0, xmin + xoffset, 0.0, -1.0, ymin + yoffset + 256
        )

        profile = f.profile.copy()
        profile["transform"] = dst_transform
        profile["height"] = 256
        profile["width"] = 256
        profile["photometric"] = "RGB"
        profile["compress"] = "deflate"
        profile["predictor"] = 2
        profile["blockxsize"] = 256
        profile["blockysize"] = 256

        with rasterio.open(row["fn"], "w", **profile) as g:
            g.write(data)

    transformer = Transformer.from_crs(src_crs, "EPSG:4326")
    lon, lat = transformer.transform(xmin + xoffset + 128, ymin + yoffset + 128)

    row.update(
        {
            "tile_xmin": xmin,
            "tile_ymin": ymin,
            "tile_xmax": xmax,
            "tile_ymax": ymax,
            "xoffset": xoffset,
            "yoffset": yoffset,
            "centroid_latitude": lat,
            "centroid_longitude": lon,
            "fn": os.path.basename(row["fn"]),
        }
    )

    return row


def main():
    # the Pool workers seems to randomly hang without this
    multiprocessing.set_start_method("spawn")

    print("Downloading the NAIP geoparquet dataset (this will take a bit)")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )
    asset = catalog.get_collection("naip").assets["geoparquet-items"]
    df = geopandas.read_parquet(
        asset.href, storage_options=asset.extra_fields["table:storage_options"]
    )
    print(f"Found {df.shape[0]} items")

    print("Filtering results by the most recent tiles captured for each state")
    df["year"] = df["naip:year"].apply(lambda x: int(x))

    states = set(df["naip:state"])
    subset = df[df["year"] >= 2018]

    seen_states = set()
    repeated_state_years = []
    for year, state in sorted(
        list(set(zip(subset["year"], subset["naip:state"]))), reverse=True
    ):
        if state not in seen_states:
            seen_states.add(state)
        else:
            repeated_state_years.append((year, state))
    assert len(states - seen_states) == 0

    for year, state in repeated_state_years:
        mask = (subset["year"] == year) & (subset["naip:state"] == state)
        subset = subset[~mask]
    print(f"{subset.shape[0]} items remain after filtering")

    idxs = np.random.choice(subset.shape[0], size=NUM_POINTS, replace=True)
    assets = subset.iloc[idxs]["assets"].values
    years = subset.iloc[idxs]["naip:year"].values
    state = subset.iloc[idxs]["naip:state"].values

    input_rows = []
    for i in tqdm(range(NUM_POINTS)):
        row = {
            "idx": i,
            "naip:url": assets[i]["image"]["href"],
            "fn": f"data/images/sample_{i}.tif",
            "naip:year": years[i],
            "naip:state": state[i],
        }
        input_rows.append(row)
    df = pd.DataFrame(input_rows)
    df.to_csv("data/index_input.csv", index=False)

    print(f"Running parallel downloads with {NUM_PROCESSES} processes")
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        rows = list(
            tqdm(
                pool.imap_unordered(random_crop_to_file, input_rows),
                total=NUM_POINTS,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv("data/index.csv", index=False)


if __name__ == "__main__":
    main()
