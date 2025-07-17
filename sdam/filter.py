import geopandas as gpd
import numpy as np
from scipy.stats import zscore
from typing import Tuple


def filter_minmax(
    gdf: gpd.GeoDataFrame, minmax: Tuple[float, float]
) -> gpd.GeoDataFrame:
    """
    Filter the GeoDataFrame to only include rows where the depth is within the specified range.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the insitu data.
        minmax (Tuple[float, float]): Tuple specifying the minimum and maximum depth values.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with rows within the specified depth range.
    """
    return gdf[gdf["z"].between(*minmax)]


def filter_sigma(gdf: gpd.GeoDataFrame, sigma: float) -> gpd.GeoDataFrame:
    """
    Filter outliers from the GeoDataFrame using the z-score of the log-ratio column.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the insitu data.
        sigma (float): Threshold for the z-score. Rows with z-scores greater than this value are removed.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with outliers removed based on z-score.
    """
    gdf["zscore"] = np.abs(zscore(gdf["logratio"]))
    return gdf[gdf["zscore"] < sigma]


def filter_logvalid(gdf: gpd.GeoDataFrame, nfactor: float) -> gpd.GeoDataFrame:
    """
    Remove rows where (N * Ref_i/j) is less than or equal to 1.

    This will remove rows where:
        A) Either reflectances are negative (Sometimes an atmos. corr. artifact in cloud shadows).
        B) Either reflectances are equal to 1 after multiplying by N, causing log-ratio to be 0 and zero division errors.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the insitu data.
        nfactor (float): Normalization factor applied to reflectance values.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with invalid log-ratio rows removed.
    """
    gdf = gdf[((nfactor * gdf["band_i"]) > 1.0) & ((nfactor * gdf["band_j"]) > 1.0)]
    return gdf


def filter_naninf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove rows with NaN or infinite values in the reflectance columns.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the insitu data.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with NaN and infinite values removed.
    """
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    return gdf.dropna()
