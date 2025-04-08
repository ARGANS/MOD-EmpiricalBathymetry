import geopandas as gpd
import numpy as np
from scipy.stats import zscore
    
    
def filter_minmax(gdf: gpd.GeoDataFrame, minmax: tuple) -> gpd.GeoDataFrame:
    """ Filter the GeoDataFrame to only include rows where the depth is within the minmax tuple range.
    """
    return gdf[gdf['z'].between(*minmax)]


def filter_sigma(gdf: gpd.GeoDataFrame, sigma: float) -> gpd.GeoDataFrame:
    """ Filter outliers from the GeoDataFrame using the zscore of the logratio.
    """
    gdf['zscore'] = np.abs(zscore(gdf['logratio']))
    return gdf[gdf['zscore'] < sigma]


def filter_logvalid(gdf: gpd.GeoDataFrame, nfactor) -> gpd.GeoDataFrame:
    """ 
    Remove rows where (N * Ref_i/j) is less than or equal to 1
    This will remove any rows where: 
                    A) Either reflectances are negative (L2A artefact in cloud shadows)
                    B) Either reflectances are equal to 1 after multiplying by N. These cause logratio to be 0, and zero div errors. (e.g. 0.0001 * 10000 = 1.0)
    """
    gdf = gdf[((nfactor * gdf['band_i']) > 1.0 ) & 
              ((nfactor * gdf['band_j']) > 1.0 ) ]
    return gdf


def filter_naninf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove any rows with NaN values in the reflectance columns.
    """
    # Change -Inf and Inf values to NaN, Remove rows with NaN values in any data column
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    return gdf.dropna()
