import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
import xarray as xr
import geopandas as gpd


class Utilities:
    """
    Utility methods for the Empirical Bathymetry Method.
    """

    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE) between true and predicted values.

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The RMSE value.
        """
        y_true = np.asarray(y_true).reshape(-1, 1)
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _apply_coeffs(self, m0: float, m1: float) -> xr.DataArray:
        """
        Apply the m0 and m1 coefficients to the data array to calculate depth.

        Args:
            m0 (float): Intercept coefficient.
            m1 (float): Slope coefficient.

        Returns:
            xr.DataArray: DataArray with the calculated depth band.
        """
        # Check if Depth has already been calculated, if so, remove it before recalculating
        if "z" in self._da.band:
            self._da = self._da.drop_sel(band="z")

        # Apply the coefficients to the logratio band to calculate depth
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            depth = xr.apply_ufunc(
                lambda m0, m1, logratio: m1 * logratio + m0,
                m0,
                m1,
                self._da.sel(band="logratio"),
                output_dtypes=[float],
            )

        # Identify infinite values
        inf_mask = np.isinf(depth)

        # Replace infinite values with the no-data value
        depth = depth.where(~inf_mask, np.nan)

        # Add the depth band to the data array and copy CRS
        depth = depth.expand_dims(band=["z"])
        depth.rio.write_crs(self._da.rio.crs, inplace=True)

        return depth

    def _calculate_logratio(
        self, da: xr.DataArray, bandi: str, bandj: str
    ) -> xr.DataArray:
        """
        Calculate the log-ratio band as an additional band in the data array.

        Args:
            da (xr.DataArray): DataArray containing the imagery data.
            bandi (str): Name of the first band (Band I).
            bandj (str): Name of the second band (Band J).

        Returns:
            xr.DataArray: DataArray with the added log-ratio band.
        """
        # Get the bands for the method
        band_i = da.sel(band=bandi)
        band_j = da.sel(band=bandj)

        # Calculate the logratio
        with np.errstate(divide="ignore", invalid="ignore"):
            logratio = np.log(self._nfactor * band_i) / np.log(self._nfactor * band_j)

        # Append and return
        logratio = logratio.expand_dims(band=["logratio"])
        return xr.concat([da, logratio], dim="band")

    def _sample_raster(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add reflectance values sampled at the in-situ points to the GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing the in-situ points.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with sampled reflectance values added.
        """
        # Use vectorized nearest-neighbor sampling
        selected = self._da.sel(
            x=xr.DataArray(gdf.geometry.x.values, dims="z"),
            y=xr.DataArray(gdf.geometry.y.values, dims="z"),
            method="nearest",
        )

        # Assign sampled values to GeoDataFrame
        gdf["band_i"] = selected.sel(band=self._bandi).values
        gdf["band_j"] = selected.sel(band=self._bandj).values
        gdf["logratio"] = selected.sel(band="logratio").values

        return gdf
