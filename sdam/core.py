from logging import getLogger
from typing import Optional, Tuple

import geopandas as gpd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sdam.filter import filter_logvalid, filter_naninf, filter_minmax, filter_sigma
from sdam.results import ResultsHandler
from sdam.utilities import Utilities
from sdam.visuals import Visualisation

log = getLogger(__name__)


class SDAM(Utilities, Visualisation):
    """
    Stumpfs Differential Attenuation Method Class for remotely derived bathymetry.
    """

    def __init__(
        self, 
        nfactor: float = 10000.0, 
        visualise: bool = False, 
        verbose: bool = False
    ) -> None:
        """
        Initialize the SDAM class.

        Args:
            nfactor (float): Offset factor for filtering log-ratio values. Defaults to 10000.0.
            visualise (bool): Whether to visualize the results with matplotib figures. Defaults to False.
            verbose (bool): Whether to print verbose output. Defaults to False.
        """
        # Input
        self._da: Optional[xr.DataArray] = None  # da: DataArray
        self._gdf: Optional[gpd.GeoDataFrame] = None  # gdf: GeoDataFrame

        # Options
        self._visualise: bool = visualise
        self._verbose: bool = verbose

        # Method
        self._nfactor: float = nfactor
        self._sigma: Optional[float] = None
        self._minmax: Optional[Tuple[float, float]] = None
        self._bandi: Optional[str] = None
        self._bandj: Optional[str] = None
        self._unit: Optional[str] = None

        # Results
        self._calibrated: bool = False
        self._results: Optional[ResultsHandler] = None

    @property
    def imagery(
        self
        ) -> xr.DataArray:
        """
        Get the imagery data.

        Returns:
            xr.DataArray: The imagery data array.

        Raises:
            ValueError: If imagery data is not set.
        """
        if self._da is None:
            raise ValueError("DataArray is None. Set imagery using .set_imagery()")
        return self._da

    @property
    def insitu(
        self
        ) -> gpd.GeoDataFrame:
        """
        Get the insitu data.

        Returns:
            gpd.GeoDataFrame: The insitu GeoDataFrame.

        Raises:
            ValueError: If insitu data is not set.
        """
        if self._gdf is None:
            raise ValueError("GeoDataFrame is None. Set insitu using .set_insitu()")
        return self._gdf

    @property
    def results(
        self
    ) -> ResultsHandler:
        """
        Get the calibration results.

        Returns:
            ResultsHandler: The results handler object.

        Raises:
            ValueError: If results are not available.
        """
        if self._results is None:
            raise ValueError("No results available. Run .calibrate() first")
        return self._results

    def set_imagery(
        self,
        data_array: xr.DataArray,
        band_i: Optional[str] = "band_i",
        band_j: Optional[str] = "band_j",
        discard: bool = False,
    ) -> None:
        """
        Set and preprocess the imagery data for the Empirical Bathymetry method.

        Args:
            data_array (xr.DataArray): DataArray containing the imagery data.
            band_i (Optional[str]): Band name in xr.DataArray to assign as Band I within the method. Defaults to 'band_i'.
            band_j (Optional[str]): Band name in xr.DataArray to assign as Band J within the method. Defaults to 'band_j'.
            discard (bool): Whether to discard band i and band j after calculating the log ratio band. Defaults to False.

        Raises:
            ValueError: If specified bands are not in the DataArray.
        """
        if band_i not in data_array.band or band_j not in data_array.band:
            raise ValueError(f"'{band_i}' or '{band_j}' not in data array. Available bands: {data_array.band.values}")

        temp = self._calculate_logratio(data_array, band_i, band_j)
        self._da = temp
        self._bandi, self._bandj = band_i, band_j
        self._reset_results()

        if discard:
            msg = f"Discarding bands {self._bandi} and {self._bandj} due to set_imagery( ... discard=True)"
            log.info(msg)
            if self._verbose:
                print(msg)

            del self._da[band_i]
            del self._da[band_j]

        if self._visualise:
            self._da.plot.imshow(
                col="band", col_wrap=2, cmap="viridis", aspect=1, size=4
            )

    def set_insitu(
        self,
        gdf: gpd.GeoDataFrame,
        depth_col: str = "z",
        minmax: Optional[Tuple[float, float]] = None,
        sigma: Optional[float] = None,
        unit: Optional[str] = None,
    ) -> None:
        """
        Set and preprocess the insitu data for the Empirical Bathymetry method.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing the insitu data.
            depth_col (str): Column name in the GeoDataFrame containing the depth data. Defaults to 'z'.
            minmax (Optional[Tuple[float, float]]): Tuple of minimum and maximum depth values to filter the data. Defaults to None.
            sigma (Optional[float]): Number of standard deviations to filter the data from LogRatio. Defaults to None.
            unit (Optional[str]): Unit of the depth data. Defaults to None.

        Raises:
            ValueError: If imagery data is not set or required columns are missing in the GeoDataFrame.
        """
        if self._da is None:
            raise ValueError("Please set the imagery data array first using .set_imagery()")

        if any(col not in gdf.columns for col in [depth_col, "geometry"]):
            raise ValueError(f"Insitu GeoDataFrame must have columns: {depth_col}, geometry")

        temp = gdf[[depth_col, "geometry"]].copy()
        temp = temp.rename(columns={depth_col: "z"})
        temp = temp.to_crs(self._da.rio.crs.to_epsg())
        temp = self._sample_raster(temp)
        temp = filter_logvalid(temp, self._nfactor)
        temp = filter_naninf(temp)

        if minmax:
            temp = filter_minmax(temp, minmax)
        if sigma:
            temp = filter_sigma(temp, sigma)

        if len(temp) < 2:
            raise ValueError(f"Cannot continue with only {len(temp)} points. Please check the filters.")

        self._gdf = temp
        self._minmax = minmax
        self._sigma = sigma
        self._unit = unit
        self._reset_results()

        if self._visualise:
            self._gdf.plot(
                color="red",
                markersize=5,
                label=f"Insitu Points (NB : {len(self._gdf)})",
            )

    def calibrate(
        self, 
        validation: Optional[float] = None
    ) -> Tuple[dict, ResultsHandler]:
        """
        Perform the calibration of the empirical bathymetry model using the loaded imagery and insitu data.

        Args:
            validation (Optional[float]): Fraction of the data to use for validation. Must be a float between 0 and 1. Defaults to None.

        Returns:
            ResultsHandler: An object containing the calibration metrics, depth array and results.

        Raises:
            ValueError: If imagery or insitu data is not set.
        """
        if self._da is None or self._gdf is None:
            raise ValueError("Please set the imagery and insitu data first using .set_imagery() and .set_insitu()")

        metrics = {}

        if validation:
            if validation > 1 or validation < 0:
                raise ValueError("Validation must be a proportion float between 0.0 and 1.0 (e.g. 0.2 for 20% validation set)")
            train_gdf, test_gdf = train_test_split(self._gdf, test_size=validation)
        else:
            train_gdf, test_gdf = self._gdf.copy(), None

        train_x = train_gdf["logratio"]
        train_y = train_gdf["z"]
        model = LinearRegression().fit(
            train_x.values.reshape(-1, 1), train_y.values.reshape(-1, 1)
        )

        train_pred = model.predict(train_x.values.reshape(-1, 1))
        metrics.update({
                "m0": model.intercept_[0],
                "m1": model.coef_[0][0],
                "nb": len(train_gdf),
                "rmse": self._calculate_rmse(train_y, train_pred),
            })

        if validation:
            test_x = test_gdf["logratio"]
            test_y = test_gdf["z"]
            test_pred = model.predict(test_x.values.reshape(-1, 1))
            metrics.update({
                    "val_nb": len(test_gdf),
                    "val_rmse": self._calculate_rmse(
                        test_y.values.reshape(-1, 1), test_pred),
                })

        array = self._apply_coeffs(metrics["m0"], metrics["m1"])
        plot = self._plot_calibration(train_gdf, test_gdf, metrics)

        self._calibrated = True
        self._results = ResultsHandler(
            self, metrics=metrics, estimator=model, plot=plot, array=array
        )

        if self._visualise:
            self._results.plot.show()

    def _reset_results(self) -> None:
        """
        Reset the results of the calibration in events such as a calibration has already occurred, but the user re-sets the insitu data.
        """
        if self._calibrated:
            print(
                "New insitu/imagery data has been set. Resetting calibration results..."
            )

        if "z" in self._da.band:
            self._da = self._da.drop_sel(band="z")

        self._calibrated = False
        self._results = None
