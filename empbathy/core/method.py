import pathlib
import warnings
import datetime
from typing import Optional, Union

import xarray as xr
import geopandas as gpd
import numpy as np

from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from empbathy.core.filters import *
from empbathy.tools.plotting import plot_regression

class EmpiricalBathymetry:
    
    """ 
    Class to run the Empirical Bathymetry Method.
    
    There five main steps for usage are:
        - Create an instance of the method class: method = EmpiricalBathymetry()
        - Set the imagery data: method.set_imagery(data_array, *args)
        - Set the insitu data: method.set_insitu(geodataframe, *args)
        - Perform the calibration: method.calibrate(*args)
        - Export the results: method.export()
    """

    def __init__(self, nfactor: float=10000.0):
        
        # Data sources
        self._da  = None   # da  : DataArray
        self._gdf = None   # gdf : GeoDataFrame
        
        # Method parameters
        self._nfactor = nfactor
        self._sigma   = None
        self._minmax  = None
        self._bandi   = None
        self._bandj   = None

        # Calibration results
        self._calibrated = False
        self._m0         = None
        self._m1         = None
        self._rmse       = None
        self._val_rmse   = None
        self._nb         = None


    def set_imagery(self, data_array: xr.DataArray, band_i: Optional[str]='band_i', band_j: Optional[str]='band_j', visualise: bool=False) -> None:
        
        """ Set and preprocess the imagery data for the Empirical Bathymetry method.

        Args:
            data_array (xr.DataArray): DataArray containing the imagery data.
            band_i (str, optional): Band name in xr.DataArray to assign as Band I within the method. Defaults to 'band_i'.
            band_j (str, optional): Band name in xr.DataArray to assign as Band J within the method. Defaults to 'band_i'.
            visualise (bool, optional): Visualise the data array. Default to no visualisation.
            
        """
        
        # Check bands are in the DataArray
        if ((band_i not in data_array.band) or 
            (band_j not in data_array.band)):
            raise ValueError(f"'{band_i}' or '{band_j}' not in data array. Available bands: {data_array.band.values}")

        # Create new logratio band
        temp = self._calculate_logratio(data_array, band_i, band_j)
        
        # Assign the DataArray and band names as attributes
        self._da = temp
        self._bandi, self._bandj = band_i, band_j 
        self._reset_results()
        
        if visualise:
            self._da.plot.imshow(col='band', col_wrap=2, cmap='viridis', aspect=1, size=4)

        
    def set_insitu(self, gdf: gpd.GeoDataFrame, depth_col: str='z', minmax: tuple=None, sigma: float=None, visualise: bool=False) -> None:

        """ Set and preprocess the insitu data for the Empirical Bathymetry method.
        
        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing the insitu data.
            depth_col (str, optional): Column name in the GeoDataFrame containing the depth data. Defaults to 'z'.
            minmax (tuple, optional): Tuple of minimum and maximum depth values to filter the data. Defaults to None.
            sigma (float, optional): Number of standard deviations to filter the data from LogRatio. Defaults to None.
            visualise (bool, optional): Visualise the GeoDataFrame. Default to no visualisation.
            
        """

        if self._da is None:
            raise ValueError('Please set the imagery data array first using .set_imagery()')
                
        # Ensure the GeoDataFrame has the required columns
        if any([i not in gdf.columns for i in [depth_col,'geometry']]):
            raise ValueError('Insitu GeoDataFrame must have columns: z, geometry')
        
        # Subset to the required columns and rename to standardized 'z'
        temp = gdf[[depth_col,'geometry']].copy()
        temp = temp.rename(columns={depth_col:'z'})

        # Align the CRS of the GeoDataFrame to the DataArray
        temp = temp.to_crs(self._da.rio.crs.to_epsg())

        # Create new columns for Band_i, Band_j, and logratio sampled at the geometries
        temp = self._sample_raster(temp)
        
        # Filter out NaN and Inf values
        temp = filter_logvalid(temp, self._nfactor)        
        temp = filter_naninf(temp)   

        if minmax:
            temp = filter_minmax(temp, minmax)
        if sigma:
            temp = filter_sigma(temp, sigma)
           
        # Check if there are enough points to continue
        if len(temp) < 2:
            raise ValueError(f'Cannot continue with only {len(temp)} points. Please check the filters.')
           
        # Assign the GeoDataFrame and filter values as attributes
        self._gdf = temp
        self._minmax = minmax
        self._sigma = sigma
        
        self._zcol = depth_col
        
        # Reset calibration results, if any
        self._reset_results()
           
        if visualise:
            self._gdf.plot()
           
    def calibrate(self, validation: float=None, visualise: bool=False) -> tuple:
        
        """ Perform the calibration of the empirical bathymetry model using the loaded imagery and insitu data.
        
        Args:
            validation (float, optional): Fraction of the data to use for validation. Defaults to None.
            visualise (bool, optional): Visualise the calibration. Default to no visualisation.
            
        """

        if (self._da is None) or (self._gdf is None):
            raise ValueError('Please set the imagery and insitu data first using .set_imagery() and .set_insitu()')

        # Check if validation is required, check the argument is valid and split the data
        if validation:
            if validation > 1 or validation < 0:
                raise ValueError('Validation must be a proportion float between 0 and 1.')
            train_gdf, test_gdf = train_test_split(self._gdf, test_size=validation)
        else:
            train_gdf = self._gdf.copy()

        # Fit a linear regression model to the log-ratio and depth data
        train_x = train_gdf['logratio'].values.reshape(-1,1)
        train_y = train_gdf['z'].values.reshape(-1,1)
        
        model = LinearRegression().fit(train_x, train_y)

        # At our sample points, compare the modelled and measured depth
        pred_y = model.predict(train_x)

        self._rmse = np.sqrt(mean_squared_error(train_y.flatten(),pred_y.flatten()))
        self._m0   = model.intercept_[0]
        self._m1   = model.coef_[0][0]
        self._nb   = len(train_gdf)

        if validation:
            test_x = test_gdf['logratio'].values.reshape(-1,1)
            test_y = model.predict(test_x)
            self._val_rmse = np.sqrt(mean_squared_error(test_gdf['z'].values.reshape(-1,1).flatten(),test_y.flatten()))

        self._apply_coeffs(self._m0, self._m1)

        self._calibrated = True
        
        if visualise:
            plot_regression(train_gdf, test_gdf=test_gdf, m0=self._m0, m1=self._m1, metric=self._rmse, col=self._zcol)

    @property
    def array(self, bands: list=None) -> xr.DataArray:
        self._iscalibrated()
        if bands is None:
            return self._da
        else:
            return self._da.sel(band=bands)
    
    @property
    def depth(self) -> xr.DataArray:
        self._iscalibrated()
        return self._da.sel(band='z')
    
    @property
    def insitu(self) -> gpd.GeoDataFrame:
        self._iscalibrated()
        return self._gdf
    
    @property
    def stats(self) -> dict:
        self._iscalibrated()
        return {
            'm0': float(self._m0),
            'm1': float(self._m1),
            'nb': int(self._nb),            
            'rmse': float(self._rmse),
            'validation_rmse': float(self._val_rmse) if self._val_rmse else None,
        }


    def _iscalibrated(self) -> bool:
        """ Check if the model has been calibrated.
        """
        if self._calibrated is False:
            raise ValueError('Please calibrate the model before exporting the results with .calibrate()')


    def _apply_coeffs(self, m0: float, m1: float) -> None:
        """ Apply the m0 and m1 coefficients to the data array to calculate depth.
        Args:
            m0 (float): Intercept coefficient.
            m1 (float): Slope coefficient.
        """
        
        # Check if Depth has already been calculated, if so, remove it before recalculating
        if 'z' in self._da.band:
            self._da = self._da.drop_sel(band='z')
        
        # Apply the coefficients to the logratio band to calculate depth
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            depth = xr.apply_ufunc(
                lambda m0, m1, logratio: m1 * logratio + m0,
                m0,
                m1,
                self._da.sel(band='logratio'),
                output_dtypes=[float]
            )
        
        # Identify infinite values
        inf_mask = np.isinf(depth)
        
        # Replace infinite values with the no-data value
        depth = depth.where(~inf_mask, np.nan)    
        
        # Add the depth band to the data array and copy CRS
        depth = depth.expand_dims(band=['z'])
        depth.rio.write_crs(self._da.rio.crs, inplace=True)
        
        # Append to the data array
        self._da = xr.concat([self._da, depth], dim='band')
           
           
    def _sample_raster(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add reflectance values sampled at the in-situ points to the GeoDataFrame."""
        
        # Use vectorized nearest-neighbor sampling
        selected = self._da.sel(
            x=xr.DataArray(gdf.geometry.x.values, dims='z'),
            y=xr.DataArray(gdf.geometry.y.values, dims='z'),
            method='nearest'
        )

        # Assign sampled values to GeoDataFrame
        gdf['band_i'] = selected.sel(band=self._bandi).values
        gdf['band_j'] = selected.sel(band=self._bandj).values
        gdf['logratio'] = selected.sel(band='logratio').values

        return gdf
        
        
    def _calculate_logratio(self, da: xr.DataArray, bandi: str, bandj: str) -> None:
        """Calculate band ratio as an additional band in the data array.
        """
        # Get the bands for the method
        band_i = da.sel(band=bandi)
        band_j = da.sel(band=bandj)
        # Calculate the logratio
        with np.errstate(divide='ignore', invalid='ignore'):
            logratio = (np.log(self._nfactor * band_i) / np.log(self._nfactor * band_j))
        # Append and return
        logratio = logratio.expand_dims(band=['logratio'])
        return xr.concat([da, logratio], dim='band')

        
    def _reset_results(self):
        """ Reset the results of the calibration in events such as, a calibration has already occured, but the user re-sets the insitu data.
        """
        if self._calibrated == True:
            print('New insitu/imagery data has been set. Resetting calibration results...')
            
        self._calibrated = False
        self._m0 = None
        self._m1 = None
        self._nb = None
        self._rmse = None
        self._val_rmse = None