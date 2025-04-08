import xarray as xr
import warnings
import numpy as np
import tarfile
import os
from pathlib import Path
from datetime import datetime

def untar_test_data(tar_path) -> None:
    """Check if test data is untarred. If not, untar the test data.
    """
    if isinstance(tar_path, str):
        tar_path = Path(tar_path)
        
    # Extract the Test Data Set if it does not exist
    if not os.path.exists(tar_path.stem):
        with tarfile.open(tar_path) as tar:
            tar.extractall(tar_path.parent, filter='data')

def apply_coeffs(data_array: xr.DataArray, 
                 m0: float, 
                 m1: float) -> None:
    """Apply the m0 and m1 coefficients to the data array to calculate depth.

    Args:
        m0 (float): Intercept coefficient.
        m1 (float): Slope coefficient.
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        depth = xr.apply_ufunc(
            lambda m0, m1, logratio: m1 * logratio + m0,
            m0,
            m1,
            data_array.sel(band='logratio'),
            output_dtypes=[float]
        )
    
    
    # Identify infinite values
    inf_mask = np.isinf(depth)
    # Replace infinite values with the no-data value
    depth = depth.where(~inf_mask, np.nan)    
    
    depth = depth.expand_dims(band=['depth'])
    depth.rio.write_crs(data_array.rio.crs, inplace=True)
    
    # Append to the data array
    data_array = xr.concat([data_array, depth], dim='band')
    
    return data_array


def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)