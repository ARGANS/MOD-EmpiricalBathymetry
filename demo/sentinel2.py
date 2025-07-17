import geopandas as gpd
import xarray as xr
import numpy as np
import rioxarray as rio
from sdam import SDAM

# --- Setup ---
# Step 1) Use Geopandas to read in in-situ vector file.
insitu = gpd.read_file("/home/cmackenzie/development/copphil/APP-CopPhilBathyService/.ignore/BORACAY-INSITU-ALLPOINTS.geojson")
insitu["z"] = insitu["z"].astype(float)  # Define datatype, sometimes read in as string

# Step 2) Load Sentinel-2 L2A Bands
B02 = rio.open_rasterio("/home/cmackenzie/development/copphil/APP-CopPhilBathyService/.ignore/S2A_MSIL2A_20230926T022331_N0509_R103_T51PUP_20230926T062553.SAFE/GRANULE/L2A_T51PUP_A043143_20230926T022328/IMG_DATA/R10m/T51PUP_20230926T022331_B02_10m.jp2").squeeze()
B03 = rio.open_rasterio("/home/cmackenzie/development/copphil/APP-CopPhilBathyService/.ignore/S2A_MSIL2A_20230926T022331_N0509_R103_T51PUP_20230926T062553.SAFE/GRANULE/L2A_T51PUP_A043143_20230926T022328/IMG_DATA/R10m/T51PUP_20230926T022331_B03_10m.jp2").squeeze()
SCL = rio.open_rasterio("/home/cmackenzie/development/copphil/APP-CopPhilBathyService/.ignore/S2A_MSIL2A_20230926T022331_N0509_R103_T51PUP_20230926T062553.SAFE/GRANULE/L2A_T51PUP_A043143_20230926T022328/IMG_DATA/R20m/T51PUP_20230926T022331_SCL_20m.jp2").squeeze()

# Step 3) Combine into a DataArray with named bands
imagery = xr.concat([B02, B03], dim="band")
imagery["band"] = ["B02", "B03"]

# Step 4) Prepare and apply the mask to the imagery
SCL = SCL.rio.reproject_match(imagery)
valid_values = [6.0]  # 6 = Water Pixels 
valid_mask = np.isin(SCL, valid_values)
imagery = imagery.where(valid_mask)  # Apply mask

# --- SDAM ---
# Step 5) Initialize the SDAM method object
method = SDAM(nfactor=15000.0, visualise=True)

# Step 6) Set the imagery, in-situ data and perform calibration
method.set_imagery(imagery, band_i="B02", band_j="B03")
method.set_insitu(insitu, minmax=(3.0, 12.0), sigma=3.0)
method.calibrate(validation=0.1)

# Step 7) Print the resulting statistics and write the depth image to disk
print(method.results.metrics)
method.results.array.rio.to_raster("EXAMPLE-S2-TIF.tif")
method.results.plot.savefig("EXAMPLE-S2-PLOT.png")
