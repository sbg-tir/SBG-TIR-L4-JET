from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

def NDVI_to_FVC(NDVI: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Converts Normalized Difference Vegetation Index (NDVI) to Fractional Vegetation Cover (FVC).

    The formula assumes a linear relationship between NDVI and FVC, where NDVI is scaled
    between bare soil (NDVIs) and full vegetation (NDVIv). The resulting FVC is clipped
    to the range [0, 1].

    Parameters:
        NDVI (Union[Raster, np.ndarray]): Input NDVI values as a Raster or NumPy array.

    Returns:
        Union[Raster, np.ndarray]: Fractional Vegetation Cover (FVC) values.

    Reference:
        Gutman, G., & Ignatov, A. (1998). "The derivation of the green vegetation fraction 
        from NOAA/AVHRR data for use in numerical weather prediction models." 
        International Journal of Remote Sensing, 19(8), 1533-1543. 
        DOI: 10.1080/014311698215333
    """
    NDVIv = 0.52  # +- 0.03
    NDVIs = 0.04  # +- 0.03
    FVC = rt.clip((NDVI - NDVIs) / (NDVIv - NDVIs), 0, 1)

    return FVC