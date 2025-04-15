from datetime import datetime
from dateutil import parser

from check_distribution import check_distribution

import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry, RasterGrid, linear_downscale


def downscale_vapor_pressure_deficit(
        time_UTC: datetime,
        VPD_Pa_coarse: Raster,
        ST_K: Raster,
        fine_geometry: RasterGeometry = None,
        coarse_geometry: RasterGeometry = None,
        resampling: str = None,
        upsampling: str = None,
        downsampling: str = None,
        return_scale_and_bias: bool = False) -> Raster:
    if upsampling is None:
        upsampling = "average"

    if downsampling is None:
        downsampling = "linear"

    if fine_geometry is None:
        fine_geometry = ST_K.geometry

    if coarse_geometry is None:
        coarse_geometry = VPD_Pa_coarse.geometry

    return linear_downscale(
        coarse_image=VPD_Pa_coarse,
        fine_image=ST_K,
        upsampling=upsampling,
        downsampling=downsampling,
        return_scale_and_bias=return_scale_and_bias
    )
