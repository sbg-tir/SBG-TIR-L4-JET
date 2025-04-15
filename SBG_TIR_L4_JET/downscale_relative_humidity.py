from datetime import datetime
from dateutil import parser

from check_distribution import check_distribution

import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry, RasterGrid, linear_downscale, bias_correct

def downscale_relative_humidity(
        time_UTC: datetime,
        RH_coarse: Raster,
        SM,
        ST_K,
        VPD_kPa,
        water_mask: Raster = None,
        fine_geometry: RasterGeometry = None,
        coarse_geometry: RasterGeometry = None,
        resampling: str = None,
        upsampling: str = None,
        downsampling: str = None) -> Raster:
    if upsampling is None:
        upsampling = "average"

    if downsampling is None:
        downsampling = "linear"

    if fine_geometry is None:
        fine_geometry = SM.geometry

    if coarse_geometry is None:
        coarse_geometry = RH_coarse.geometry

    bias_fine = None

    RH_estimate_fine = SM ** (1 / VPD_kPa)

    RH = bias_correct(
        coarse_image=RH_coarse,
        fine_image=RH_estimate_fine,
        upsampling=upsampling,
        downsampling=downsampling,
        return_bias=False
    )

    if water_mask is not None:
        ST_K_water = rt.where(water_mask, ST_K, np.nan)
        RH_coarse_complement = 1 - RH_coarse
        RH_complement_water = linear_downscale(
            coarse_image=RH_coarse_complement,
            fine_image=ST_K_water,
            upsampling=upsampling,
            downsampling=downsampling,
            apply_bias=True,
            return_scale_and_bias=False
        )

        RH_water = 1 - RH_complement_water
        RH = rt.where(water_mask, RH_water, RH)

    RH = rt.clip(RH, 0, 1)

    return RH

