from datetime import datetime
from dateutil import parser

import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry, linear_downscale

def downscale_air_temperature(
        time_UTC: datetime,
        Ta_K_coarse: Raster,
        ST_K: Raster,
        water: Raster = None,
        fine_geometry: RasterGeometry = None,
        coarse_geometry: RasterGeometry = None,
        upsampling: str = None,
        downsampling: str = None,
        apply_scale: bool = True,
        apply_bias: bool = True,
        return_scale_and_bias: bool = False) -> Raster:
    """
    near-surface air temperature (Ta) in Kelvin
    :param time_UTC: date/time in UTC
    :param geometry: optional target geometry
    :param resampling: optional sampling method for resampling to target geometry
    :return: raster of Ta
    """

    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    if fine_geometry is None:
        fine_geometry = ST_K.geometry

    if coarse_geometry is None:
        coarse_geometry = Ta_K_coarse.geometry

    if upsampling is None:
        upsampling = "average"

    if downsampling is None:
        downsampling = "cubic"

    ST_K_water = None

    if water is not None:
        ST_K_water = rt.where(water, ST_K, np.nan)
        ST_K = rt.where(water, np.nan, ST_K)

    scale = None
    bias = None

    Ta_K = linear_downscale(
        coarse_image=Ta_K_coarse,
        fine_image=ST_K,
        upsampling=upsampling,
        downsampling=downsampling,
        apply_scale=apply_scale,
        apply_bias=apply_bias,
        return_scale_and_bias=return_scale_and_bias
    )

    if water is not None:
        Ta_K_water = linear_downscale(
            coarse_image=Ta_K_coarse,
            fine_image=ST_K_water,
            upsampling=upsampling,
            downsampling=downsampling,
            apply_scale=apply_scale,
            apply_bias=apply_bias,
            return_scale_and_bias=False
        )

        Ta_K = rt.where(water, Ta_K_water, Ta_K)

    Ta_K.filenames = Ta_K_coarse.filenames

    return Ta_K
