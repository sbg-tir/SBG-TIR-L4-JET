from datetime import datetime
from dateutil import parser
import numpy as np
from rasters import Raster, RasterGeometry, RasterGrid
import rasters as rt

DEFAULT_UPSAMPLING = "average"
DEFAULT_DOWNSAMPLING = "linear"

def bias_correct(
        coarse_image: Raster,
        fine_image: Raster,
        upsampling: str = "average",
        downsampling: str = "linear",
        return_bias: bool = False):
    fine_geometry = fine_image.geometry
    coarse_geometry = coarse_image.geometry
    upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)
    bias_coarse = upsampled - coarse_image
    bias_fine = bias_coarse.to_geometry(fine_geometry, resampling=downsampling)
    bias_corrected_fine = fine_image - bias_fine

    if return_bias:
        return bias_corrected_fine, bias_fine
    else:
        return bias_corrected_fine

def linear_downscale(
        coarse_image: Raster,
        fine_image: Raster,
        upsampling: str = "average",
        downsampling: str = "linear",
        use_gap_filling: bool = False,
        apply_scale: bool = True,
        apply_bias: bool = True,
        return_scale_and_bias: bool = False) -> Raster:
    if upsampling is None:
        upsampling = DEFAULT_UPSAMPLING

    if downsampling is None:
        downsampling = DEFAULT_DOWNSAMPLING

    coarse_geometry = coarse_image.geometry
    fine_geometry = fine_image.geometry
    upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)

    if apply_scale:
        scale_coarse = coarse_image / upsampled
        scale_coarse = rt.where(coarse_image == 0, 0, scale_coarse)
        scale_coarse = rt.where(upsampled == 0, 0, scale_coarse)
        scale_fine = scale_coarse.to_geometry(fine_geometry, resampling=downsampling)
        scale_corrected_fine = fine_image * scale_fine
        fine_image = scale_corrected_fine
    else:
        scale_fine = fine_image * 0 + 1

    if apply_bias:
        upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)
        bias_coarse = upsampled - coarse_image
        bias_fine = bias_coarse.to_geometry(fine_geometry, resampling=downsampling)
        bias_corrected_fine = fine_image - bias_fine
        fine_image = bias_corrected_fine
    else:
        bias_fine = fine_image * 0

    if use_gap_filling:
        gap_fill = coarse_image.to_geometry(fine_geometry, resampling=downsampling)
        fine_image = fine_image.fill(gap_fill)

    if return_scale_and_bias:
        fine_image["scale"] = scale_fine
        fine_image["bias"] = bias_fine

    return fine_image

def NDVI_to_FVC(NDVI: Raster) -> Raster:
    NDVIv = 0.52  # +- 0.03
    NDVIs = 0.04  # +- 0.03
    FVC = rt.clip((NDVI - NDVIs) / (NDVIv - NDVIs), 0, 1)

    return FVC

def downscale_air_temperature(
        time_UTC: datetime,
        Ta_K_coarse: Raster,
        ST_K: Raster,
        water: Raster = None,
        fine_geometry: RasterGeometry = None,
        coarse_geometry: RasterGeometry = None,
        resampling: str = None,
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


def downscale_soil_moisture(
        time_UTC: datetime,
        fine_geometry: RasterGrid,
        coarse_geometry: RasterGrid,
        SM_coarse: Raster,
        SM_resampled: Raster,
        ST_fine: Raster,
        NDVI_fine: Raster,
        water: Raster,
        fvlim=0.5,
        a=0.5,
        smoothing="linear") -> Raster:
    fine = fine_geometry
    ST_fine = ST_fine.mask(~water)
    NDVI_fine = NDVI_fine.mask(~water)
    FVC_fine = NDVI_to_FVC(NDVI_fine)
    soil_fine = FVC_fine < fvlim
    Tmin_coarse = ST_fine.to_geometry(coarse_geometry, resampling="min")
    Tmax_coarse = ST_fine.to_geometry(coarse_geometry, resampling="max")
    Ts_fine = ST_fine.mask(soil_fine)
    Tsmin_coarse = Ts_fine.to_geometry(coarse_geometry, resampling="min").fill(Tmin_coarse)
    Tsmax_coarse = Ts_fine.to_geometry(coarse_geometry, resampling="max").fill(Tmax_coarse)
    ST_coarse = ST_fine.to_geometry(coarse_geometry, resampling="average")
    SEE_coarse = (Tsmax_coarse - ST_coarse) / rt.clip(Tsmax_coarse - Tsmin_coarse, 1, None)
    SM_SEE_proportion = (SM_coarse / SEE_coarse).to_geometry(fine, resampling=smoothing)
    Tsmax_fine = Tsmax_coarse.to_geometry(fine_geometry, resampling=smoothing)
    Tsrange_fine = (Tsmax_coarse - Tsmin_coarse).to_geometry(fine, resampling=smoothing)
    SEE_fine = (Tsmax_fine - ST_fine) / rt.clip(Tsrange_fine, 1, None)

    SEE_mean = SEE_coarse.to_geometry(fine, resampling=smoothing)
    SM_fine = rt.clip(SM_resampled + a * SM_SEE_proportion * (SEE_fine - SEE_mean), 0, 1)
    SM_fine = SM_fine.mask(~water)

    return SM_fine


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


def downscale_relative_humidity(
        time_UTC: datetime,
        RH_coarse: Raster,
        SM,
        ST_K,
        VPD_kPa,
        water: Raster = None,
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

    if water is not None:
        ST_K_water = rt.where(water, ST_K, np.nan)
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
        RH = rt.where(water, RH_water, RH)

    RH = rt.clip(RH, 0, 1)

    return RH

