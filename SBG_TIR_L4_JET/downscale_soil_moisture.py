from datetime import datetime
from dateutil import parser

from check_distribution import check_distribution

import numpy as np
import rasters as rt
from rasters import Raster, RasterGrid, linear_downscale

from .NDVI_to_FVC import NDVI_to_FVC

def downscale_soil_moisture(
        time_UTC: datetime,
        geometry: RasterGrid,
        coarse_geometry: RasterGrid,
        target: str,
        SM_coarse: Raster,
        SM_resampled: Raster,
        ST_fine: Raster,
        NDVI_fine: Raster,
        water: Raster,
        fvlim=0.5,
        a=0.5,
        downsampling="linear") -> Raster:
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    date_UTC = time_UTC.date()
    ST_fine = ST_fine.mask(~water)
    check_distribution(ST_fine, "ST_fine", date_UTC=date_UTC, target=target)
    NDVI_fine = NDVI_fine.mask(~water)
    check_distribution(NDVI_fine, "NDVI_fine", date_UTC=date_UTC, target=target)
    FVC_fine = NDVI_to_FVC(NDVI_fine)
    check_distribution(FVC_fine, "FVC_fine", date_UTC=date_UTC, target=target)
    soil_fine = FVC_fine < fvlim
    check_distribution(soil_fine, "soil_fine", date_UTC=date_UTC, target=target)
    Tmin_coarse = ST_fine.to_geometry(coarse_geometry, resampling="min")
    check_distribution(Tmin_coarse, "Tmin_coarse", date_UTC=date_UTC, target=target)
    Tmax_coarse = ST_fine.to_geometry(coarse_geometry, resampling="max")
    check_distribution(Tmax_coarse, "Tmax_coarse", date_UTC=date_UTC, target=target)
    Ts_fine = ST_fine.mask(soil_fine)
    check_distribution(Ts_fine, "Ts_fine", date_UTC=date_UTC, target=target)
    Tsmin_coarse = Ts_fine.to_geometry(coarse_geometry, resampling="min").fill(Tmin_coarse)
    check_distribution(Tsmin_coarse, "Tsmin_coarse", date_UTC=date_UTC, target=target)
    Tsmax_coarse = Ts_fine.to_geometry(coarse_geometry, resampling="max").fill(Tmax_coarse)
    check_distribution(Tsmax_coarse, "Tsmax_coarse", date_UTC=date_UTC, target=target)
    ST_coarse = ST_fine.to_geometry(coarse_geometry, resampling="average")
    check_distribution(ST_coarse, "ST_coarse", date_UTC=date_UTC, target=target)
    SEE_coarse = (Tsmax_coarse - ST_coarse) / rt.clip(Tsmax_coarse - Tsmin_coarse, 1, None)
    check_distribution(SEE_coarse, "SEE_coarse", date_UTC=date_UTC, target=target)
    SM_SEE_proportion = (SM_coarse / SEE_coarse).to_geometry(geometry, resampling=downsampling)
    check_distribution(SM_SEE_proportion, "SM_SEE_proportion", date_UTC=date_UTC, target=target)
    Tsmax_fine = Tsmax_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(Tsmax_fine, "Tsmax_fine", date_UTC=date_UTC, target=target)
    Tsrange_fine = (Tsmax_coarse - Tsmin_coarse).to_geometry(geometry, resampling=downsampling)
    check_distribution(Tsrange_fine, "Tsrange_fine", date_UTC=date_UTC, target=target)
    SEE_fine = (Tsmax_fine - ST_fine) / rt.clip(Tsrange_fine, 1, None)
    check_distribution(SEE_fine, "SEE_fine", date_UTC=date_UTC, target=target)
    SEE_mean = SEE_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(SEE_mean, "SEE_mean", date_UTC=date_UTC, target=target)
    SM_fine = rt.clip(SM_resampled + a * SM_SEE_proportion * (SEE_fine - SEE_mean), 0, 1)
    SM_fine = SM_fine.mask(~water)
    check_distribution(SM_fine, "SM_fine", date_UTC=date_UTC, target=target)

    return SM_fine
