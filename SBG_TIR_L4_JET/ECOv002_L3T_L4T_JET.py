import logging
import posixpath
import shutil
import socket
import sys
import warnings
from datetime import datetime
from os import makedirs
from os.path import join, abspath, dirname, expanduser, exists, basename
from shutil import which
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from dateutil import parser

import colored_logging as cl

import rasters as rt
from rasters import Raster, RasterGrid, RasterGeometry

from koppengeiger import load_koppen_geiger
import FLiESANN
from geos5fp import GEOS5FP, FailedGEOS5FPDownload
from sun_angles import calculate_SZA_from_DOY_and_hour
from ECOv002_granules import L2TLSTE, L2TSTARS, L3TJET, L3TSM, L3TSEB, L3TMET, L4TESI, L4TWUE
from ECOv002_granules import ET_COLORMAP, SM_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP, RH_COLORMAP, GPP_COLORMAP

from .exit_codes import *
from .BESS.BESS import BESS
from .runconfig import read_runconfig, ECOSTRESSRunConfig
from .FLiES import BlankOutputError
from .FLiES.FLiESLUT import FLiESLUT
from .LPDAAC.LPDAACDataPool import LPDAACServerUnreachable
from .MCD12.MCD12C1 import MCD12C1
from .MOD16.MOD16 import MOD16
from .PTJPL import PTJPL
from .PTJPLSM import PTJPLSM
from .STIC import STIC
from .downscaling.linear_downscale import linear_downscale, bias_correct
from .model.model import check_distribution
from .timer import Timer

from .PGEVersion import PGEVersion

with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read()

__version__ = version

# constant latent heat of vaporization for water: the number of joules of energy it takes to evaporate one kilogram
LATENT_VAPORIZATION_JOULES_PER_KILOGRAM = 2450000.0

L3T_L4T_JET_TEMPLATE = join(abspath(dirname(__file__)), "L3T_L4T_JET.xml")
DEFAULT_BUILD = "0700"
DEFAULT_OUTPUT_DIRECTORY = "L3T_L4T_JET_output"
DEFAULT_PTJPL_SOURCES_DIRECTORY = "L3T_L4T_JET_sources"
DEFAULT_STATIC_DIRECTORY = "L3T_L4T_static"
DEFAULT_SRTM_DIRECTORY = "SRTM_directory"
DEFAULT_GEDI_DIRECTORY = "GEDI_download"
DEFAULT_MODISCI_DIRECTORY = "MODISCI_download"
DEFAULT_MCD12C1_DIRECTORY = "MCD12C1_download"
DEFAULT_SOIL_GRIDS_DIRECTORY = "SoilGrids_download"
DEFAULT_GEOS5FP_DIRECTORY = "GEOS5FP_download"

L3T_SEB_SHORT_NAME = "ECO_L3T_SEB"
L3T_SEB_LONG_NAME = "ECOSTRESS Tiled Surface Energy Balance Instantaneous L3 Global 70 m"

L3T_SM_SHORT_NAME = "ECO_L3T_SM"
L3T_SM_LONG_NAME = "ECOSTRESS Tiled Downscaled Soil Moisture Instantaneous L3 Global 70 m"

L3T_MET_SHORT_NAME = "ECO_L3T_MET"
L3T_MET_LONG_NAME = "ECOSTRESS Tiled Downscaled Meteorology Instantaneous L3 Global 70 m"

L3T_JET_SHORT_NAME = "ECO_L3T_JET"
L3T_JET_LONG_NAME = "ECOSTRESS Tiled Evapotranspiration Ensemble Instantaneous and Daytime L3 Global 70 m"

L4T_ESI_SHORT_NAME = "ECO_L4T_ESI"
L4T_ESI_LONG_NAME = "ECOSTRESS Tiled Evaporative Stress Index Instantaneous L4 Global 70 m"

L4T_WUE_SHORT_NAME = "ECO_L4T_WUE"
L4T_WUE_LONG_NAME = "ECOSTRESS Tiled Water Use Efficiency Instantaneous L4 Global 70 m"

GEOS_IN_SENTINEL_COARSE_CELL_SIZE = 13720

STRIP_CONSOLE = False
SAVE_INTERMEDIATE = False
SHOW_DISTRIBUTION = True
INCLUDE_SEB_DIAGNOSTICS = False
INCLUDE_JET_DIAGNOSTICS = False
BIAS_CORRECT_FLIES_ANN = True
SHARPEN_METEOROLOGY = True
SHARPEN_SOIL_MOISTURE = True

# SWIN_MODEL_NAME = "FLiES-ANN"
SWIN_MODEL_NAME = "GEOS5FP"
RN_MODEL_NAME = "verma"
FLOOR_TOPT = True

SZA_DEGREE_CUTOFF = 90

logger = logging.getLogger(__name__)


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


def calculate_UTC_offset_hours(geometry: RasterGeometry) -> Raster:
    return Raster(np.radians(geometry.lon) / np.pi * 12, geometry=geometry)


def calculate_day_of_year(time_UTC: datetime, geometry: RasterGeometry) -> Raster:
    doy_UTC = time_UTC.timetuple().tm_yday
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = calculate_UTC_offset_hours(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    doy = doy_UTC
    doy = rt.where(hour_of_day < 0, doy - 1, doy)
    doy = rt.where(hour_of_day > 24, doy + 1, doy)

    return doy


def calculate_hour_of_day(time_UTC: datetime, geometry: RasterGeometry) -> Raster:
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = calculate_UTC_offset_hours(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    hour_of_day = rt.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = rt.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day


def day_angle_rad_from_doy(doy: Raster) -> Raster:
    """
    This function calculates day angle in radians from day of year between 1 and 365.
    """
    return (2 * np.pi * (doy - 1)) / 365


def solar_dec_deg_from_day_angle_rad(day_angle_rad: float) -> float:
    """
    This function calculates solar declination in degrees from day angle in radians.
    """
    return (0.006918 - 0.399912 * np.cos(day_angle_rad) + 0.070257 * np.sin(day_angle_rad) - 0.006758 * np.cos(
        2 * day_angle_rad) + 0.000907 * np.sin(2 * day_angle_rad) - 0.002697 * np.cos(
        3 * day_angle_rad) + 0.00148 * np.sin(
        3 * day_angle_rad)) * (180 / np.pi)


def SHA_deg_from_doy_lat(doy: Raster, latitude: np.ndarray) -> Raster:
    """
    This function calculates sunrise hour angle in degrees from latitude in degrees and day of year between 1 and 365.
    """
    # calculate day angle in radians
    day_angle_rad = day_angle_rad_from_doy(doy)

    # calculate solar declination in degrees
    solar_dec_deg = solar_dec_deg_from_day_angle_rad(day_angle_rad)

    # convert latitude to radians
    latitude_rad = np.radians(latitude)

    # convert solar declination to radians
    solar_dec_rad = np.radians(solar_dec_deg)

    # calculate cosine of sunrise angle at latitude and solar declination
    # need to keep the cosine for polar correction
    sunrise_cos = -np.tan(latitude_rad) * np.tan(solar_dec_rad)

    # calculate sunrise angle in radians from cosine
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sunrise_rad = np.arccos(sunrise_cos)

    # convert to degrees
    sunrise_deg = np.degrees(sunrise_rad)

    # apply polar correction
    sunrise_deg = rt.where(sunrise_cos >= 1, 0, sunrise_deg)
    sunrise_deg = rt.where(sunrise_cos <= -1, 180, sunrise_deg)

    return sunrise_deg


def sunrise_from_sha(sha_deg: Raster) -> Raster:
    """
    This function calculates sunrise hour from sunrise hour angle in degrees.
    """
    return 12.0 - (sha_deg / 15.0)


def daylight_from_sha(sha_deg: Raster) -> Raster:
    """
    This function calculates daylight hours from sunrise hour angle in degrees.
    """
    return (2.0 / 15.0) * sha_deg


def daily_Rn_integration_verma(
        Rn: Raster,
        hour_of_day: Raster,
        sunrise_hour: Raster,
        daylight_hours: Raster) -> Raster:
    """
    calculate daily net radiation using solar parameters
    this is the average rate of energy transfer from sunrise to sunset
    in watts per square meter
    watts are joules per second
    to get the total amount of energy transferred, factor seconds out of joules
    the number of seconds for which this average is representative is (daylight_hours * 3600)
    documented in verma et al, bisht et al, and lagouARDe et al
    :param Rn:
    :param hour_of_day:
    :param sunrise_hour:
    :param daylight_hours:
    :return:
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return 1.6 * Rn / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))


def generate_L3T_L4T_JET_runconfig(
        L2T_LSTE_filename: str,
        L2T_STARS_filename: str,
        orbit: int = None,
        scene: int = None,
        tile: str = None,
        working_directory: str = None,
        sources_directory: str = None,
        static_directory: str = None,
        SRTM_directory: str = None,
        executable_filename: str = None,
        output_directory: str = None,
        runconfig_filename: str = None,
        log_filename: str = None,
        build: str = None,
        processing_node: str = None,
        production_datetime: datetime = None,
        job_ID: str = None,
        instance_ID: str = None,
        product_counter: int = None,
        template_filename: str = None) -> str:
    L2T_LSTE_granule = L2TLSTE(L2T_LSTE_filename)

    if orbit is None:
        orbit = L2T_LSTE_granule.orbit

    if scene is None:
        scene = L2T_LSTE_granule.scene

    if tile is None:
        tile = L2T_LSTE_granule.tile

    if template_filename is None:
        template_filename = L3T_L4T_JET_TEMPLATE

    template_filename = abspath(expanduser(template_filename))

    if build is None:
        build = DEFAULT_BUILD

    if product_counter is None:
        product_counter = 1

    time_UTC = L2T_LSTE_granule.time_UTC
    timestamp = f"{time_UTC:%Y%m%dT%H%M%S}"
    granule_ID = f"ECOv002_L3T_JET_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"

    if runconfig_filename is None:
        runconfig_filename = join(working_directory, "runconfig", f"{granule_ID}.xml")

    runconfig_filename = abspath(expanduser(runconfig_filename))

    if working_directory is None:
        working_directory = granule_ID

    working_directory = abspath(expanduser(working_directory))

    if executable_filename is None:
        executable_filename = which("L3T_L4T_JET")

    if executable_filename is None:
        executable_filename = "L3T_L4T_JET"

    if output_directory is None:
        output_directory = join(working_directory, DEFAULT_OUTPUT_DIRECTORY)

    output_directory = abspath(expanduser(output_directory))

    if sources_directory is None:
        sources_directory = join(working_directory, DEFAULT_PTJPL_SOURCES_DIRECTORY)

    sources_directory = abspath(expanduser(sources_directory))

    if static_directory is None:
        static_directory = join(working_directory, DEFAULT_STATIC_DIRECTORY)

    static_directory = abspath(expanduser(static_directory))

    if SRTM_directory is None:
        SRTM_directory = join(working_directory, DEFAULT_SRTM_DIRECTORY)

    SRTM_directory = abspath(expanduser(SRTM_directory))

    if log_filename is None:
        log_filename = join(working_directory, "log", f"{granule_ID}.log")

    log_filename = abspath(expanduser(log_filename))

    if processing_node is None:
        processing_node = socket.gethostname()

    if production_datetime is None:
        production_datetime = datetime.utcnow()

    if isinstance(production_datetime, datetime):
        production_datetime = str(production_datetime)

    if job_ID is None:
        job_ID = production_datetime

    if instance_ID is None:
        instance_ID = str(uuid4())

    L2T_LSTE_filename = abspath(expanduser(str(L2T_LSTE_filename)))
    L2T_STARS_filename = abspath(expanduser(str(L2T_STARS_filename)))
    # working_directory = abspath(expanduser(str(working_directory)))
    # sources_directory = abspath(expanduser(str(sources_directory)))
    # static_directory = abspath(expanduser(str(static_directory)))
    # SRTM_directory = abspath(expanduser(str(SRTM_directory)))

    logger.info(f"generating run-config for orbit {cl.val(orbit)} scene {cl.val(scene)}")
    logger.info(f"loading L3T_L4T_JET template: {cl.file(template_filename)}")

    with open(template_filename, "r") as file:
        template = file.read()

    logger.info(f"orbit: {cl.val(orbit)}")
    template = template.replace("orbit_number", f"{orbit:05d}")
    logger.info(f"scene: {cl.val(scene)}")
    template = template.replace("scene_ID", f"{scene:03d}")
    logger.info(f"tile: {cl.val(tile)}")
    template = template.replace("tile_ID", f"{tile}")
    logger.info(f"L2T_LSTE file: {cl.file(L2T_LSTE_filename)}")
    template = template.replace("L2T_LSTE_filename", L2T_LSTE_filename)
    logger.info(f"L2T_STARS file: {cl.file(L2T_STARS_filename)}")
    template = template.replace("L2T_STARS_filename", L2T_STARS_filename)
    logger.info(f"working directory: {cl.dir(working_directory)}")
    template = template.replace("working_directory", working_directory)
    logger.info(f"sources directory: {cl.dir(sources_directory)}")
    template = template.replace("sources_directory", sources_directory)
    logger.info(f"static directory: {cl.dir(static_directory)}")
    template = template.replace("static_directory", static_directory)
    logger.info(f"executable: {cl.file(executable_filename)}")
    template = template.replace("executable_filename", executable_filename)
    logger.info(f"output directory: {cl.dir(output_directory)}")
    template = template.replace("output_directory", output_directory)
    logger.info(f"run-config: {cl.file(runconfig_filename)}")
    template = template.replace("runconfig_filename", runconfig_filename)
    logger.info(f"log: {cl.file(log_filename)}")
    template = template.replace("log_filename", log_filename)
    logger.info(f"build: {cl.val(build)}")
    template = template.replace("build_ID", build)
    logger.info(f"processing node: {cl.val(processing_node)}")
    template = template.replace("processing_node", processing_node)
    logger.info(f"production date/time: {cl.time(production_datetime)}")
    template = template.replace("production_datetime", production_datetime)
    logger.info(f"job ID: {cl.val(job_ID)}")
    template = template.replace("job_ID", job_ID)
    logger.info(f"instance ID: {cl.val(instance_ID)}")
    template = template.replace("instance_ID", instance_ID)
    logger.info(f"product counter: {cl.val(product_counter)}")
    template = template.replace("product_counter", f"{product_counter:02d}")

    makedirs(dirname(abspath(runconfig_filename)), exist_ok=True)
    logger.info(f"writing run-config file: {cl.file(runconfig_filename)}")

    with open(runconfig_filename, "w") as file:
        file.write(template)

    return runconfig_filename


class L3TL4TJETConfig(ECOSTRESSRunConfig):
    def __init__(self, filename: str):
        try:
            logger.info(f"loading L3T_L4T_JET run-config: {cl.file(filename)}")
            runconfig = read_runconfig(filename)

            # print(JSON_highlight(runconfig))

            if "StaticAncillaryFileGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"missing StaticAncillaryFileGroup in L3T_L4T_JET run-config: {filename}")

            if "L3T_L4T_JET_WORKING" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing StaticAncillaryFileGroup/L3T_L4T_JET_WORKING in L3T_L4T_JET run-config: {filename}")

            working_directory = abspath(runconfig["StaticAncillaryFileGroup"]["L3T_L4T_JET_WORKING"])
            logger.info(f"working directory: {cl.dir(working_directory)}")

            if "L3T_L4T_JET_SOURCES" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing StaticAncillaryFileGroup/L3T_L4T_JET_WORKING in L3T_L4T_JET run-config: {filename}")

            sources_directory = abspath(runconfig["StaticAncillaryFileGroup"]["L3T_L4T_JET_SOURCES"])
            logger.info(f"sources directory: {cl.dir(sources_directory)}")

            GEOS5FP_directory = join(sources_directory, DEFAULT_GEOS5FP_DIRECTORY)

            if "L3T_L4T_STATIC" not in runconfig["StaticAncillaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing StaticAncillaryFileGroup/L3T_L4T_STATIC in L3T_L4T_JET run-config: {filename}")

            static_directory = abspath(runconfig["StaticAncillaryFileGroup"]["L3T_L4T_STATIC"])
            logger.info(f"static directory: {cl.dir(static_directory)}")

            if "ProductPathGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"missing ProductPathGroup in L3T_L4T_JET run-config: {filename}")

            if "ProductPath" not in runconfig["ProductPathGroup"]:
                raise MissingRunConfigValue(
                    f"missing ProductPathGroup/ProductPath in L3T_L4T_JET run-config: {filename}")

            output_directory = abspath(runconfig["ProductPathGroup"]["ProductPath"])
            logger.info(f"output directory: {cl.dir(output_directory)}")

            if "InputFileGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"missing InputFileGroup in L3T_L4T_JET run-config: {filename}")

            if "L2T_LSTE" not in runconfig["InputFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing InputFileGroup/L2T_LSTE in L3T_L4T_JET run-config: {filename}")

            L2T_LSTE_filename = abspath(runconfig["InputFileGroup"]["L2T_LSTE"])
            logger.info(f"L2T_LSTE file: {cl.file(L2T_LSTE_filename)}")

            if "L2T_STARS" not in runconfig["InputFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing InputFileGroup/L2T_STARS in L3T_L4T_JET run-config: {filename}")

            L2T_STARS_filename = abspath(runconfig["InputFileGroup"]["L2T_STARS"])
            logger.info(f"L2T_STARS file: {cl.file(L2T_STARS_filename)}")

            orbit = int(runconfig["Geometry"]["OrbitNumber"])
            logger.info(f"orbit: {cl.val(orbit)}")

            if "SceneId" not in runconfig["Geometry"]:
                raise MissingRunConfigValue(
                    f"missing Geometry/SceneId in L2T_STARS run-config: {filename}")

            scene = int(runconfig["Geometry"]["SceneId"])
            logger.info(f"scene: {cl.val(scene)}")

            if "TileId" not in runconfig["Geometry"]:
                raise MissingRunConfigValue(
                    f"missing Geometry/TileId in L2T_STARS run-config: {filename}")

            tile = str(runconfig["Geometry"]["TileId"])
            logger.info(f"tile: {cl.val(tile)}")

            if "BuildID" not in runconfig["PrimaryExecutable"]:
                raise MissingRunConfigValue(f"missing PrimaryExecutable/BuildID in L2G_L2T_LSTE run-config {filename}")

            build = str(runconfig["PrimaryExecutable"]["BuildID"])

            if "ProductCounter" not in runconfig["ProductPathGroup"]:
                raise MissingRunConfigValue(
                    f"missing ProductPathGroup/ProductCounter in L2G_L2T_LSTE run-config {filename}")

            product_counter = int(runconfig["ProductPathGroup"]["ProductCounter"])

            L2T_LSTE_granule = L2TLSTE(L2T_LSTE_filename)
            time_UTC = L2T_LSTE_granule.time_UTC
            timestamp = f"{time_UTC:%Y%m%dT%H%M%S}"
            granule_ID = f"ECOv002_L3T_JET_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"

            GEDI_directory = abspath(expanduser(join(static_directory, DEFAULT_GEDI_DIRECTORY)))
            MODISCI_directory = abspath(expanduser(join(static_directory, DEFAULT_MODISCI_DIRECTORY)))
            MCD12_directory = abspath(expanduser(join(static_directory, DEFAULT_MCD12C1_DIRECTORY)))
            soil_grids_directory = abspath(expanduser(join(static_directory, DEFAULT_SOIL_GRIDS_DIRECTORY)))

            L3T_JET_granule_ID = f"ECOv002_L3T_JET_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L3T_JET_directory = join(output_directory, L3T_JET_granule_ID)
            L3T_JET_zip_filename = f"{L3T_JET_directory}.zip"
            L3T_JET_browse_filename = f"{L3T_JET_directory}.png"

            L3T_BESS_granule_ID = f"ECOv002_L3T_BESS_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L3T_BESS_directory = join(output_directory, L3T_BESS_granule_ID)
            L3T_BESS_zip_filename = f"{L3T_BESS_directory}.zip"
            L3T_BESS_browse_filename = f"{L3T_BESS_directory}.png"

            L3T_MET_granule_ID = f"ECOv002_L3T_MET_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L3T_MET_directory = join(output_directory, L3T_MET_granule_ID)
            L3T_MET_zip_filename = f"{L3T_MET_directory}.zip"
            L3T_MET_browse_filename = f"{L3T_MET_directory}.png"

            L3T_SEB_granule_ID = f"ECOv002_L3T_SEB_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L3T_SEB_directory = join(output_directory, L3T_SEB_granule_ID)
            L3T_SEB_zip_filename = f"{L3T_SEB_directory}.zip"
            L3T_SEB_browse_filename = f"{L3T_SEB_directory}.png"

            L3T_SM_granule_ID = f"ECOv002_L3T_SM_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L3T_SM_directory = join(output_directory, L3T_SM_granule_ID)
            L3T_SM_zip_filename = f"{L3T_SM_directory}.zip"
            L3T_SM_browse_filename = f"{L3T_SM_directory}.png"

            L4T_ESI_granule_ID = f"ECOv002_L4T_ESI_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L4T_ESI_directory = join(output_directory, L4T_ESI_granule_ID)
            L4T_ESI_zip_filename = f"{L4T_ESI_directory}.zip"
            L4T_ESI_browse_filename = f"{L4T_ESI_directory}.png"

            L4T_WUE_granule_ID = f"ECOv002_L4T_WUE_{orbit:05d}_{scene:03d}_{tile}_{timestamp}_{build}_{product_counter:02d}"
            L4T_WUE_directory = join(output_directory, L4T_WUE_granule_ID)
            L4T_WUE_zip_filename = f"{L4T_WUE_directory}.zip"
            L4T_WUE_browse_filename = f"{L4T_WUE_directory}.png"

            PGE_name = "L3T_L4T_JET"
            PGE_version = PGEVersion

            self.working_directory = working_directory
            self.sources_directory = sources_directory
            self.GEOS5FP_directory = GEOS5FP_directory
            self.static_directory = static_directory
            self.GEDI_directory = GEDI_directory
            self.MODISCI_directory = MODISCI_directory
            self.MCD12_directory = MCD12_directory
            self.soil_grids_directory = soil_grids_directory
            self.output_directory = output_directory
            self.L2T_LSTE_filename = L2T_LSTE_filename
            self.L2T_STARS_filename = L2T_STARS_filename
            self.orbit = orbit
            self.scene = scene
            self.tile = tile
            self.build = build
            self.product_counter = product_counter
            self.granule_ID = granule_ID

            self.L3T_JET_granule_ID = L3T_JET_granule_ID
            self.L3T_JET_directory = L3T_JET_directory
            self.L3T_JET_zip_filename = L3T_JET_zip_filename
            self.L3T_JET_browse_filename = L3T_JET_browse_filename

            self.L3T_BESS_granule_ID = L3T_BESS_granule_ID
            self.L3T_BESS_directory = L3T_BESS_directory
            self.L3T_BESS_zip_filename = L3T_BESS_zip_filename
            self.L3T_BESS_browse_filename = L3T_BESS_browse_filename

            self.L3T_MET_granule_ID = L3T_MET_granule_ID
            self.L3T_MET_directory = L3T_MET_directory
            self.L3T_MET_zip_filename = L3T_MET_zip_filename
            self.L3T_MET_browse_filename = L3T_MET_browse_filename

            self.L3T_SEB_granule_ID = L3T_SEB_granule_ID
            self.L3T_SEB_directory = L3T_SEB_directory
            self.L3T_SEB_zip_filename = L3T_SEB_zip_filename
            self.L3T_SEB_browse_filename = L3T_SEB_browse_filename

            self.L3T_SM_granule_ID = L3T_SM_granule_ID
            self.L3T_SM_directory = L3T_SM_directory
            self.L3T_SM_zip_filename = L3T_SM_zip_filename
            self.L3T_SM_browse_filename = L3T_SM_browse_filename

            self.L4T_WUE_granule_ID = L4T_WUE_granule_ID
            self.L4T_WUE_directory = L4T_WUE_directory
            self.L4T_WUE_zip_filename = L4T_WUE_zip_filename
            self.L4T_WUE_browse_filename = L4T_WUE_browse_filename

            self.L4T_ESI_granule_ID = L4T_ESI_granule_ID
            self.L4T_ESI_directory = L4T_ESI_directory
            self.L4T_ESI_zip_filename = L4T_ESI_zip_filename
            self.L4T_ESI_browse_filename = L4T_ESI_browse_filename

            self.PGE_name = PGE_name
            self.PGE_version = PGE_version
        except MissingRunConfigValue as e:
            raise e
        except ECOSTRESSExitCodeException as e:
            raise e
        except Exception as e:
            logger.exception(e)
            raise UnableToParseRunConfig(f"unable to parse run-config file: {filename}")


def L3T_L4T_JET(
        runconfig_filename: str,
        upsampling: str = None,
        downsampling: str = None,
        SWin_model_name: str = SWIN_MODEL_NAME,
        Rn_model_name: str = RN_MODEL_NAME,
        include_SEB_diagnostics: bool = INCLUDE_SEB_DIAGNOSTICS,
        include_JET_diagnostics: bool = INCLUDE_JET_DIAGNOSTICS,
        bias_correct_FLiES_ANN: bool = BIAS_CORRECT_FLIES_ANN,
        sharpen_meteorology: bool = SHARPEN_METEOROLOGY,
        sharpen_soil_moisture: bool = SHARPEN_SOIL_MOISTURE,
        strip_console: bool = STRIP_CONSOLE,
        save_intermediate: bool = SAVE_INTERMEDIATE,
        show_distribution: bool = SHOW_DISTRIBUTION,
        floor_Topt: bool = FLOOR_TOPT) -> int:
    """
    ECOSTRESS Collection 2 L3T L4T JET PGE
    :param runconfig_filename: filename for XML run-config
    :param log_filename: filename for logger output
    :return: exit code number
    """
    exit_code = SUCCESS_EXIT_CODE

    if upsampling is None:
        upsampling = "average"

    if downsampling is None:
        downsampling = "linear"

    try:
        runconfig = L3TL4TJETConfig(runconfig_filename)
        working_directory = runconfig.working_directory
        granule_ID = runconfig.granule_ID
        log_filename = join(working_directory, "log", f"{granule_ID}.log")
        cl.configure(filename=log_filename, strip_console=strip_console)
        timer = Timer()
        logger.info(f"started L3T L4T JET run at {cl.time(datetime.utcnow())} UTC")
        logger.info(f"L3T_L4T_JET PGE ({cl.val(runconfig.PGE_version)})")
        logger.info(f"L3T_L4T_JET run-config: {cl.file(runconfig_filename)}")

        L3T_JET_granule_ID = runconfig.L3T_JET_granule_ID
        logger.info(f"L3T JET granule ID: {cl.val(L3T_JET_granule_ID)}")

        L3T_JET_directory = runconfig.L3T_JET_directory
        logger.info(f"L3T JET granule directory: {cl.dir(L3T_JET_directory)}")
        L3T_JET_zip_filename = runconfig.L3T_JET_zip_filename
        logger.info(f"L3T JET zip file: {cl.file(L3T_JET_zip_filename)}")
        L3T_JET_browse_filename = runconfig.L3T_JET_browse_filename
        logger.info(f"L3T JET preview: {cl.file(L3T_JET_browse_filename)}")

        L3T_BESS_directory = runconfig.L3T_BESS_directory
        logger.info(f"L3T BESS granule directory: {cl.dir(L3T_BESS_directory)}")
        L3T_BESS_zip_filename = runconfig.L3T_BESS_zip_filename
        logger.info(f"L3T BESS zip file: {cl.file(L3T_BESS_zip_filename)}")
        L3T_BESS_browse_filename = runconfig.L3T_BESS_browse_filename
        logger.info(f"L3T BESS preview: {cl.file(L3T_BESS_browse_filename)}")

        L3T_MET_directory = runconfig.L3T_MET_directory
        logger.info(f"L3T MET granule directory: {cl.dir(L3T_MET_directory)}")
        L3T_MET_zip_filename = runconfig.L3T_MET_zip_filename
        logger.info(f"L3T MET zip file: {cl.file(L3T_MET_zip_filename)}")
        L3T_MET_browse_filename = runconfig.L3T_MET_browse_filename
        logger.info(f"L3T MET preview: {cl.file(L3T_MET_browse_filename)}")

        L3T_SEB_directory = runconfig.L3T_SEB_directory
        logger.info(f"L3T SEB granule directory: {cl.dir(L3T_SEB_directory)}")
        L3T_SEB_zip_filename = runconfig.L3T_SEB_zip_filename
        logger.info(f"L3T SEB zip file: {cl.file(L3T_SEB_zip_filename)}")
        L3T_SEB_browse_filename = runconfig.L3T_SEB_browse_filename
        logger.info(f"L3T SEB preview: {cl.file(L3T_SEB_browse_filename)}")

        L3T_SM_directory = runconfig.L3T_SM_directory
        logger.info(f"L3T SM granule directory: {cl.dir(L3T_SM_directory)}")
        L3T_SM_zip_filename = runconfig.L3T_SM_zip_filename
        logger.info(f"L3T SM zip file: {cl.file(L3T_SM_zip_filename)}")
        L3T_SM_browse_filename = runconfig.L3T_SM_browse_filename
        logger.info(f"L3T SM preview: {cl.file(L3T_SM_browse_filename)}")

        L4T_ESI_granule_ID = runconfig.L4T_ESI_granule_ID
        logger.info(f"L4T ESI PT-JPL granule ID: {cl.val(L4T_ESI_granule_ID)}")
        L4T_ESI_directory = runconfig.L4T_ESI_directory
        logger.info(f"L4T ESI PT-JPL granule directory: {cl.dir(L4T_ESI_directory)}")
        L4T_ESI_zip_filename = runconfig.L4T_ESI_zip_filename
        logger.info(f"L4T ESI PT-JPL zip file: {cl.file(L4T_ESI_zip_filename)}")
        L4T_ESI_browse_filename = runconfig.L4T_ESI_browse_filename
        logger.info(f"L4T ESI PT-JPL preview: {cl.file(L4T_ESI_browse_filename)}")

        L4T_WUE_granule_ID = runconfig.L4T_WUE_granule_ID
        logger.info(f"L4T WUE granule ID: {cl.val(L4T_WUE_granule_ID)}")
        L4T_WUE_directory = runconfig.L4T_WUE_directory
        logger.info(f"L4T WUE granule directory: {cl.dir(L4T_WUE_directory)}")
        L4T_WUE_zip_filename = runconfig.L4T_WUE_zip_filename
        logger.info(f"L4T WUE zip file: {cl.file(L4T_WUE_zip_filename)}")
        L4T_WUE_browse_filename = runconfig.L4T_WUE_browse_filename
        logger.info(f"L4T WUE preview: {cl.file(L4T_WUE_browse_filename)}")

        required_files = [
            L3T_JET_zip_filename,
            L3T_JET_browse_filename,
            L3T_MET_zip_filename,
            L3T_MET_browse_filename,
            L3T_SEB_zip_filename,
            L3T_SEB_browse_filename,
            L3T_SM_zip_filename,
            L3T_SM_browse_filename,
            L4T_ESI_zip_filename,
            L4T_ESI_browse_filename,
            L4T_WUE_zip_filename,
            L4T_WUE_browse_filename
        ]

        some_files_missing = False

        for filename in required_files:
            if exists(filename):
                logger.info(f"found product file: {cl.file(filename)}")
            else:
                logger.info(f"product file not found: {cl.file(filename)}")
                some_files_missing = True

        if not some_files_missing:
            logger.info("L3T_L4T_JET output already found")
            return SUCCESS_EXIT_CODE

        logger.info(f"working_directory: {cl.dir(working_directory)}")
        output_directory = runconfig.output_directory
        logger.info(f"output directory: {cl.dir(output_directory)}")
        sources_directory = runconfig.sources_directory
        logger.info(f"sources directory: {cl.dir(sources_directory)}")
        GEOS5FP_directory = runconfig.GEOS5FP_directory
        logger.info(f"GEOS-5 FP directory: {cl.dir(GEOS5FP_directory)}")
        static_directory = runconfig.static_directory
        logger.info(f"static directory: {cl.dir(static_directory)}")
        GEDI_directory = runconfig.GEDI_directory
        logger.info(f"GEDI directory: {cl.dir(GEDI_directory)}")
        MODISCI_directory = runconfig.MODISCI_directory
        logger.info(f"MODIS CI directory: {cl.dir(MODISCI_directory)}")
        MCD12_directory = runconfig.MCD12_directory
        logger.info(f"MCD12C1 IGBP directory: {cl.dir(MCD12_directory)}")
        soil_grids_directory = runconfig.soil_grids_directory
        logger.info(f"SoilGrids directory: {cl.dir(soil_grids_directory)}")
        logger.info(f"log: {cl.file(log_filename)}")
        orbit = runconfig.orbit
        logger.info(f"orbit: {cl.val(orbit)}")
        scene = runconfig.scene
        logger.info(f"scene: {cl.val(scene)}")
        tile = runconfig.tile
        logger.info(f"tile: {cl.val(tile)}")
        build = runconfig.build
        logger.info(f"build: {cl.val(build)}")
        product_counter = runconfig.product_counter
        logger.info(f"product counter: {cl.val(product_counter)}")
        L2T_LSTE_filename = runconfig.L2T_LSTE_filename
        logger.info(f"L2T_LSTE file: {cl.file(L2T_LSTE_filename)}")
        L2T_STARS_filename = runconfig.L2T_STARS_filename
        logger.info(f"L2T_STARS file: {cl.file(L2T_STARS_filename)}")

        if not exists(L2T_LSTE_filename):
            raise InputFilesInaccessible(f"L2T LSTE file does not exist: {L2T_LSTE_filename}")

        L2T_LSTE_granule = L2TLSTE(L2T_LSTE_filename)

        if not exists(L2T_STARS_filename):
            raise InputFilesInaccessible(f"L2T STARS file does not exist: {L2T_STARS_filename}")

        L2T_STARS_granule = L2TSTARS(L2T_STARS_filename)

        metadata = L2T_STARS_granule.metadata_dict
        metadata["StandardMetadata"]["PGEVersion"] = PGEVersion
        metadata["StandardMetadata"]["PGEName"] = "L3T_L4T_JET"
        metadata["StandardMetadata"]["ProcessingLevelID"] = "L3T"
        metadata["StandardMetadata"]["SISName"] = "Level 3 Product Specification Document"
        metadata["StandardMetadata"]["SISVersion"] = "Preliminary"
        metadata["StandardMetadata"]["AncillaryInputPointer"] = "AncillaryNWP"

        geometry = L2T_LSTE_granule.geometry
        time_UTC = L2T_LSTE_granule.time_UTC
        date_UTC = time_UTC.date()
        time_solar = L2T_LSTE_granule.time_solar
        logger.info(
            f"orbit {cl.val(orbit)} scene {cl.val(scene)} tile {cl.place(tile)} overpass time: {cl.time(time_UTC)} UTC ({cl.time(time_solar)} solar)")
        timestamp = f"{time_UTC:%Y%m%dT%H%M%S}"

        hour_of_day = calculate_hour_of_day(time_UTC=time_UTC, geometry=geometry)
        day_of_year = calculate_day_of_year(time_UTC=time_UTC, geometry=geometry)

        ST_K = L2T_LSTE_granule.ST_K

        logger.info(f"reading elevation from L2T LSTE: {L2T_LSTE_granule.product_filename}")
        elevation_km = L2T_LSTE_granule.elevation_km
        check_distribution(elevation_km, "elevation_km", date_UTC=date_UTC, target=tile)

        emissivity = L2T_LSTE_granule.emissivity
        water = L2T_LSTE_granule.water
        cloud = L2T_LSTE_granule.cloud
        NDVI = L2T_STARS_granule.NDVI
        albedo = L2T_STARS_granule.albedo

        percent_cloud = 100 * np.count_nonzero(cloud) / cloud.size
        metadata["ProductMetadata"]["QAPercentCloudCover"] = percent_cloud

        GEOS5FP_connection = GEOS5FP(
            working_directory=working_directory,
            download_directory=GEOS5FP_directory
        )

        PTJPLSM_model = PTJPLSM(
            working_directory=working_directory,
            GEDI_download=GEDI_directory,
            CI_directory=MODISCI_directory,
            soil_grids_download=soil_grids_directory,
            GEOS5FP_connection=GEOS5FP_connection,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution,
            floor_Topt=floor_Topt
        )

        PTJPL_model = PTJPL(
            working_directory=working_directory,
            GEDI_download=GEDI_directory,
            CI_directory=MODISCI_directory,
            GEOS5FP_connection=GEOS5FP_connection,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution
        )

        # FLiES_ANN_model = FLiES(
        #     working_directory=working_directory,
        #     GEOS5FP_connection=GEOS5FP_connection,
        #     save_intermediate=save_intermediate,
        #     show_distribution=show_distribution
        # )


        # MCD12_connnection = MCD12C1(
        #     working_directory=static_directory,
        #     download_directory=MCD12_directory
        # )

        # FIXME replace FLiESLUT sub-module with package
        FLiES_LUT_model = FLiESLUT(
            working_directory=working_directory,
            static_directory=static_directory,
            GEOS5FP_connection=GEOS5FP_connection,
            # MCD12_connnection=MCD12_connnection,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution
        )

        # FIXME replace BESS sub-module with package
        BESS_model = BESS(
            working_directory=working_directory,
            GEDI_download=GEDI_directory,
            CI_directory=MODISCI_directory,
            GEOS5FP_connection=GEOS5FP_connection,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution
        )

        # SZA = FLiES_ANN_model.SZA(day_of_year=day_of_year, hour_of_day=hour_of_day, geometry=geometry)
        SZA = calculate_SZA_from_DOY_and_hour(
            lat=geometry.lat,
            lon=geometry.lon,
            DOY=day_of_year,
            hour=hour_of_day
        )

        check_distribution(SZA, "SZA", date_UTC=date_UTC, target=tile)

        if np.all(SZA >= SZA_DEGREE_CUTOFF):
            raise DaytimeFilter(f"solar zenith angle exceeds {SZA_DEGREE_CUTOFF} for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        logger.info("retrieving GEOS-5 FP aerosol optical thickness raster")
        AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry)
        check_distribution(AOT, "AOT", date_UTC=date_UTC, target=tile)

        logger.info("generating GEOS-5 FP cloud optical thickness raster")
        COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry)
        check_distribution(COT, "COT", date_UTC=date_UTC, target=tile)

        logger.info("generating GEOS5-FP water vapor raster in grams per square centimeter")
        vapor_gccm = GEOS5FP_connection.vapor_gccm(time_UTC=time_UTC, geometry=geometry)
        check_distribution(vapor_gccm, "vapor_gccm", date_UTC=date_UTC, target=tile)

        logger.info("generating GEOS5-FP ozone raster in grams per square centimeter")
        ozone_cm = GEOS5FP_connection.ozone_cm(time_UTC=time_UTC, geometry=geometry)
        check_distribution(ozone_cm, "ozone_cm", date_UTC=date_UTC, target=tile)

        logger.info(f"running Forest Light Environmental Simulator for {cl.place(tile)} at {cl.time(time_UTC)} UTC")

        # Ra, SWin_FLiES_ANN_raw, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = FLiES_ANN_model.FLiES(
        #     geometry=geometry,
        #     target=tile,
        #     time_UTC=time_UTC,
        #     albedo=albedo,
        #     COT=COT,
        #     AOT=AOT,
        #     SZA=SZA,
        #     vapor_gccm=vapor_gccm,
        #     ozone_cm=ozone_cm,
        #     elevation_km=elevation_km
        # )

        doy_solar = time_solar.timetuple().tm_yday
        kg = load_koppen_geiger(albedo.geometry)

        FLiES_results = FLiESANN.process_FLiES(
            doy=doy_solar,
            albedo=albedo,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            elevation_km=elevation_km,
            SZA=SZA,
            KG_climate=kg
        )

        Ra = FLiES_results["Ra"]
        SWin_FLiES_ANN_raw = FLiES_results["Rg"]
        UV = FLiES_results["UV"]
        VIS = FLiES_results["VIS"]
        NIR = FLiES_results["NIR"]
        VISdiff = FLiES_results["VISdiff"]
        NIRdiff = FLiES_results["NIRdiff"]
        VISdir = FLiES_results["VISdir"]
        NIRdir = FLiES_results["NIRdir"]

        SWin_FLiES_LUT = FLiES_LUT_model.FLiES_LUT(
            geometry=geometry,
            target=tile,
            time_UTC=time_UTC,
            cloud_mask=cloud,
            COT=COT,
            albedo=albedo,
            AOT=AOT
        )

        # Rg = Rg.mask(~np.isnan(ST_K))
        # check_distribution(Rg, "Rg", date_UTC=date_UTC, target=tile)
        coarse_geometry = geometry.rescale(GEOS_IN_SENTINEL_COARSE_CELL_SIZE)

        SWin_coarse = GEOS5FP_connection.SWin(
            time_UTC=time_UTC,
            geometry=coarse_geometry,
            resampling=downsampling
        )

        if bias_correct_FLiES_ANN:
            SWin_FLiES_ANN = bias_correct(
                coarse_image=SWin_coarse,
                fine_image=SWin_FLiES_ANN_raw,
                upsampling=upsampling,
                downsampling=downsampling
            )
        else:
            SWin_FLiES_ANN = SWin_FLiES_ANN_raw

        check_distribution(SWin_FLiES_ANN, "SWin_FLiES_ANN", date_UTC=date_UTC, target=tile)

        SWin_GEOS5FP = GEOS5FP_connection.SWin(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=downsampling
        )

        check_distribution(SWin_GEOS5FP, "SWin_GEOS5FP", date_UTC=date_UTC, target=tile)

        if SWin_model_name == "GEOS5FP":
            SWin = SWin_GEOS5FP
        elif SWin_model_name == "FLiES-ANN":
            SWin = SWin_FLiES_ANN
        elif SWin_model_name == "FLiES-LUT":
            SWin = SWin_FLiES_LUT
        else:
            raise ValueError(f"unrecognized solar radiation model: {SWin_model_name}")

        SWin = rt.where(np.isnan(ST_K), np.nan, SWin)

        if np.all(np.isnan(SWin)) or np.all(SWin == 0):
            raise BlankOutput(f"blank solar radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        ST_C = ST_K - 273.15

        NDVI_coarse = NDVI.to_geometry(coarse_geometry, resampling=upsampling)
        albedo_coarse = albedo.to_geometry(coarse_geometry, resampling=upsampling)

        if sharpen_meteorology:
            ST_C_coarse = ST_C.to_geometry(coarse_geometry, resampling=upsampling)
            Ta_C_coarse = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling)
            Td_C_coarse = GEOS5FP_connection.Td_C(time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling)
            SM_coarse = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling)

            coarse_samples = pd.DataFrame({
                "Ta_C": np.array(Ta_C_coarse).ravel(),
                "Td_C": np.array(Td_C_coarse).ravel(),
                "SM": np.array(SM_coarse).ravel(),
                "ST_C": np.array(ST_C_coarse).ravel(),
                "NDVI": np.array(NDVI_coarse).ravel(),
                "albedo": np.array(albedo_coarse).ravel()
            })

            coarse_samples = coarse_samples.dropna()

            Ta_C_model = sklearn.linear_model.LinearRegression()
            Ta_C_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["Ta_C"])
            Ta_C_intercept = Ta_C_model.intercept_
            ST_C_Ta_C_coef, NDVI_Ta_C_coef, albedo_Ta_C_coef = Ta_C_model.coef_
            logger.info(
                f"air temperature regression: Ta_C = {Ta_C_intercept:0.2f} + {ST_C_Ta_C_coef:0.2f} * ST_C + {NDVI_Ta_C_coef:0.2f} * NDVI + {albedo_Ta_C_coef:0.2f} * albedo")
            Ta_C_prediction = ST_C * ST_C_Ta_C_coef + NDVI * NDVI_Ta_C_coef + albedo * albedo_Ta_C_coef + Ta_C_intercept
            check_distribution(Ta_C_prediction, "Ta_C_prediction", date_UTC, tile)
            logger.info(
                f"up-sampling predicted air temperature from {int(Ta_C_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            Ta_C_prediction_coarse = Ta_C_prediction.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(Ta_C_prediction_coarse, "Ta_C_prediction_coarse", date_UTC, tile)
            Ta_C_bias_coarse = Ta_C_prediction_coarse - Ta_C_coarse
            check_distribution(Ta_C_bias_coarse, "Ta_C_bias_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling air temperature bias from {int(Ta_C_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            Ta_C_bias_smooth = Ta_C_bias_coarse.to_geometry(geometry, resampling=downsampling)
            check_distribution(Ta_C_bias_smooth, "Ta_C_bias_smooth", date_UTC, tile)
            logger.info("bias-correcting air temperature")
            Ta_C = Ta_C_prediction - Ta_C_bias_smooth
            check_distribution(Ta_C, "Ta_C", date_UTC, tile)
            Ta_C_smooth = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            check_distribution(Ta_C_smooth, "Ta_C_smooth", date_UTC, tile)
            logger.info("gap-filling air temperature")
            Ta_C = rt.where(np.isnan(Ta_C), Ta_C_smooth, Ta_C)
            check_distribution(Ta_C, "Ta_C", date_UTC, tile)
            logger.info(
                f"up-sampling final air temperature from {int(Ta_C.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            Ta_C_final_coarse = Ta_C.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(Ta_C_final_coarse, "Ta_C_final_coarse", date_UTC, tile)
            Ta_C_error_coarse = Ta_C_final_coarse - Ta_C_coarse
            check_distribution(Ta_C_error_coarse, "Ta_C_error_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling air temperature error from {int(Ta_C_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            Ta_C_error = Ta_C_error_coarse.to_geometry(geometry, resampling=downsampling)
            check_distribution(Ta_C_error, "Ta_C_error", date_UTC, tile)

            if np.all(np.isnan(Ta_C)):
                raise BlankOutput(
                    f"blank air temperature output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

            Td_C_model = sklearn.linear_model.LinearRegression()
            Td_C_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["Td_C"])
            Td_C_intercept = Td_C_model.intercept_
            ST_C_Td_C_coef, NDVI_Td_C_coef, albedo_Td_C_coef = Td_C_model.coef_

            logger.info(
                f"dew-point temperature regression: Td_C = {Td_C_intercept:0.2f} + {ST_C_Td_C_coef:0.2f} * ST_C + {NDVI_Td_C_coef:0.2f} * NDVI + {albedo_Td_C_coef:0.2f} * albedo")
            Td_C_prediction = ST_C * ST_C_Td_C_coef + NDVI * NDVI_Td_C_coef + albedo * albedo_Td_C_coef + Td_C_intercept
            check_distribution(Td_C_prediction, "Td_C_prediction", date_UTC, tile)
            logger.info(
                f"up-sampling predicted dew-point temperature from {int(Td_C_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            Td_C_prediction_coarse = Td_C_prediction.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(Td_C_prediction_coarse, "Td_C_prediction_coarse", date_UTC, tile)
            Td_C_bias_coarse = Td_C_prediction_coarse - Td_C_coarse
            check_distribution(Td_C_bias_coarse, "Td_C_bias_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling dew-point temperature bias from {int(Td_C_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            Td_C_bias_smooth = Td_C_bias_coarse.to_geometry(geometry, resampling=downsampling)
            check_distribution(Td_C_bias_smooth, "Td_C_bias_smooth", date_UTC, tile)
            logger.info("bias-correcting dew-point temperature")
            Td_C = Td_C_prediction - Td_C_bias_smooth
            check_distribution(Td_C, "Td_C", date_UTC, tile)
            Td_C_smooth = GEOS5FP_connection.Td_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            check_distribution(Td_C_smooth, "Td_C_smooth", date_UTC, tile)
            logger.info("gap-filling dew-point temperature")
            Td_C = rt.where(np.isnan(Td_C), Td_C_smooth, Td_C)
            check_distribution(Td_C, "Td_C", date_UTC, tile)
            logger.info(
                f"up-sampling final dew-point temperature from {int(Td_C.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            Td_C_final_coarse = Td_C.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(Td_C_final_coarse, "Td_C_final_coarse", date_UTC, tile)
            Td_C_error_coarse = Td_C_final_coarse - Td_C_coarse
            check_distribution(Td_C_error_coarse, "Td_C_error_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling dew-point temperature error from {int(Td_C_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            Td_C_error = Td_C_error_coarse.to_geometry(geometry, resampling=downsampling)
            check_distribution(Td_C_error, "Td_C_error", date_UTC, tile)

            # SM_model = sklearn.linear_model.LinearRegression()
            # SM_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["SM"])
            # SM_intercept = SM_model.intercept_
            # ST_C_SM_coef, NDVI_SM_coef, albedo_SM_coef = SM_model.coef_
            # logger.info(
            #     f"soil moisture regression: SM = {SM_intercept:0.2f} + {ST_C_SM_coef:0.2f} * ST_C + {NDVI_SM_coef:0.2f} * NDVI + {albedo_SM_coef:0.2f} * albedo")
            # SM_prediction = rt.clip(ST_C * ST_C_SM_coef + NDVI * NDVI_SM_coef + albedo * albedo_SM_coef + SM_intercept, 0,
            #                         1)
            # check_distribution(SM_prediction, "SM_prediction", date_UTC, tile)
            # logger.info(
            #     f"up-sampling predicted soil moisture from {int(SM_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            # SM_prediction_coarse = SM_prediction.to_geometry(coarse_geometry, resampling=upsampling)
            # check_distribution(SM_prediction_coarse, "SM_prediction_coarse", date_UTC, tile)
            # SM_bias_coarse = SM_prediction_coarse - SM_coarse
            # check_distribution(SM_bias_coarse, "SM_bias_coarse", date_UTC, tile)
            # logger.info(
            #     f"down-sampling soil moisture bias from {int(SM_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            # SM_bias_smooth = SM_bias_coarse.to_geometry(geometry, resampling=downsampling)
            # check_distribution(SM_bias_smooth, "SM_bias_smooth", date_UTC, tile)
            # logger.info("bias-correcting soil moisture")
            # SM = rt.clip(SM_prediction - SM_bias_smooth, 0, 1)
            # check_distribution(SM, "SM", date_UTC, tile)
            # SM_smooth = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            # check_distribution(SM_smooth, "SM_smooth", date_UTC, tile)
            # logger.info("gap-filling soil moisture")
            # SM = rt.clip(rt.where(np.isnan(SM), SM_smooth, SM), 0, 1)
            # SM = rt.where(water, np.nan, SM)
            # check_distribution(SM, "SM", date_UTC, tile)
            # logger.info(
            #     f"up-sampling final soil moisture from {int(SM.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            # SM_final_coarse = SM.to_geometry(coarse_geometry, resampling=upsampling)
            # check_distribution(SM_final_coarse, "SM_final_coarse", date_UTC, tile)
            # SM_error_coarse = SM_final_coarse - SM_coarse
            # check_distribution(SM_error_coarse, "SM_error_coarse", date_UTC, tile)
            # logger.info(
            #     f"down-sampling soil moisture error from {int(SM_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            # SM_error = rt.where(water, np.nan, SM_error_coarse.to_geometry(geometry, resampling=downsampling))
            # check_distribution(SM_error, "SM_error", date_UTC, tile)

            # if np.all(np.isnan(SM)):
            #     raise BlankOutput(
            #         f"blank soil moisture output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

            Ta_K = Ta_C + 273.15
            RH = rt.clip(np.exp((17.625 * Td_C) / (243.04 + Td_C)) / np.exp((17.625 * Ta_C) / (243.04 + Ta_C)), 0, 1)

            if np.all(np.isnan(RH)):
                raise BlankOutput(
                    f"blank humidity output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")
        else:
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            Ta_C_smooth = Ta_C
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            # SM = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)

        if sharpen_soil_moisture:
            SM_model = sklearn.linear_model.LinearRegression()
            SM_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["SM"])
            SM_intercept = SM_model.intercept_
            ST_C_SM_coef, NDVI_SM_coef, albedo_SM_coef = SM_model.coef_
            logger.info(
                f"soil moisture regression: SM = {SM_intercept:0.2f} + {ST_C_SM_coef:0.2f} * ST_C + {NDVI_SM_coef:0.2f} * NDVI + {albedo_SM_coef:0.2f} * albedo")
            SM_prediction = rt.clip(ST_C * ST_C_SM_coef + NDVI * NDVI_SM_coef + albedo * albedo_SM_coef + SM_intercept, 0,
                                    1)
            check_distribution(SM_prediction, "SM_prediction", date_UTC, tile)
            logger.info(
                f"up-sampling predicted soil moisture from {int(SM_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            SM_prediction_coarse = SM_prediction.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(SM_prediction_coarse, "SM_prediction_coarse", date_UTC, tile)
            SM_bias_coarse = SM_prediction_coarse - SM_coarse
            check_distribution(SM_bias_coarse, "SM_bias_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling soil moisture bias from {int(SM_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            SM_bias_smooth = SM_bias_coarse.to_geometry(geometry, resampling=downsampling)
            check_distribution(SM_bias_smooth, "SM_bias_smooth", date_UTC, tile)
            logger.info("bias-correcting soil moisture")
            SM = rt.clip(SM_prediction - SM_bias_smooth, 0, 1)
            check_distribution(SM, "SM", date_UTC, tile)
            SM_smooth = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            check_distribution(SM_smooth, "SM_smooth", date_UTC, tile)
            logger.info("gap-filling soil moisture")
            SM = rt.clip(rt.where(np.isnan(SM), SM_smooth, SM), 0, 1)
            SM = rt.where(water, np.nan, SM)
            check_distribution(SM, "SM", date_UTC, tile)
            logger.info(
                f"up-sampling final soil moisture from {int(SM.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            SM_final_coarse = SM.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(SM_final_coarse, "SM_final_coarse", date_UTC, tile)
            SM_error_coarse = SM_final_coarse - SM_coarse
            check_distribution(SM_error_coarse, "SM_error_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling soil moisture error from {int(SM_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            SM_error = rt.where(water, np.nan, SM_error_coarse.to_geometry(geometry, resampling=downsampling))
            check_distribution(SM_error, "SM_error", date_UTC, tile)

            if np.all(np.isnan(SM)):
                raise BlankOutput(
                    f"blank soil moisture output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")
        else:
            SM = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)

        SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]
        Ea_Pa = RH * SVP_Pa
        Ea_kPa = Ea_Pa / 1000
        Ta_K = Ta_C + 273.15

        logger.info(f"running Breathing Earth System Simulator for {cl.place(tile)} at {cl.time(time_UTC)} UTC")

        BESS_results = BESS_model.BESS(
            geometry=geometry,
            target=tile,
            time_UTC=time_UTC,
            ST_K=ST_K,
            Ta_K=Ta_K,
            RH=RH,
            elevation_km=elevation_km,
            NDVI=NDVI,
            albedo=albedo,
            Rg=SWin_FLiES_ANN,
            SM=SM,
            VISdiff=VISdiff,
            VISdir=VISdir,
            NIRdiff=NIRdiff,
            NIRdir=NIRdir,
            UV=UV,
            water=water,
            output_variables=["Rn", "LE", "GPP"]
        )

        Rn_BESS = BESS_results["Rn"]
        LE_BESS = BESS_results["LE"]
        GPP = BESS_results["GPP"]  # [umol m-2 s-1]
        GPP = GPP.mask(~water)

        if np.all(np.isnan(GPP)):
            raise BlankOutput(f"blank GPP output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        NWP_filenames = sorted([posixpath.basename(filename) for filename in BESS_model.GEOS5FP_connection.filenames])
        AncillaryNWP = ",".join(NWP_filenames)
        metadata["ProductMetadata"]["AncillaryNWP"] = AncillaryNWP

        Rn_verma = PTJPLSM_model.Rn(
            date_UTC=date_UTC,
            target=tile,
            SWin=SWin,
            albedo=albedo,
            ST_C=ST_C,
            emissivity=emissivity,
            Ea_kPa=Ea_kPa,
            Ta_C=Ta_C,
            cloud_mask=cloud
        )

        if Rn_model_name == "verma":
            Rn = Rn_verma
        elif Rn_model_name == "BESS":
            Rn = Rn_BESS
        else:
            raise ValueError(f"unrecognized net radiation model: {Rn_model_name}")

        if np.all(np.isnan(Rn)) or np.all(Rn == 0):
            raise BlankOutput(f"blank net radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        STIC_model = STIC(
            working_directory=working_directory,
            static_directory=static_directory,
            GEOS5FP_connection=GEOS5FP_connection,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution
        )

        STIC_results = STIC_model.STIC(
            geometry=geometry,
            target=tile,
            time_UTC=time_UTC,
            Rn=Rn,
            RH=RH,
            Rg=SWin,
            Ta_C=Ta_C_smooth,
            ST_C=ST_C,
            albedo=albedo,
            emissivity=emissivity,
            NDVI=NDVI,
            water=water,
            max_iterations=3
        )

        LE_STIC = STIC_results["LE"]
        LEt_STIC = STIC_results["LEt"]
        G_STIC = STIC_results["G"]

        # STICcanopy = rt.clip((LEt_STIC / LE_STIC) * 100, 0, 100)
        STICcanopy = rt.clip(rt.where((LEt_STIC == 0) | (LE_STIC == 0), 0, LEt_STIC / LE_STIC), 0, 1)

        # G = calculate_G_SEBAL(Rn, ST_C, NDVI, albedo)
        # G = G_STIC

        PTJPLSM_results = PTJPLSM_model.PTJPL(
            geometry=geometry,
            target=tile,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            SWin=SWin,
            Rn=Rn,
            # G=G,
            Ta_C=Ta_C,
            RH=RH,
            SM=SM,
            Ea_kPa=Ea_kPa,
            water=water,
            output_variables=["LE", "canopy_proportion", "LE_canopy", "soil_proportion", "interception_proportion",
                              "ET", "ESI", "PET", "SM", "Rn", "Rn_daily"]
        )

        if Rn is None:
            Rn = rt.clip(PTJPLSM_results["Rn"], 0, None)

        if np.all(np.isnan(Rn)):
            raise BlankOutput(
                f"blank instantaneous net radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        Rn_daily = rt.clip(PTJPLSM_results["Rn_daily"], 0, None)

        if np.all(np.isnan(Rn_daily)):
            raise BlankOutput(
                f"blank daily net radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        LE_PTJPLSM = rt.clip(PTJPLSM_results["LE"], 0, None)

        if np.all(np.isnan(LE_PTJPLSM)):
            raise BlankOutput(
                f"blank PT-JPL-SM instantaneous ET output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        if np.all(np.isnan(LE_PTJPLSM)):
            raise BlankOutput(
                f"blank daily ET output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        LEt_PTJPLSM = rt.clip(PTJPLSM_results["LE_canopy"], 0, None)

        PTJPLSMcanopy = rt.clip(PTJPLSM_results["canopy_proportion"], 0, 1)
        PTJPLSMsoil = rt.clip(PTJPLSM_results["soil_proportion"], 0, 1)
        PTJPLSMinterception = rt.clip(PTJPLSM_results["interception_proportion"], 0, 1)
        ESI_PTJPLSM = rt.clip(PTJPLSM_results["ESI"], 0, 1)

        if np.all(np.isnan(ESI_PTJPLSM)):
            raise BlankOutput(f"blank ESI output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        PET_PTJPLSM = rt.clip(PTJPLSM_results["PET"], 0, None)
        SM = rt.clip(PTJPLSM_results["SM"], 0, 1)

        if np.all(np.isnan(SM)):
            raise BlankOutput(
                f"blank soil moisture output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        MOD16_model = MOD16(
            working_directory=working_directory,
            static_directory=static_directory,
            GEOS5FP_connection=GEOS5FP_connection,
            # MCD12_connnection=MCD12_connnection,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution
        )

        # Ta_K = Ta_C + 273.15
        # Ea_Pa = Ea_kPa * 1000

        MOD16_results = MOD16_model.MOD16(
            geometry=geometry,
            target=tile,
            time_UTC=time_UTC,
            ST_K=ST_K,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            Ta_K=Ta_K,
            Ea_Pa=Ea_Pa,
            elevation_km=elevation_km,
            SWin=SWin,
            Rn=Rn,
            Rn_daily=Rn_daily,
            # G=G,
            water=water
        )

        LE_MOD16 = MOD16_results["LE"]

        LE_BESS = LE_BESS.mask(~water)

        ETinst = rt.Raster(
            np.nanmedian([np.array(LE_PTJPLSM), np.array(LE_BESS), np.array(LE_MOD16), np.array(LE_STIC)], axis=0),
            geometry=geometry)

        EF = rt.where((ETinst == 0) | (Rn == 0), 0, ETinst / Rn)

        # hour_of_day = calculate_hour_of_day(time_UTC=time_UTC, geometry=geometry)
        # day_of_year = calculate_day_of_year(time_UTC=time_UTC, geometry=geometry)
        SHA = SHA_deg_from_doy_lat(day_of_year, geometry.lat)
        sunrise_hour = sunrise_from_sha(SHA)
        daylight_hours = daylight_from_sha(SHA)
        Rn_daily = daily_Rn_integration_verma(
            Rn,
            hour_of_day,
            sunrise_hour,
            daylight_hours
        )

        # constrain negative values of daily integrated net radiation
        Rn_daily = rt.clip(Rn_daily, 0, None)
        LE_daily = rt.clip(EF * Rn_daily, 0, None)

        daylight_seconds = daylight_hours * 3600.0

        # factor seconds out of watts to get joules and divide by latent heat of vaporization to get kilograms
        ET_daily_kg = np.clip(LE_daily * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)

        ETinstUncertainty = rt.Raster(
            np.nanstd([np.array(LE_PTJPLSM), np.array(LE_BESS), np.array(LE_MOD16), np.array(LE_STIC)], axis=0),
            geometry=geometry).mask(~water)

        if exists(L3T_JET_zip_filename):
            logger.info(f"found L3T PT-JPL file: {L3T_JET_zip_filename}")

        L3T_JET_granule = L3TJET(
            product_location=L3T_JET_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        PTJPLSMcanopy = PTJPLSMcanopy.mask(~water)
        STICcanopy = STICcanopy.mask(~water)
        PTJPLSMsoil = PTJPLSMsoil.mask(~water)
        PTJPLSMinterception = PTJPLSMinterception.mask(~water)

        LE_STIC.nodata = np.nan
        LE_PTJPLSM.nodata = np.nan
        LE_BESS.nodata = np.nan
        LE_MOD16.nodata = np.nan
        ET_daily_kg.nodata = np.nan
        ETinstUncertainty.nodata = np.nan
        PTJPLSMcanopy.nodata = np.nan
        STICcanopy.nodata = np.nan
        PTJPLSMsoil.nodata = np.nan
        PTJPLSMinterception.nodata = np.nan

        L3T_JET_granule.add_layer("STICinst", LE_STIC.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("PTJPLSMinst", LE_PTJPLSM.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("BESSinst", LE_BESS.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("MOD16inst", LE_MOD16.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("ETdaily", ET_daily_kg.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("ETinstUncertainty", ETinstUncertainty.astype(np.float32), cmap="jet")
        L3T_JET_granule.add_layer("PTJPLSMcanopy", PTJPLSMcanopy.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("STICcanopy", STICcanopy.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("PTJPLSMsoil", PTJPLSMsoil.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("PTJPLSMinterception", PTJPLSMinterception.astype(np.float32), cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        L3T_JET_granule.add_layer("cloud", cloud.astype(np.uint8), cmap=CLOUD_COLORMAP)

        percent_good_quality = 100 * (1 - np.count_nonzero(np.isnan(LE_PTJPLSM)) / LE_PTJPLSM.size)
        metadata["ProductMetadata"]["QAPercentGoodQuality"] = percent_good_quality

        metadata["StandardMetadata"]["LocalGranuleID"] = basename(L3T_JET_zip_filename)
        metadata["StandardMetadata"]["SISName"] = "Level 3/4 JET Product Specification Document"

        short_name = L3T_JET_SHORT_NAME
        logger.info(f"L3T JET short name: {cl.name(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L3T_JET_LONG_NAME
        logger.info(f"L3T JET long name: {cl.name(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        metadata["StandardMetadata"]["ProcessingLevelDescription"] = "Level 3 Tiled Evapotranspiration Ensemble"

        L3T_JET_granule.write_metadata(metadata)
        logger.info(f"writing L3T JET product zip: {cl.file(L3T_JET_zip_filename)}")
        L3T_JET_granule.write_zip(L3T_JET_zip_filename)
        logger.info(f"writing L3T JET browse image: {cl.file(L3T_JET_browse_filename)}")
        L3T_JET_granule.write_browse_image(PNG_filename=L3T_JET_browse_filename, cmap=ET_COLORMAP)
        logger.info(f"removing L3T JET tile granule directory: {cl.dir(L3T_JET_directory)}")
        shutil.rmtree(L3T_JET_directory)

        L3T_MET_granule = L3TMET(
            product_location=L3T_MET_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        L3T_MET_granule.add_layer("Ta", Ta_C.astype(np.float32), cmap="jet")
        L3T_MET_granule.add_layer("RH", RH.astype(np.float32), cmap=RH_COLORMAP)
        L3T_MET_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        L3T_MET_granule.add_layer("cloud", cloud.astype(np.uint8), cmap=CLOUD_COLORMAP)

        percent_good_quality = 100 * (1 - np.count_nonzero(np.isnan(Ta_C)) / Ta_C.size)
        metadata["ProductMetadata"]["QAPercentGoodQuality"] = percent_good_quality

        metadata["StandardMetadata"]["LocalGranuleID"] = basename(L3T_MET_zip_filename)
        metadata["StandardMetadata"]["SISName"] = "Level 3/4 JET Product Specification Document"

        short_name = L3T_MET_SHORT_NAME
        logger.info(f"L3T MET short name: {cl.name(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L3T_MET_LONG_NAME
        logger.info(f"L3T MET long name: {cl.name(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        metadata["StandardMetadata"]["ProcessingLevelDescription"] = "Level 3 Tiled Meteorology"

        L3T_MET_granule.write_metadata(metadata)
        logger.info(f"writing L3T MET PT-JPL-MET product zip: {cl.file(L3T_MET_zip_filename)}")
        L3T_MET_granule.write_zip(L3T_MET_zip_filename)
        logger.info(f"writing L3T MET PT-JPL-MET browse image: {cl.file(L3T_MET_browse_filename)}")
        L3T_MET_granule.write_browse_image(PNG_filename=L3T_MET_browse_filename, cmap="jet")
        logger.info(f"removing L3T MET PT-JPL-MET tile granule directory: {cl.dir(L3T_MET_directory)}")
        shutil.rmtree(L3T_MET_directory)

        L3T_SEB_granule = L3TSEB(
            product_location=L3T_SEB_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        L3T_SEB_granule.add_layer("Rg", SWin.astype(np.float32), cmap="jet")

        # temporary diagnostics
        if include_SEB_diagnostics:
            L3T_SEB_granule.add_layer("Rg_FLiES_ANN", SWin_FLiES_ANN.astype(np.float32), cmap="jet")
            L3T_SEB_granule.add_layer("Rg_FLiES_LUT", SWin_FLiES_LUT.astype(np.float32), cmap="jet")
            L3T_SEB_granule.add_layer("Rg_GEOS5FP", SWin_GEOS5FP.astype(np.float32), cmap="jet")

        L3T_SEB_granule.add_layer("Rn", Rn.astype(np.float32), cmap="jet")
        L3T_SEB_granule.add_layer("Rg", SWin_FLiES_ANN.astype(np.float32), cmap="jet")
        L3T_SEB_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        L3T_SEB_granule.add_layer("cloud", cloud.astype(np.uint8), cmap=CLOUD_COLORMAP)

        # temporary diagnostics
        if include_SEB_diagnostics:
            # L3T_SEB_granule.add_layer("Rn_BESS", Rn_BESS.astype(np.float32), cmap="jet")
            L3T_SEB_granule.add_layer("Rn", Rn_verma.astype(np.float32), cmap="jet")

        L3T_SEB_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        # L3T_SEB_granule.add_layer("Rn_BESS", Rn_BESS.astype(np.float32), cmap="jet")

        percent_good_quality = 100 * (1 - np.count_nonzero(np.isnan(Rn)) / Rn.size)
        metadata["ProductMetadata"]["QAPercentGoodQuality"] = percent_good_quality

        metadata["StandardMetadata"]["LocalGranuleID"] = basename(L3T_SEB_zip_filename)
        metadata["StandardMetadata"]["SISName"] = "Level 3/4 JET Product Specification Document"

        short_name = L3T_SEB_SHORT_NAME
        logger.info(f"L3T SEB short name: {cl.name(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L3T_SEB_LONG_NAME
        logger.info(f"L3T SEB long name: {cl.name(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        metadata["StandardMetadata"]["ProcessingLevelDescription"] = "Level 3 Tiled Surface Energy Balance"

        L3T_SEB_granule.write_metadata(metadata)
        logger.info(f"writing L3T SEB product zip: {cl.file(L3T_SEB_zip_filename)}")
        L3T_SEB_granule.write_zip(L3T_SEB_zip_filename)
        logger.info(f"writing L3T SEB browse image: {cl.file(L3T_SEB_browse_filename)}")
        L3T_SEB_granule.write_browse_image(PNG_filename=L3T_SEB_browse_filename, cmap="jet")
        logger.info(f"removing L3T SEB tile granule directory: {cl.dir(L3T_SEB_directory)}")
        shutil.rmtree(L3T_SEB_directory)

        L3T_SM_granule = L3TSM(
            product_location=L3T_SM_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        L3T_SM_granule.add_layer("SM", SM.astype(np.float32), cmap=SM_COLORMAP)
        L3T_SM_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        L3T_SM_granule.add_layer("cloud", cloud.astype(np.uint8), cmap=CLOUD_COLORMAP)

        percent_good_quality = 100 * (1 - np.count_nonzero(np.isnan(SM)) / SM.size)
        metadata["ProductMetadata"]["QAPercentGoodQuality"] = percent_good_quality

        metadata["StandardMetadata"]["LocalGranuleID"] = basename(L3T_SM_zip_filename)
        metadata["StandardMetadata"]["SISName"] = "Level 3/4 PT-JPL Product Specification Document"

        short_name = L3T_SM_SHORT_NAME
        logger.info(f"L3T SM short name: {cl.name(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L3T_SM_LONG_NAME
        logger.info(f"L3T SM long name: {cl.name(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        metadata["StandardMetadata"]["ProcessingLevelDescription"] = "Level 3 Tiled Soil Moisture"
        L3T_SM_granule.write_metadata(metadata)
        logger.info(f"writing L3T SM product zip: {cl.file(L3T_SM_zip_filename)}")
        L3T_SM_granule.write_zip(L3T_SM_zip_filename)
        logger.info(f"writing L3T SM browse image: {cl.file(L3T_SM_browse_filename)}")
        L3T_SM_granule.write_browse_image(PNG_filename=L3T_SM_browse_filename, cmap=SM_COLORMAP)
        logger.info(f"removing L3T SM tile granule directory: {cl.dir(L3T_SM_directory)}")
        shutil.rmtree(L3T_SM_directory)

        if exists(L4T_ESI_zip_filename):
            logger.info(f"found L4T ESI file: {L4T_ESI_zip_filename}")

        L4T_ESI_granule = L4TESI(
            product_location=L4T_ESI_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        L4T_ESI_granule.add_layer("ESI", ESI_PTJPLSM.astype(np.float32), cmap=ET_COLORMAP)
        L4T_ESI_granule.add_layer("PET", PET_PTJPLSM.astype(np.float32), cmap=ET_COLORMAP)
        L4T_ESI_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        L4T_ESI_granule.add_layer("cloud", cloud.astype(np.uint8), cmap=CLOUD_COLORMAP)

        percent_good_quality = 100 * (1 - np.count_nonzero(np.isnan(ESI_PTJPLSM)) / ESI_PTJPLSM.size)
        metadata["ProductMetadata"]["QAPercentGoodQuality"] = percent_good_quality
        metadata["StandardMetadata"]["LocalGranuleID"] = basename(L4T_ESI_zip_filename)

        short_name = L4T_ESI_SHORT_NAME
        logger.info(f"L4T ESI short name: {cl.name(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L4T_ESI_LONG_NAME
        logger.info(f"L4T ESI long name: {cl.name(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        metadata["StandardMetadata"]["SISName"] = "Level 3/4 JET Product Specification Document"
        metadata["StandardMetadata"]["ProcessingLevelID"] = "L4T"
        metadata["StandardMetadata"]["ProcessingLevelDescription"] = "Level 4 Tiled Evaporative Stress Index"

        L4T_ESI_granule.write_metadata(metadata)
        logger.info(f"writing L4T ESI product zip: {cl.file(L4T_ESI_zip_filename)}")
        L4T_ESI_granule.write_zip(L4T_ESI_zip_filename)
        logger.info(f"writing L4T ESI browse image: {cl.file(L4T_ESI_browse_filename)}")
        L4T_ESI_granule.write_browse_image(PNG_filename=L4T_ESI_browse_filename, cmap=ET_COLORMAP)
        logger.info(f"removing L4T ESI tile granule directory: {cl.dir(L4T_ESI_directory)}")
        shutil.rmtree(L4T_ESI_directory)

        if exists(L4T_WUE_zip_filename):
            logger.info(f"found L4T WUE file: {L4T_WUE_zip_filename}")

        L4T_WUE_granule = L4TWUE(
            product_location=L4T_WUE_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        # GPP from BESS is micro-mole per square meter per second [umol m-2 s-1]
        # transpiration from PT-JPL-SM is watts per square meter per second
        # we need to convert micro-moles to grams and watts to kilograms
        # GPP in grams per square meter per second
        GPP_inst_g_m2_s = GPP / 1000000 * 12.011
        # transpiration in kilograms per square meter per second
        ETt_inst_kg_m2_s = LEt_PTJPLSM / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM
        # divide grams of carbon by kilograms of water
        # watts per square meter per second factor out on both sides
        # WUE = rt.where((GPP_inst_g_m2_s == 0) | (ETt_inst_kg_m2_s < 1), 0, GPP / LEt_PTJPLSM)
        WUE = GPP_inst_g_m2_s / ETt_inst_kg_m2_s
        WUE = rt.where(np.isinf(WUE), np.nan, WUE)
        WUE = rt.clip(WUE, 0, 10)
        WUE = WUE.mask(~water)

        L4T_WUE_granule.add_layer("WUE", WUE.astype(np.float32), cmap=GPP_COLORMAP)
        L4T_WUE_granule.add_layer("GPP", GPP.astype(np.float32), cmap=GPP_COLORMAP)
        L4T_WUE_granule.add_layer("water", water.astype(np.uint8), cmap=WATER_COLORMAP)
        L4T_WUE_granule.add_layer("cloud", cloud.astype(np.uint8), cmap=CLOUD_COLORMAP)

        percent_good_quality = 100 * (1 - np.count_nonzero(np.isnan(WUE)) / WUE.size)
        metadata["ProductMetadata"]["QAPercentGoodQuality"] = percent_good_quality
        metadata["StandardMetadata"]["LocalGranuleID"] = basename(L4T_WUE_zip_filename)

        short_name = L4T_WUE_SHORT_NAME
        logger.info(f"L4T WUE short name: {cl.name(short_name)}")
        metadata["StandardMetadata"]["ShortName"] = short_name

        long_name = L4T_WUE_LONG_NAME
        logger.info(f"L4T WUE long name: {cl.name(long_name)}")
        metadata["StandardMetadata"]["LongName"] = long_name

        metadata["StandardMetadata"]["SISName"] = "Level 3/4 JET Product Specification Document"
        metadata["StandardMetadata"]["ProcessingLevelDescription"] = "Level 4 Tiled Water Use Efficiency"

        L4T_WUE_granule.write_metadata(metadata)
        logger.info(f"writing L4T WUE product zip: {cl.file(L4T_WUE_zip_filename)}")
        L4T_WUE_granule.write_zip(L4T_WUE_zip_filename)
        logger.info(f"writing L4T WUE browse image: {cl.file(L4T_WUE_browse_filename)}")
        L4T_WUE_granule.write_browse_image(PNG_filename=L4T_WUE_browse_filename, cmap=GPP_COLORMAP)
        logger.info(f"removing L4T WUE tile granule directory: {cl.dir(L4T_WUE_directory)}")
        shutil.rmtree(L4T_WUE_directory)

        logger.info(f"finished L3T L4T JET run in {cl.time(timer)} seconds")

    except (BlankOutput, BlankOutputError) as exception:
        logger.exception(exception)
        exit_code = BLANK_OUTPUT

    except (FailedGEOS5FPDownload, LPDAACServerUnreachable, ConnectionError) as exception:
        logger.exception(exception)
        exit_code = ANCILLARY_SERVER_UNREACHABLE

    except ECOSTRESSExitCodeException as exception:
        logger.exception(exception)
        exit_code = exception.exit_code

    return exit_code


def main(argv=sys.argv):
    if len(argv) == 1 or "--version" in argv:
        print(f"L3T_L4T_JET PGE ({ECOSTRESS.PGEVersion})")
        print(f"usage: L3T_L4T_JET RunConfig.xml")

        if "--version" in argv:
            return SUCCESS_EXIT_CODE
        else:
            return RUNCONFIG_FILENAME_NOT_SUPPLIED

    strip_console = "--strip-console" in argv
    save_intermediate = "--save-intermediate" in argv
    show_distribution = "--show-distribution" in argv
    runconfig_filename = str(argv[1])

    exit_code = L3T_L4T_JET(
        runconfig_filename=runconfig_filename,
        strip_console=strip_console,
        save_intermediate=save_intermediate,
        show_distribution=show_distribution
    )

    logger.info(f"L3T_L4T_JET exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
