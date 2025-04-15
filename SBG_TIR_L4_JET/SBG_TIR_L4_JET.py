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
from pytictoc import TicToc
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from dateutil import parser

import colored_logging as cl

import rasters as rt
from rasters import Raster, RasterGrid, RasterGeometry
from rasters import linear_downscale, bias_correct

from check_distribution import check_distribution

from solar_apparent_time import UTC_offset_hours_for_area

from koppengeiger import load_koppen_geiger
import FLiESANN
from GEOS5FP import GEOS5FP, FailedGEOS5FPDownload
from sun_angles import calculate_SZA_from_DOY_and_hour
from ECOv002_granules import L2TLSTE, L2TSTARS, L3TJET, L3TSM, L3TSEB, L3TMET, L4TESI, L4TWUE
from ECOv002_granules import ET_COLORMAP, SM_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP, RH_COLORMAP, GPP_COLORMAP
from MCD12C1_2019_v006 import load_MCD12C1_IGBP
from FLiESLUT import process_FLiES_LUT_raster
from FLiESANN import FLiESANN

from BESS_JPL import BESS_JPL
from PMJPL import PMJPL
from STIC_JPL import STIC_JPL
from PTJPLSM import PTJPLSM
from verma_net_radiation import process_verma_net_radiation, daily_Rn_integration_verma, SHA_deg_from_doy_lat, sunrise_from_SHA, daylight_from_SHA

from .version import __version__
from .constants import *
from .exit_codes import *
from .runconfig import read_runconfig, ECOSTRESSRunConfig

from .generate_L3T_L4T_JET_runconfig import generate_L3T_L4T_JET_runconfig
from .L3TL4TJETConfig import L3TL4TJETConfig

from .NDVI_to_FVC import NDVI_to_FVC

from .downscale_air_temperature import downscale_air_temperature
from .downscale_soil_moisture import downscale_soil_moisture
from .downscale_vapor_pressure_deficit import downscale_vapor_pressure_deficit
from .downscale_relative_humidity import downscale_relative_humidity

from .write_L3T_JET import write_L3T_JET
from .write_L3T_MET import write_L3T_MET
from .write_L3T_SEB import write_L3T_SEB
from .write_L3T_SM import write_L3T_SM
from .write_L4T_ESI import write_L4T_ESI
from .write_L4T_WUE import write_L4T_WUE

class LPDAACServerUnreachable(Exception):
    pass

with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read()

__version__ = version

logger = logging.getLogger(__name__)

class BlankOutputError(Exception):
    pass

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
        timer = TicToc()
        timer.tic()
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
        metadata["StandardMetadata"]["PGEVersion"] = __version__
        metadata["StandardMetadata"]["PGEName"] = "L3T_L4T_JET"
        metadata["StandardMetadata"]["ProcessingLevelID"] = "L3T"
        metadata["StandardMetadata"]["SISName"] = "Level 3 Product Specification Document"
        metadata["StandardMetadata"]["SISVersion"] = "Preliminary"
        metadata["StandardMetadata"]["AuxiliaryInputPointer"] = "AuxiliaryNWP"

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
        water_mask = L2T_LSTE_granule.water
        cloud_mask = L2T_LSTE_granule.cloud
        NDVI = L2T_STARS_granule.NDVI
        albedo = L2T_STARS_granule.albedo

        percent_cloud = 100 * np.count_nonzero(cloud_mask) / cloud_mask.size
        metadata["ProductMetadata"]["QAPercentCloudCover"] = percent_cloud

        GEOS5FP_connection = GEOS5FP(
            working_directory=working_directory,
            download_directory=GEOS5FP_directory
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

        doy_solar = time_solar.timetuple().tm_yday
        KG_climate = load_koppen_geiger(albedo.geometry)

        FLiES_results = FLiESANN(
            albedo=albedo,
            geometry=geometry,
            time_UTC=time_UTC,
            day_of_year=doy_solar,
            hour_of_day=hour_of_day,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            elevation_km=elevation_km,
            SZA=SZA,
            KG_climate=KG_climate
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

        SWin_FLiES_LUT= process_FLiES_LUT_raster(
            geometry=geometry,
            time_UTC=time_UTC,
            cloud_mask=cloud_mask,
            COT=COT,
            koppen_geiger=KG_climate,
            albedo=albedo,
            SZA=SZA,
            GEOS5FP_connection=GEOS5FP_connection
        )

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
            SM = rt.where(water_mask, np.nan, SM)
            check_distribution(SM, "SM", date_UTC, tile)
            logger.info(
                f"up-sampling final soil moisture from {int(SM.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
            SM_final_coarse = SM.to_geometry(coarse_geometry, resampling=upsampling)
            check_distribution(SM_final_coarse, "SM_final_coarse", date_UTC, tile)
            SM_error_coarse = SM_final_coarse - SM_coarse
            check_distribution(SM_error_coarse, "SM_error_coarse", date_UTC, tile)
            logger.info(
                f"down-sampling soil moisture error from {int(SM_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
            SM_error = rt.where(water_mask, np.nan, SM_error_coarse.to_geometry(geometry, resampling=downsampling))
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

        BESS_results = BESS_JPL(
            ST_C=ST_C,
            NDVI=NDVI,
            albedo=albedo,
            elevation_km=elevation_km,
            geometry=geometry,
            time_UTC=time_UTC,
            hour_of_day=hour_of_day,
            day_of_year=day_of_year,
            GEOS5FP_connection=GEOS5FP_connection,
            Ta_C=Ta_C,
            RH=RH,
            Rg=SWin_FLiES_ANN,
            VISdiff=VISdiff,
            VISdir=VISdir,
            NIRdiff=NIRdiff,
            NIRdir=NIRdir,
            UV=UV,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            KG_climate=KG_climate,
            SZA=SZA
        )

        Rn_BESS = BESS_results["Rn"]
        check_distribution(Rn_BESS, "Rn_BESS", date_UTC=date_UTC, target=tile)
        
        # total latent heat flux in watts per square meter from BESS
        LE_BESS = BESS_results["LE"]

        # water-mask BESS latent heat flux
        if water_mask is not None:
            LE_BESS = rt.where(water_mask, np.nan, LE_BESS)

        check_distribution(LE_BESS, "LE_BESS", date_UTC=date_UTC, target=tile)
        
        # gross primary productivity from BESS
        GPP_inst_umol_m2_s = BESS_results["GPP"]  # [umol m-2 s-1]
        
        # water-mask GPP
        if water_mask is not None:
            GPP_inst_umol_m2_s = rt.where(water_mask, np.nan, GPP_inst_umol_m2_s)

        check_distribution(GPP_inst_umol_m2_s, "GPP", date_UTC=date_UTC, target=tile)

        if np.all(np.isnan(GPP_inst_umol_m2_s)):
            raise BlankOutput(f"blank GPP output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        NWP_filenames = sorted([posixpath.basename(filename) for filename in GEOS5FP_connection.filenames])
        AuxiliaryNWP = ",".join(NWP_filenames)
        metadata["ProductMetadata"]["AuxiliaryNWP"] = AuxiliaryNWP

        verma_results = process_verma_net_radiation(
            SWin=SWin,
            albedo=albedo,
            ST_C=ST_C,
            emissivity=emissivity,
            Ta_C=Ta_C,
            RH=RH
        )

        Rn_verma = verma_results["Rn"]

        if Rn_model_name == "verma":
            Rn = Rn_verma
        elif Rn_model_name == "BESS":
            Rn = Rn_BESS
        else:
            raise ValueError(f"unrecognized net radiation model: {Rn_model_name}")

        if np.all(np.isnan(Rn)) or np.all(Rn == 0):
            raise BlankOutput(f"blank net radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        STIC_results = STIC_JPL(
            geometry=geometry,
            time_UTC=time_UTC,
            Rn_Wm2=Rn,
            RH=RH,
            # Rg_Wm2=SWin,
            Ta_C=Ta_C_smooth,
            ST_C=ST_C,
            albedo=albedo,
            emissivity=emissivity,
            NDVI=NDVI,
            max_iterations=3
        )

        LE_STIC = STIC_results["LE"]
        LEt_STIC = STIC_results["LEt"]
        G_STIC = STIC_results["G"]

        STICcanopy = rt.clip(rt.where((LEt_STIC == 0) | (LE_STIC == 0), 0, LEt_STIC / LE_STIC), 0, 1)

        PTJPLSM_results = PTJPLSM(
            geometry=geometry,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            Rn=Rn,
            Ta_C=Ta_C,
            RH=RH,
            soil_moisture=SM,
        )

        # total latent heat flux from PT-JPL-SM
        LE_PTJPLSM = rt.clip(PTJPLSM_results["LE"], 0, None)

        if np.all(np.isnan(LE_PTJPLSM)):
            raise BlankOutput(
                f"blank PT-JPL-SM instantaneous ET output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        if np.all(np.isnan(LE_PTJPLSM)):
            raise BlankOutput(
                f"blank daily ET output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        # canopy transpiration in watts per square meter from PT-JPL-SM
        LE_canopy_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_canopy"], 0, None)

        # normalize canopy transpiration as a fraction of total latent heat flux
        PTJPLSMcanopy = rt.clip(LE_canopy_PTJPLSM_Wm2 / LE_PTJPLSM, 0, 1)

        # water-mask canopy transpiration
        if water_mask is not None:
            PTJPLSMcanopy = rt.where(water_mask, np.nan, PTJPLSMcanopy)
        
        # soil evaporation in watts per square meter from PT-JPL-SM
        LE_soil_PTJPLSM = rt.clip(PTJPLSM_results["LE_soil"], 0, None)

        # normalize soil evaporation as a fraction of total latent heat flux
        PTJPLSMsoil = rt.clip(LE_soil_PTJPLSM / LE_PTJPLSM, 0, 1)

        # water-mask soil evaporation
        if water_mask is not None:
            PTJPLSMsoil = rt.where(water_mask, np.nan, PTJPLSMsoil)
        
        # interception evaporation in watts per square meter from PT-JPL-SM
        LE_interception_PTJPLSM = rt.clip(PTJPLSM_results["LE_interception"], 0, None)

        # normalize interception evaporation as a fraction of total latent heat flux
        PTJPLSMinterception = rt.clip(LE_interception_PTJPLSM / LE_PTJPLSM, 0, 1)

        # water-mask interception evaporation
        if water_mask is not None:
            PTJPLSMinterception = rt.where(water_mask, np.nan, PTJPLSMinterception)
        
        # potential evapotranspiration in watts per square meter from PT-JPL-SM
        PET_PTJPLSM = rt.clip(PTJPLSM_results["PET"], 0, None)

        # normalize total latent heat flux as a fraction of potential evapotranspiration
        ESI_PTJPLSM = rt.clip(LE_PTJPLSM / PET_PTJPLSM, 0, 1)

        # water-mask ESI
        if water_mask is not None:
            ESI_PTJPLSM = rt.where(water_mask, np.nan, ESI_PTJPLSM)

        if np.all(np.isnan(ESI_PTJPLSM)):
            raise BlankOutput(f"blank ESI output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        PMJPL_results = PMJPL(
            geometry=geometry,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            Ta_C=Ta_C,
            RH=RH,
            elevation_km=elevation_km,
            Rn=Rn
        )

        LE_PMJPL = PMJPL_results["LE"]

        ETinst = rt.Raster(
            np.nanmedian([np.array(LE_PTJPLSM), np.array(LE_BESS), np.array(LE_PMJPL), np.array(LE_STIC)], axis=0),
            geometry=geometry)

        ## FIXME need to revise evaporative fraction to take soil heat flux into account
        EF = rt.where((ETinst == 0) | (Rn == 0), 0, ETinst / Rn)

        SHA = SHA_deg_from_doy_lat(day_of_year, geometry.lat)
        sunrise_hour = sunrise_from_SHA(SHA)
        daylight_hours = daylight_from_SHA(SHA)

        Rn_daily = daily_Rn_integration_verma(
            Rn=Rn,
            hour_of_day=hour_of_day,
            doy=day_of_year,
            lat=geometry.lat,
        )

        # constrain negative values of daily integrated net radiation
        Rn_daily = rt.clip(Rn_daily, 0, None)
        LE_daily = rt.clip(EF * Rn_daily, 0, None)

        daylight_seconds = daylight_hours * 3600.0

        # factor seconds out of watts to get joules and divide by latent heat of vaporization to get kilograms
        ET_daily_kg = np.clip(LE_daily * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)

        ETinstUncertainty = rt.Raster(
            np.nanstd([np.array(LE_PTJPLSM), np.array(LE_BESS), np.array(LE_PMJPL), np.array(LE_STIC)], axis=0),
            geometry=geometry).mask(~water_mask)

        # GPP from BESS is micro-mole per square meter per second [umol m-2 s-1]
        # transpiration from PT-JPL-SM is watts per square meter per second
        # we need to convert micro-moles to grams and watts to kilograms
        # GPP in grams per square meter per second
        GPP_inst_g_m2_s = GPP_inst_umol_m2_s / 1000000 * 12.011
        # transpiration in kilograms per square meter per second
        ETt_inst_kg_m2_s = LE_canopy_PTJPLSM_Wm2 / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM
        # divide grams of carbon by kilograms of water
        # watts per square meter per second factor out on both sides
        # WUE = rt.where((GPP_inst_g_m2_s == 0) | (ETt_inst_kg_m2_s < 1), 0, GPP / LEt_PTJPLSM)
        WUE = GPP_inst_g_m2_s / ETt_inst_kg_m2_s
        WUE = rt.where(np.isinf(WUE), np.nan, WUE)
        WUE = rt.clip(WUE, 0, 10)

        # write the L3T JET product
        write_L3T_JET(
            L3T_JET_zip_filename=L3T_JET_zip_filename,
            L3T_JET_browse_filename=L3T_JET_browse_filename,
            L3T_JET_directory=L3T_JET_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            product_counter=product_counter,
            LE_STIC=LE_STIC,
            LE_PTJPLSM=LE_PTJPLSM,
            LE_BESS=LE_BESS,
            LE_PMJPL=LE_PMJPL,
            ET_daily_kg=ET_daily_kg,
            ETinstUncertainty=ETinstUncertainty,
            PTJPLSMcanopy=PTJPLSMcanopy,
            STICcanopy=STICcanopy,
            PTJPLSMsoil=PTJPLSMsoil,
            PTJPLSMinterception=PTJPLSMinterception,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            metadata=metadata
        )

        # write the L3T MET product
        write_L3T_MET(
            L3T_MET_zip_filename=L3T_MET_zip_filename,
            L3T_MET_browse_filename=L3T_MET_browse_filename,
            L3T_MET_directory=L3T_MET_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            product_counter=product_counter,
            Ta_C=Ta_C,
            RH=RH,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            metadata=metadata
        )

        # write the L3T SEB product
        write_L3T_SEB(
            L3T_SEB_zip_filename=L3T_SEB_zip_filename,
            L3T_SEB_browse_filename=L3T_SEB_browse_filename,
            L3T_SEB_directory=L3T_SEB_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            product_counter=product_counter,
            Rg=SWin,
            Rn=Rn_BESS,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            metadata=metadata
        )

        # write the L3T SM product
        write_L3T_SM(
            L3T_SM_zip_filename=L3T_SM_zip_filename,
            L3T_SM_browse_filename=L3T_SM_browse_filename,
            L3T_SM_directory=L3T_SM_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            product_counter=product_counter,
            SM=SM,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            metadata=metadata
        )

        # write the L4T ESI product
        write_L4T_ESI(
            L4T_ESI_zip_filename=L4T_ESI_zip_filename,
            L4T_ESI_browse_filename=L4T_ESI_browse_filename,
            L4T_ESI_directory=L4T_ESI_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            product_counter=product_counter,
            ESI=ESI_PTJPLSM,
            PET=PET_PTJPLSM,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            metadata=metadata
        )

        write_L4T_WUE(
            L4T_WUE_zip_filename=L4T_WUE_zip_filename,
            L4T_WUE_browse_filename=L4T_WUE_browse_filename,
            L4T_WUE_directory=L4T_WUE_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            product_counter=product_counter,
            WUE=WUE,
            GPP=GPP_inst_g_m2_s,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            metadata=metadata
        )

        logger.info(f"finished L3T L4T JET run in {cl.time(timer)} seconds")

    except (BlankOutput, BlankOutputError) as exception:
        logger.exception(exception)
        exit_code = BLANK_OUTPUT

    except (FailedGEOS5FPDownload, ConnectionError) as exception:
        logger.exception(exception)
        exit_code = Auxiliary_SERVER_UNREACHABLE

    except ECOSTRESSExitCodeException as exception:
        logger.exception(exception)
        exit_code = exception.exit_code

    return exit_code


def main(argv=sys.argv):
    if len(argv) == 1 or "--version" in argv:
        print(f"L3T_L4T_JET PGE ({__version__})")
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
