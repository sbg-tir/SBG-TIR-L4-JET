from os.path import exists, basename
from datetime import datetime
import numpy as np
import rasters as rt
import logging

import shutil

import colored_logging as cl

from rasters import Raster
from ECOv002_granules import L3TJET, ET_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP

from .constants import L3T_JET_SHORT_NAME, L3T_JET_LONG_NAME

logger = logging.getLogger(__name__)

def write_L3T_JET(
            L3T_JET_zip_filename: str,
            L3T_JET_browse_filename: str,
            L3T_JET_directory: str,
            orbit: int,
            scene: int,
            tile: str,
            time_UTC: datetime,
            build: str,
            product_counter: int,
            LE_STIC: Raster,
            LE_PTJPLSM: Raster,
            LE_BESS: Raster,
            LE_PMJPL: Raster,
            ET_daily_kg: Raster,
            ETinstUncertainty: Raster,
            PTJPLSMcanopy: Raster,
            STICcanopy: Raster,
            PTJPLSMsoil: Raster,
            PTJPLSMinterception: Raster,
            water_mask: Raster,
            cloud_mask: Raster,
            metadata: dict):
        L3T_JET_granule = L3TJET(
            product_location=L3T_JET_directory,
            orbit=orbit,
            scene=scene,
            tile=tile,
            time_UTC=time_UTC,
            build=build,
            process_count=product_counter
        )

        LE_STIC.nodata = np.nan
        LE_STIC = rt.where(water_mask, np.nan, LE_STIC)
        LE_STIC = LE_STIC.astype(np.float32)
        
        LE_PTJPLSM.nodata = np.nan
        LE_PTJPLSM = rt.where(water_mask, np.nan, LE_PTJPLSM)
        LE_PTJPLSM = LE_PTJPLSM.astype(np.float32)
        
        LE_BESS.nodata = np.nan
        LE_BESS = rt.where(water_mask, np.nan, LE_BESS)
        LE_BESS = LE_BESS.astype(np.float32)
        
        LE_PMJPL.nodata = np.nan
        LE_PMJPL = rt.where(water_mask, np.nan, LE_PMJPL)
        LE_PMJPL = LE_PMJPL.astype(np.float32)
        
        ET_daily_kg.nodata = np.nan
        ET_daily_kg = rt.where(water_mask, np.nan, ET_daily_kg)
        ET_daily_kg = ET_daily_kg.astype(np.float32)
        
        ETinstUncertainty.nodata = np.nan
        ETinstUncertainty = rt.where(water_mask, np.nan, ETinstUncertainty)
        ETinstUncertainty = ETinstUncertainty.astype(np.float32)
        
        PTJPLSMcanopy.nodata = np.nan
        PTJPLSMcanopy = rt.where(water_mask, np.nan, PTJPLSMcanopy)
        PTJPLSMcanopy = PTJPLSMcanopy.astype(np.float32)
        
        STICcanopy.nodata = np.nan
        STICcanopy = rt.where(water_mask, np.nan, STICcanopy)
        STICcanopy = STICcanopy.astype(np.float32)
        
        PTJPLSMsoil.nodata = np.nan
        PTJPLSMsoil = rt.where(water_mask, np.nan, PTJPLSMsoil)
        PTJPLSMsoil = PTJPLSMsoil.astype(np.float32)
        
        PTJPLSMinterception.nodata = np.nan
        PTJPLSMinterception = rt.where(water_mask, np.nan, PTJPLSMinterception)
        PTJPLSMinterception = PTJPLSMinterception.astype(np.float32)
        
        water_mask = water_mask.astype(np.uint8)
        cloud_mask = cloud_mask.astype(np.uint8)

        L3T_JET_granule.add_layer("STICinst", LE_STIC, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("PTJPLSMinst", LE_PTJPLSM, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("BESSinst", LE_BESS, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("MOD16inst", LE_PMJPL, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("ETdaily", ET_daily_kg, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("ETinstUncertainty", ETinstUncertainty, cmap="jet")
        L3T_JET_granule.add_layer("PTJPLSMcanopy", PTJPLSMcanopy, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("STICcanopy", STICcanopy, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("PTJPLSMsoil", PTJPLSMsoil, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("PTJPLSMinterception", PTJPLSMinterception, cmap=ET_COLORMAP)
        L3T_JET_granule.add_layer("water", water_mask, cmap=WATER_COLORMAP)
        L3T_JET_granule.add_layer("cloud", cloud_mask, cmap=CLOUD_COLORMAP)

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
