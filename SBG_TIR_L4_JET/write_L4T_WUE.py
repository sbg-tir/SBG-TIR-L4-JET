from os.path import exists, basename
from datetime import datetime
import numpy as np
import shutil
import logging

import colored_logging as cl

from rasters import Raster
import rasters as rt
from ECOv002_granules import L4TWUE, GPP_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP

from .constants import L4T_WUE_SHORT_NAME, L4T_WUE_LONG_NAME

logger = logging.getLogger(__name__)

def write_L4T_WUE(
        L4T_WUE_zip_filename: str,
        L4T_WUE_browse_filename: str,
        L4T_WUE_directory: str,
        orbit: int,
        scene: int,
        tile: str,
        time_UTC: datetime,
        build: str,
        product_counter: int,
        WUE: Raster,
        GPP: Raster,
        water_mask: Raster,
        cloud_mask: Raster,
        metadata: dict):
    L4T_WUE_granule = L4TWUE(
        product_location=L4T_WUE_directory,
        orbit=orbit,
        scene=scene,
        tile=tile,
        time_UTC=time_UTC,
        build=build,
        process_count=product_counter
    )
    
    WUE.nodata = np.nan
    WUE = rt.where(water_mask, np.nan, WUE)
    WUE = WUE.astype(np.float32)

    GPP.nodata = np.nan
    GPP = rt.where(water_mask, np.nan, GPP)
    GPP = GPP.astype(np.float32)

    water_mask = water_mask.astype(np.uint8)
    cloud_mask = cloud_mask.astype(np.uint8)

    L4T_WUE_granule.add_layer("WUE", WUE, cmap=GPP_COLORMAP)
    L4T_WUE_granule.add_layer("GPP", GPP, cmap=GPP_COLORMAP)
    L4T_WUE_granule.add_layer("water", water_mask, cmap=WATER_COLORMAP)
    L4T_WUE_granule.add_layer("cloud", cloud_mask, cmap=CLOUD_COLORMAP)

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
