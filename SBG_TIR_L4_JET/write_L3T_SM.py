from os.path import exists, basename
from datetime import datetime
import numpy as np
import shutil
import logging

import colored_logging as cl

from rasters import Raster
import rasters as rt
from ECOv002_granules import L3TSM, SM_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP

from .constants import L3T_SM_SHORT_NAME, L3T_SM_LONG_NAME

logger = logging.getLogger(__name__)


def write_L3T_SM(
        L3T_SM_zip_filename: str,
        L3T_SM_browse_filename: str,
        L3T_SM_directory: str,
        orbit: int,
        scene: int,
        tile: str,
        time_UTC: datetime,
        build: str,
        product_counter: int,
        SM: Raster,
        water_mask: Raster,
        cloud_mask: Raster,
        metadata: dict):
    L3T_SM_granule = L3TSM(
        product_location=L3T_SM_directory,
        orbit=orbit,
        scene=scene,
        tile=tile,
        time_UTC=time_UTC,
        build=build,
        process_count=product_counter
    )

    SM.nodata = np.nan
    SM = rt.where(water_mask, np.nan, SM)
    SM = SM.astype(np.float32)

    water_mask = water_mask.astype(np.uint8)
    cloud_mask = cloud_mask.astype(np.uint8)

    L3T_SM_granule.add_layer("SM", SM, cmap=SM_COLORMAP)
    L3T_SM_granule.add_layer("water", water_mask, cmap=WATER_COLORMAP)
    L3T_SM_granule.add_layer("cloud", cloud_mask, cmap=CLOUD_COLORMAP)

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
