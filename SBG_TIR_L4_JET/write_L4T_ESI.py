from os.path import exists, basename
from datetime import datetime
import numpy as np
import shutil
import logging

import colored_logging as cl

from rasters import Raster
import rasters as rt
from ECOv002_granules import L4TESI, ET_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP

from .constants import L4T_ESI_SHORT_NAME, L4T_ESI_LONG_NAME

logger = logging.getLogger(__name__)

def write_L4T_ESI(
        L4T_ESI_zip_filename: str,
        L4T_ESI_browse_filename: str,
        L4T_ESI_directory: str,
        orbit: int,
        scene: int,
        tile: str,
        time_UTC: datetime,
        build: str,
        product_counter: int,
        ESI: Raster,
        PET: Raster,
        water_mask: Raster,
        cloud_mask: Raster,
        metadata: dict):
    L4T_ESI_granule = L4TESI(
        product_location=L4T_ESI_directory,
        orbit=orbit,
        scene=scene,
        tile=tile,
        time_UTC=time_UTC,
        build=build,
        process_count=product_counter
    )

    ESI.nodata = np.nan
    ESI = rt.where(water_mask, np.nan, ESI)
    ESI = ESI.astype(np.float32)

    PET.nodata = np.nan
    PET = rt.where(water_mask, np.nan, PET)
    PET = PET.astype(np.float32)

    water_mask = water_mask.astype(np.uint8)
    cloud_mask = cloud_mask.astype(np.uint8)

    L4T_ESI_granule.add_layer("ESI", ESI, cmap=ET_COLORMAP)
    L4T_ESI_granule.add_layer("PET", PET, cmap=ET_COLORMAP)
    L4T_ESI_granule.add_layer("water", water_mask, cmap=WATER_COLORMAP)
    L4T_ESI_granule.add_layer("cloud", cloud_mask, cmap=CLOUD_COLORMAP)

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
