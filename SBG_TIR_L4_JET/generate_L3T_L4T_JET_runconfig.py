from os import makedirs
from os.path import join, dirname, abspath, expanduser
from shutil import which
import socket
from datetime import datetime
from uuid import uuid4  # Add this import
from ECOv002_granules import L2TLSTE

import logging
import colored_logging as cl

from .constants import *

logger = logging.getLogger(__name__)

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
