from os.path import abspath, expanduser, join

from ECOv002_granules import L2TLSTE

from .constants import *
from .exit_codes import *
from .runconfig import read_runconfig, ECOSTRESSRunConfig

import logging
import colored_logging as cl

logger = logging.getLogger(__name__)

class L3TL4TJETConfig(ECOSTRESSRunConfig):
    def __init__(self, filename: str):
        try:
            logger.info(f"loading L3T_L4T_JET run-config: {cl.file(filename)}")
            runconfig = read_runconfig(filename)

            # print(JSON_highlight(runconfig))

            if "StaticAuxiliaryFileGroup" not in runconfig:
                raise MissingRunConfigValue(
                    f"missing StaticAuxiliaryFileGroup in L3T_L4T_JET run-config: {filename}")

            if "L3T_L4T_JET_WORKING" not in runconfig["StaticAuxiliaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing StaticAuxiliaryFileGroup/L3T_L4T_JET_WORKING in L3T_L4T_JET run-config: {filename}")

            working_directory = abspath(runconfig["StaticAuxiliaryFileGroup"]["L3T_L4T_JET_WORKING"])
            logger.info(f"working directory: {cl.dir(working_directory)}")

            if "L3T_L4T_JET_SOURCES" not in runconfig["StaticAuxiliaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing StaticAuxiliaryFileGroup/L3T_L4T_JET_WORKING in L3T_L4T_JET run-config: {filename}")

            sources_directory = abspath(runconfig["StaticAuxiliaryFileGroup"]["L3T_L4T_JET_SOURCES"])
            logger.info(f"sources directory: {cl.dir(sources_directory)}")

            GEOS5FP_directory = join(sources_directory, DEFAULT_GEOS5FP_DIRECTORY)

            if "L3T_L4T_STATIC" not in runconfig["StaticAuxiliaryFileGroup"]:
                raise MissingRunConfigValue(
                    f"missing StaticAuxiliaryFileGroup/L3T_L4T_STATIC in L3T_L4T_JET run-config: {filename}")

            static_directory = abspath(runconfig["StaticAuxiliaryFileGroup"]["L3T_L4T_STATIC"])
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
            PGE_version = __version__

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
