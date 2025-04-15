from os.path import join, dirname, abspath

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