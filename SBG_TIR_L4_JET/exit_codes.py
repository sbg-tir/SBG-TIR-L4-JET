"""
ECOSTRESS Collection 2 Exit Codes
https://wiki.jpl.nasa.gov/pages/viewpage.action?pageId=442764459
"""
__author__ = "Gregory Halverson"

SUCCESS_EXIT_CODE = 0
UNCLASSIFIED_FAILURE_EXIT_CODE = 1
RUNCONFIG_FILENAME_NOT_SUPPLIED = 2
UNABLE_TO_OPEN_RUNCONFIG_FILE = 3
UNABLE_TO_PARSE_RUNCONFIG_FILE = 4
CORRUPTED_RUNCONFIG_STRUCTURE = 5
MISSING_RUNCONFIG_VALUE = 6
FILESYSTEM_INACCESSIBLE = 7
INPUT_FILES_INACCESSIBLE = 8
UNCALIBRATED_GEOLOCATION = 9
INVALID_COORDINATES = 10
LAND_FILTER = 11
DAYTIME_FILTER = 12
CONUS_FILTER = 13
Auxiliary_SERVER_UNREACHABLE = 14
Auxiliary_LATENCY = 15
DOWNLOAD_FAILED = 16
BLANK_OUTPUT = 17
PRODUCT_WRITE_FAILED = 18


class ECOSTRESSExitCodeException(Exception):
    """
    This is the base class for all ECOSTRESS exceptions with an exit code.
    """
    exit_code = UNCLASSIFIED_FAILURE_EXIT_CODE


class RunConfigFilenameNotSupplied(ECOSTRESSExitCodeException):
    """
    Run config filename argument not supplied in PGE call
    """
    exit_code = RUNCONFIG_FILENAME_NOT_SUPPLIED


class UnableToOpenRunConfig(ECOSTRESSExitCodeException):
    """
    Unable to open run config file
    """
    exit_code = UNABLE_TO_OPEN_RUNCONFIG_FILE


class UnableToParseRunConfig(ECOSTRESSExitCodeException):
    """
    Unable to parse run config file
    """
    exit_code = UNABLE_TO_PARSE_RUNCONFIG_FILE


class CorruptedRunConfigStructure(ECOSTRESSExitCodeException):
    """
    Structure of run config data is corrupted
    """
    exit_code = CORRUPTED_RUNCONFIG_STRUCTURE


class MissingRunConfigValue(ECOSTRESSExitCodeException):
    """
    Missing value in run config file
    """
    exit_code = MISSING_RUNCONFIG_VALUE


class FilesystemInaccessible(ECOSTRESSExitCodeException):
    """
    Unable to access directory listed in StaticAuxiliaryFileGroup
    """
    exit_code = FILESYSTEM_INACCESSIBLE


class InputFilesInaccessible(ECOSTRESSExitCodeException):
    """
    Unable to access input product files listed in InputFileGroup
    """
    exit_code = INPUT_FILES_INACCESSIBLE


class UncalibratedGeolocation(ECOSTRESSExitCodeException):
    """
    ECOSTRESS scene rejected because L1B GEO file set OrbitCorrectionPerformed to false
    """
    exit_code = UNCALIBRATED_GEOLOCATION


class InvalidCoordinates(ECOSTRESSExitCodeException):
    """
    ECOSTRESS scene rejected because L1B GEO geolocation arrays contain coordinates outside the valid range of -90 to 90 latitude, -180 to 180 longitude, or the range of either exceeds 10 degrees
    """
    exit_code = INVALID_COORDINATES


class LandFilter(ECOSTRESSExitCodeException):
    """
    ECOSTRESS scene rejected because L1B metadata reports no land pixels covered
    """
    exit_code = LAND_FILTER


class DaytimeFilter(ECOSTRESSExitCodeException):
    """
    ECOSTRESS scene rejected because StandardMetadata/DayNightFlag attribute in the L2T_LSTE input product contains a value other than Day.
    """
    exit_code = DAYTIME_FILTER


class CONUSFilter(ECOSTRESSExitCodeException):
    """
    ECOSTRESS scene rejected the granule does not cover CONUS. In the case that an ECOSTRESS Orbit/Scene granule is on the edge of the CONUS boundary, some of the the L3T_L4T_ALEXI tiles may succeed while others return this exit code. The tiles that do succeed can still be mosaicked by L3G_L4G_ALEXI.
    """
    exit_code = CONUS_FILTER


class AuxiliaryServerUnreachable(ECOSTRESSExitCodeException):
    """
    This exit code reports that a connection could not be made to one of the remote servers hosting Auxiliary data. Sometimes these servers are down for maintenance for a few hours at a time.
    """
    exit_code = Auxiliary_SERVER_UNREACHABLE


class AuxiliaryLatency(ECOSTRESSExitCodeException):
    """
    This exit code reports that a required granule from an Auxiliary data source is not yet available.
    """
    exit_code = Auxiliary_LATENCY


class DownloadFailed(ECOSTRESSExitCodeException):
    """
    This exit code reports that the file transfer for an Auxiliary file from the remote server to the local filesystem failed. This can happen intermittently, but if it happens persistently, then the security or API of the remote server may have made a breaking change.
    """
    exit_code = DOWNLOAD_FAILED


class BlankOutput(ECOSTRESSExitCodeException):
    """
    This exit code reports that the output images contained only no-data values and were not written to a product file.
    """
    exit_code = BLANK_OUTPUT


class ProductWriteFailed(ECOSTRESSExitCodeException):
    """
    This exit code reports that the PGE was unable to write its product file.
    """
    exit_code = PRODUCT_WRITE_FAILED
