import logging
import posixpath
from os.path import abspath, expanduser, join
from ..LPDAAC import LPDAACDataPool
from rasters import Raster, RasterGeometry

# URL = "https://e4ftl01.cr.usgs.gov/MOTA/MCD12C1.006/2019.01.01/MCD12C1.A2019001.006.2020220162300.hdf"
URL = "https://e4ftl01.cr.usgs.gov/MOTA/MCD12C1.061/2019.01.01/MCD12C1.A2019001.061.2022170020638.hdf"
DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_DOWNLOAD_DIRECTORY = "MCD12C1_download"

logger = logging.getLogger(__name__)


class LPDAACNotAvailable(ConnectionError):
    pass


class FailedDownload(ConnectionError):
    pass


class MCD12C1(LPDAACDataPool):
    def __init__(
            self,
            username: str = None,
            password: str = None,
            working_directory: str = None,
            download_directory: str = None,
            remote: str = None,
            offline_ok: bool = False):
        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        if download_directory is None:
            download_directory = join(working_directory, DEFAULT_DOWNLOAD_DIRECTORY)

        download_directory = abspath(expanduser(download_directory))

        if remote is None:
            remote = URL

        super(MCD12C1, self).__init__(
            username=username,
            password=password,
            remote=remote,
            offline_ok=offline_ok
        )

        self.working_directory = working_directory
        self.download_directory = download_directory

    @property
    def filename_base(self) -> str:
        return posixpath.basename(self.remote)

    @property
    def filename(self) -> str:
        return join(self.download_directory, self.filename_base)

    def IGBP(self, geometry: RasterGeometry = None, **kwargs) -> Raster:
        filename = self.download_URL(self.remote, self.filename)
        URI = f'HDF4_EOS:EOS_GRID:"{filename}":MOD12C1:Majority_Land_Cover_Type_1'
        IGBP = Raster.open(URI, geometry=geometry, **kwargs)

        return IGBP
