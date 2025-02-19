# http://e4ftl01.cr.usgs.gov/MEASURES/SRTMIMGM.003/2000.02.11/N20E054.SRTMIMGM.num.zip
import json
import logging
import os
import posixpath
from os import makedirs
from os.path import exists, splitext, basename, join, expanduser, abspath, dirname
from typing import List
from zipfile import ZipFile

import numpy as np
from shapely.geometry import Polygon

# URL = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/NASADEM_HGT_n00e127.zip"
# filename = "/Users/halverso/Downloads/NASADEM_HGT_n00e127.zip"
# URI = "zip:///Users/halverso/Downloads/NASADEM_HGT_n00e127.zip!/n00e127.hgt"
import colored_logging as cl
from ..LPDAAC import LPDAACDataPool
from rasters import Raster, RasterGeometry, RasterGrid

from ..timer import Timer
import pandas as pd

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_DOWNLOAD_DIRECTORY = "SRTM_directory"

SRTM_FILENAMES_CSV = join(abspath(dirname(__file__)), "filenames.csv")

logger = logging.getLogger(__name__)

class SRTMGranule:
    def __init__(self, filename: str):
        if not exists(filename):
            raise IOError(f"SRTM file not found: {filename}")

        self.filename = filename
        self._geometry = None

    @property
    def tile(self):
        return splitext(basename(self.filename))[0].split("_")[-1]

    @property
    def hgt_URI(self) -> str:
        return f"zip://{self.filename}!/{self.tile.lower()}.hgt"

    @property
    def elevation_m(self) -> Raster:
        URI = self.hgt_URI
        logger.info(f"loading elevation: {cl.URL(URI)}")
        data = Raster.open(URI)
        data = rasters.where(data == data.nodata, np.nan, data.astype(np.float32))
        data.nodata = np.nan

        return data

    # @property
    # def ocean(self) -> Raster:
    #     return self.elevation_m == 0

    @property
    def geometry(self) -> RasterGrid:
        if self._geometry is None:
            self._geometry = RasterGrid.open(self.hgt_URI)

        return self._geometry

    @property
    def swb_URI(self) -> str:
        return f"zip://{self.filename}!/{self.tile.lower()}.swb"

    @property
    def swb(self) -> Raster:
        URI = self.swb_URI
        geometry = self.geometry
        filename = self.filename
        member_name = f"{self.tile.lower()}.swb"
        logger.info(f"loading swb: {cl.URL(URI)}")

        with ZipFile(filename, "r") as zip_file:
            with zip_file.open(member_name, "r") as file:
                data = Raster(np.frombuffer(file.read(), dtype=np.int8).reshape(geometry.shape), geometry=geometry)

        data = data != 0

        return data


class TileNotAvailable(ValueError):
    pass


class SRTM(LPDAACDataPool):
    # logger = logging.getLogger(__name__)

    def __init__(
            self,
            username: str = None,
            password: str = None,
            remote: str = None,
            working_directory: str = None,
            download_directory: str = None,
            offline_ok: bool = False):
        super(SRTM, self).__init__(username=username, password=password, remote=remote, offline_ok=offline_ok)

        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        logger.info(f"SRTM working directory: {cl.dir(working_directory)}")

        if download_directory is None:
            download_directory = join(working_directory, DEFAULT_DOWNLOAD_DIRECTORY)

        download_directory = abspath(expanduser(download_directory))

        logger.info(f"SRTM download directory: {cl.dir(download_directory)}")

        self.working_directory = working_directory
        self.download_directory = download_directory

        self._filenames = None

    def __repr__(self):
        display_dict = {
            "URL": self.remote,
            "download_directory": self.download_directory
        }

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    @property
    def filenames(self) -> List[str]:
        if self._filenames is None:
            self._filenames = list(pd.read_csv(SRTM_FILENAMES_CSV)["filename"])

        return self._filenames

    def tile_URL(self, tile):
        return posixpath.join(
            self.remote,
            "MEASURES",
            "NASADEM_HGT.001",
            "2000.02.11",
            f"NASADEM_HGT_{tile.lower()}.zip"
        )

    def tiles_intersecting_bbox(self, lon_min, lat_min, lon_max, lat_max):
        tiles = []

        for lat in range(int(np.floor(lat_min)), int(np.floor(lat_max)) + 1):
            for lon in range(int(np.floor(lon_min)), int(np.floor(lon_max)) + 1):
                tiles.append(f"{'s' if lat < 0 else 'n'}{abs(lat):02d}{'w' if lon < 0 else 'e'}{abs(lon):03d}")

        tiles = sorted(tiles)

        return tiles

    def tiles_intersecting_polygon(self, polygon: Polygon):
        lons, lats = polygon.exterior.xy
        lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
        lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)

        return self.tiles_intersecting_bbox(lon_min, lat_min, lon_max, lat_max)

    def tiles(self, geometry: Polygon or RasterGeometry) -> List[str]:
        if isinstance(geometry, Polygon):
            return self.tiles_intersecting_polygon(geometry)
        elif isinstance(geometry, RasterGeometry):
            return self.tiles_intersecting_polygon(geometry.boundary_latlon)
        else:
            raise ValueError("invalid target geometry")

    def download_tile(self, tile: str) -> SRTMGranule:
        URL = self.tile_URL(tile)
        filename_base = posixpath.basename(URL)

        if filename_base not in self.filenames:
            raise TileNotAvailable(f"SRTM does not cover tile {tile}")

        directory = self.download_directory
        makedirs(directory, exist_ok=True)
        logger.info(f"acquiring SRTM tile: {cl.val(tile)} URL: {cl.URL(URL)}")
        filename = self.download_URL(URL, directory)
        granule = SRTMGranule(filename)

        return granule

    # def ocean(self, geometry: RasterGeometry) -> Raster:
    #     """
    #     digital elevation model ocean (zero elevation) from the Shuttle Rader Topography Mission (SRTM)
    #     :param geometry: target geometry
    #     :return: raster of elevation
    #     """
    #     result = None
    #     tiles = self.tiles(geometry)
    #
    #     for tile in tiles:
    #         # logger.info(f"acquiring SRTM tile {tile}")
    #         try:
    #             granule = self.download_tile(tile)
    #         except TileNotAvailable as e:
    #             logger.warning(e)
    #             continue
    #
    #         image = granule.ocean
    #         image_projected = image.to_geometry(geometry)
    #
    #         if result is None:
    #             result = image_projected
    #         else:
    #             result = rasters.where(np.isnan(result), image_projected, result)
    #
    #     return result

    def swb(self, geometry: RasterGeometry) -> Raster:
        """
        digital elevation model surface water body from the Shuttle Rader Topography Mission (SRTM)
        :param geometry: target geometry
        :return: raster of elevation
        """
        result = Raster(np.full(geometry.shape, np.nan, dtype=np.float32), geometry=geometry)
        tiles = self.tiles(geometry)

        for tile in tiles:
            # logger.info(f"acquiring SRTM tile {tile}")
            try:
                granule = self.download_tile(tile)
            except TileNotAvailable as e:
                logger.warning(e)
                continue
            
            image = granule.swb.astype(np.float32)
            image_projected = image.to_geometry(geometry)
            result = rasters.where(np.isnan(result), image_projected, result)

        result = rasters.where(np.isnan(result), 1, result)
        result = result.astype(bool)

        return result

    def elevation_m(self, geometry: RasterGeometry) -> Raster:
        """
        digital elevation model (elevation_km) in meters from the Shuttle Rader Topography Mission (SRTM)
        :param geometry: target geometry
        :return: raster of elevation
        """
        result = None
        tiles = self.tiles(geometry)

        for tile in tiles:
            # logger.info(f"acquiring SRTM tile {tile}")
            try:
                granule = self.download_tile(tile)
            except TileNotAvailable as e:
                logger.warning(e)
                continue

            elevation_m = granule.elevation_m
            elevation_projected = elevation_m.to_geometry(geometry)

            if result is None:
                result = elevation_projected
            else:
                result = rasters.where(np.isnan(result), elevation_projected, result)

        return result

    def elevation_km(self, geometry: RasterGeometry) -> Raster:
        """
        digital elevation model (elevation_km) in kilometers from the Shuttle Rader Topography Mission (SRTM)
        :param geometry: target geometry
        :return: raster of elevation
        """
        return self.elevation_m(geometry=geometry) / 1000
