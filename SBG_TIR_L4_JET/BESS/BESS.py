"""
Forest Light Environmental Simulator (FLiES)
Artificial Neural Network Implementation
for the Breathing Earth Systems Simulator (BESS)
"""

import logging
from collections import namedtuple
from datetime import datetime, date
from os.path import join, abspath, dirname, expanduser
from typing import Union, Callable, List

import numpy as np
from dateutil import parser
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import zscore

import colored_logging as cl

import rasters as rt
from rasters import Raster, RasterGeometry, RasterGrid

from modisci import MODISCI
from geos5fp import GEOS5FP
from gedi_canopy_height import GEDICanopyHeight

from ..FLiES.FLiES import FLiES

from ..SRTM import SRTM

__author__ = "Gregory Halverson, Robert Freepartner"

MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_GEDI_DOWNLOAD = "GEDI_download"
DEFAULT_CI_DOWNLOAD = "MODISCI_download"
DEFAULT_BESS_INTERMEDIATE = "BESS_intermediate"
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_RESAMPLING = "cubic"
DEFAULT_PASSES = 1

DEFAULT_DOWNSCALE_AIR = True
DEFAULT_DOWNSCALE_HUMIDITY = True
DEFAULT_DOWNSCALE_MOISTURE = True

GEOS_IN_SENTINEL_COARSE_CELL_SIZE = 13720

DEFAULT_OUTPUT_VARIABLES = [
    "GPP",
    "GPP_daily",
    "Rg",
    "Rn",
    "Rn_soil",
    "Rn_canopy",
    "LE",
    "LE_soil",
    "LE_canopy"
]

# GPP_COLORMAP = LinearSegmentedColormap.from_list(
#     name="GPP",
#     colors=[
#         "#000000",
#         "#325e32"
#     ]
# )


class GEDINotAvailable(IOError):
    pass


class CINotAvailable(IOError):
    pass


class BESS(FLiES):
    logger = logging.getLogger(__name__)

    def __init__(
            self,
            working_directory: str = None,
            static_directory: str = None,
            SRTM_connection: SRTM = None,
            SRTM_download: str = None,
            GEOS5FP_connection: GEOS5FP = None,
            GEOS5FP_download: str = None,
            GEOS5FP_products: str = None,
            GEDI_connection: GEDICanopyHeight = None,
            GEDI_download: str = None,
            ORNL_connection: MODISCI = None,
            CI_directory: str = None,
            intermediate_directory: str = None,
            preview_quality: int = DEFAULT_PREVIEW_QUALITY,
            ANN_model: Callable = None,
            ANN_model_filename: str = None,
            resampling: str = DEFAULT_RESAMPLING,
            passes: int = DEFAULT_PASSES,
            initialize_Tf_with_ST: bool = True,
            downscale_air: bool = DEFAULT_DOWNSCALE_AIR,
            downscale_humidity: bool = DEFAULT_DOWNSCALE_HUMIDITY,
            downscale_moisture: bool = DEFAULT_DOWNSCALE_MOISTURE,
            save_intermediate: bool = False,
            include_preview: bool = True,
            show_distribution: bool = True):
        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        if static_directory is None:
            static_directory = working_directory

        static_directory = abspath(expanduser(static_directory))

        if intermediate_directory is None:
            intermediate_directory = join(working_directory, DEFAULT_BESS_INTERMEDIATE)

        super(BESS, self).__init__(
            working_directory=working_directory,
            static_directory=static_directory,
            SRTM_connection=SRTM_connection,
            SRTM_download=SRTM_download,
            GEOS5FP_connection=GEOS5FP_connection,
            GEOS5FP_download=GEOS5FP_download,
            GEOS5FP_products=GEOS5FP_products,
            intermediate_directory=intermediate_directory,
            preview_quality=preview_quality,
            ANN_model=ANN_model,
            ANN_model_filename=ANN_model_filename,
            resampling=resampling,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution,
            include_preview=include_preview
        )

        if GEDI_connection is None:
            if GEDI_download is None:
                GEDI_download = join(static_directory, DEFAULT_GEDI_DOWNLOAD)

            try:
                self.logger.info(f"preparing GEDI canopy height dataset: {cl.dir(GEDI_download)}")
                GEDI_connection = GEDICanopyHeight(source_directory=GEDI_download)
                GEDI_filename = GEDI_connection.VRT
                self.logger.info(f"GEDI VRT ready: {cl.file(GEDI_filename)}")
            except Exception as e:
                raise GEDINotAvailable(f"unable to prepare GEDI: {GEDI_download}")

        self.GEDI_connection = GEDI_connection

        if ORNL_connection is None:
            if CI_directory is None:
                CI_directory = join(static_directory, DEFAULT_CI_DOWNLOAD)

            try:
                self.logger.info(f"preparing MODIS clumping index dataset: {cl.dir(CI_directory)}")
                ORNL_connection = MODISCI(directory=CI_directory)
                filename = ORNL_connection.download()
                self.logger.info(f"MODIS clumping index ready: {cl.file(filename)}")
            except Exception as e:
                raise CINotAvailable(f"unable to prepare clumping index: {CI_directory}")

        self.ORNL_connection = ORNL_connection
        self.passes = passes
        self.initialize_Tf_with_ST = initialize_Tf_with_ST
        self.downscale_air = downscale_air
        self.downscale_humidity = downscale_humidity
        self.downscale_moisture = downscale_moisture

    def Ta_K_coarse(
            self,
            time_UTC: datetime,
            coarse_geometry: RasterGrid = None,
            fine_geometry: RasterGrid = None,
            coarse_cell_size: int = GEOS_IN_SENTINEL_COARSE_CELL_SIZE,
            resampling: str = "cubic") -> Raster:
        if coarse_geometry is None:
            coarse_geometry = fine_geometry.rescale(coarse_cell_size)

        self.logger.info("retrieving coarse GEOS-5 FP air temperature raster in Kelvin")
        Ta_K_coarse = self.GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)

        return Ta_K_coarse

    def Td_K_coarse(
            self,
            time_UTC: datetime,
            coarse_geometry: RasterGrid = None,
            fine_geometry: RasterGrid = None,
            coarse_cell_size: int = GEOS_IN_SENTINEL_COARSE_CELL_SIZE,
            resampling: str = "cubic") -> Raster:
        if coarse_geometry is None:
            coarse_geometry = fine_geometry.rescale(coarse_cell_size)

        self.logger.info("retrieving coarse GEOS-5 FP dew-point temperature raster in Kelvin")
        Td_K_coarse = self.GEOS5FP_connection.Td_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)

        return Td_K_coarse

    def Ta_K(
            self,
            time_UTC: datetime,
            geometry: RasterGeometry = None,
            ST_K: Raster = None,
            water: Raster = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = GEOS_IN_SENTINEL_COARSE_CELL_SIZE,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            apply_scale: bool = True,
            apply_bias: bool = True,
            return_scale_and_bias: bool = False) -> Raster:
        self.logger.info("retrieving GEOS-5 FP air temperature raster in Kelvin")

        if self.downscale_air and ST_K is not None:
            return self.GEOS5FP_connection.Ta_K(
                time_UTC=time_UTC,
                geometry=geometry,
                ST_K=ST_K,
                water=water,
                coarse_geometry=coarse_geometry,
                coarse_cell_size_meters=coarse_cell_size_meters,
                resampling=resampling,
                upsampling=upsampling,
                downsampling=downsampling,
                apply_scale=apply_scale,
                apply_bias=apply_bias,
                return_scale_and_bias=return_scale_and_bias
            )
        else:
            return self.GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

    def RH(
            self,
            time_UTC: datetime,
            geometry: RasterGrid,
            SM: Raster = None,
            ST_K: Raster = None,
            water: Raster = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = GEOS_IN_SENTINEL_COARSE_CELL_SIZE,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None) -> Raster:
        self.logger.info("retrieving GEOS-5 FP relative humidity raster")

        if self.downscale_humidity and SM is not None:
            return self.GEOS5FP_connection.RH(
                time_UTC=time_UTC,
                geometry=geometry,
                SM=SM,
                ST_K=ST_K,
                water=water,
                coarse_geometry=coarse_geometry,
                coarse_cell_size_meters=coarse_cell_size_meters,
                resampling=resampling,
                upsampling=upsampling,
                downsampling=downsampling
            )
        else:
            return self.GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

    def Ea_Pa(self, time_UTC: datetime, geometry: RasterGeometry, ST_K: Raster = None,
              resampling: str = None) -> Raster:
        if resampling is None:
            resampling = self.resampling

        self.logger.info("retrieving GEOS-5 FP vapor pressure raster in Pascals")
        Ea_Pa = self.GEOS5FP_connection.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

        if self.downscale_humidity and ST_K is not None:
            mean = np.nanmean(Ea_Pa)
            sd = np.nanstd(Ea_Pa)
            self.logger.info(f"downscaling vapor pressure with mean {mean} and sd {sd}")
            Ea_Pa = ST_K.contain(zscore(ST_K, nan_policy="omit") * -sd + mean)

        return Ea_Pa

    # TODO unified downscaled meteorology

    def Ca(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        # TODO option to use OCO-2
        self.logger.info("retrieving GEOS-5 FP surface carbon dioxide concentration in ppm")
        return self.GEOS5FP_connection.CO2SC(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

    def wind_speed(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        # TODO option to sharpen vapor pressure to NDVI
        self.logger.info("retrieving GEOS-5 FP wind speed raster in meters per second")
        return self.GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

    def fC4(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "c4_percent_1d_f32.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling, nodata=np.nan)
        image = rt.clip(image, 0, 100)

        return image

    def alf(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "alf.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

        return image

    def kn(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "kn.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

        return image

    def b0_C3(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "b0_C3.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

        return image

    def m_C3(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "m_C3.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

        return image

    def m_C4(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "m_C4.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

        return image

    def peakVCmax_C3(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "peakVCmax_C3.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling, nodata=np.nan)

        return image

    def peakVCmax_C4(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "peakVCmax_C4.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling, nodata=np.nan)

        return image

    def canopy_height_meters(self, geometry: RasterGeometry) -> Raster:
        image = self.GEDI_connection.canopy_height_meters(geometry=geometry, resampling=self.resampling)
        image = rt.clip(image, 0, None)

        return image

    def CI(self, geometry: RasterGeometry) -> Raster:
        return self.ORNL_connection.CI(geometry=geometry, resampling=self.resampling)

    def NDVI_minimum(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "NDVI_minimum.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling, nodata=np.nan)
        image = rt.clip(image, -1, 1)

        return image

    def NDVI_maximum(self, geometry: RasterGeometry) -> Raster:
        filename = join(abspath(dirname(__file__)), "NDVI_maximum.tif")
        image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling, nodata=np.nan)
        image = rt.clip(image, -1, 1)

        return image

    def meteorology(
            self,
            date_UTC: date,
            target: str,
            day_of_year: Raster,
            hour_of_day: Raster,
            latitude: np.ndarray,
            elevation_m: Raster,
            SZA: Raster,
            Ta_K: Raster,
            Ea_Pa: Raster,
            Rg: Raster,
            wind_speed_mps: Raster,
            canopy_height_meters: Raster):
        """
        =============================================================================

        Module     : Meteorology
        Input      : day of year (DOY) [-],
                   : latitude (LAT) [degree],
                   : altitude (ALT) [m],
                   : solar zenith angle (SZA) [degree],
                   : MODIS overpass time (Overpass) [h],
                   : air temperature (Ta) [K],
                   : dew point temperature (Td) [K],
                   : shortwave radiation (Rs) [W m-2],
                   : wind speed (WS) [m s-1],
                   : canopy height (hc) [m],
                   : land surface temperature (LST) [K],
                   : land surface emissivity (EMIS) [-],
                   : shortwave albedo (ALB_SW) [-].
        Output     : surface pressure (Ps) [Pa],
                   : psychrometric constant (gamma) [pa K-1],
                   : air density (rhoa) [kg m-3],
                   : water vapour deficit (VPD) [Pa],
                   : relative humidity (RH) [-],
                   : 1st derivative of saturated vapour pressure (desTa) [pa K-1],
                   : 2nd derivative of saturated vapour pressure (ddesTa) [pa K-2],
                   : clear-sky emissivity (epsa) [-],
                   : aerodynamic resistance (ra) [s m-1],
                   : temporal upscaling factor for fluxes (SFd) [-],
                   : temporal upscaling factor for radiation (SFd2) [-],
                   : net radiation (Rnet) [W m-2].


        Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
        March 2020

        =============================================================================
        """

        MET = namedtuple('MET',
                         'Ps, VPD, RH, desTa, ddesTa, gamma,Cp,rhoa, epsa, R, Rc, Rs, SFd, SFd2, DL, Ra, fStress')

        # Allen et al., 1998 (FAO)

        # surface pressure
        Ps_Pa = 101325.0 * (1.0 - 0.0065 * elevation_m / Ta_K) ** (9.807 / (0.0065 * 287.0))  # [Pa]
        self.diagnostic(Ps_Pa, "Ps_Pa", date_UTC, target)
        # air temperature in Celsius
        Ta_C = Ta_K - 273.16  # [Celsius]
        # # dewpoint temperature in Celsius
        # Td_C = Td_K - 273.16  # [Celsius]
        # # ambient vapour pressure
        # Ea_Pa = 0.6108 * np.exp((17.27 * Td_C) / (Td_C + 237.3)) * 1000  # [Pa]
        # saturated vapour pressure
        SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]
        self.diagnostic(SVP_Pa, "SVP_Pa", date_UTC, target)
        # water vapour deficit
        VPD_Pa = rt.clip(SVP_Pa - Ea_Pa, 0, None)  # [Pa]
        self.diagnostic(VPD_Pa, "VPD_Pa", date_UTC, target)
        # relative humidity
        RH = rt.clip(Ea_Pa / SVP_Pa, 0, 1)  # [-]
        # 1st derivative of saturated vapour pressure
        desTa = SVP_Pa * 4098.0 * pow((Ta_C + 237.3), (-2))  # [Pa K-1]
        # 2nd derivative of saturated vapour pressure
        ddesTa = 4098.0 * (desTa * pow((Ta_C + 237.3), (-2)) + (-2) * SVP_Pa * pow((Ta_C + 237.3), (-3)))  # [Pa K-2]
        self.diagnostic(ddesTa, "ddesTa", date_UTC, target)
        # latent Heat of Vaporization
        latent_heat = 2.501 - (2.361e-3 * Ta_C)  # [J kg-1]
        self.diagnostic(latent_heat, "latent_heat", date_UTC, target)
        # psychrometric constant
        gamma = 0.00163 * Ps_Pa / latent_heat  # [Pa K-1]
        self.diagnostic(gamma, "gamma", date_UTC, target)
        # specific heat
        # this formula for specific heat was generating extreme negative values that threw off the energy balance calculation
        # Cp = 0.24 * 4185.5 * (1.0 + 0.8 * (0.622 * Ea_Pa / (Ps_Pa - Ea_Pa)))  # [J kg-1 K-1]

        # ratio molecular weight of water vapour dry air
        mv_ma = 0.622  # [-] (Wiki)

        # specific humidity
        q = (mv_ma * Ea_Pa) / (Ps_Pa - 0.378 * Ea_Pa)  # 3 [-] (Garratt, 1994)

        # specific heat of dry air
        Cpd = 1005 + (Ta_K - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)

        # specific heat of air
        Cp = Cpd * (1 + 0.84 * q)  # [J kg-1 K-1] (Garratt, 1994)

        self.diagnostic(Cp, "Cp", date_UTC, target)

        # virtual temperature
        Tv_K = Ta_K * 1.0 / (1 - 0.378 * Ea_Pa / Ps_Pa)  # [K]
        # air density
        # rhoa = Ps_Pa / (287.0 * Tv_K)  # [kg m-3]

        # air density
        rhoa = Ps_Pa / (287.05 * Ta_K)  # [kg m-3] (Garratt, 1994)

        self.diagnostic(rhoa, "rhoa", date_UTC, target)
        # inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * day_of_year)  # [-]
        # solar declination
        delta = 0.409 * np.sin(2 * np.pi / 365.0 * day_of_year - 1.39)  # [rad]
        # sunset hour angle
        # Note: the value for arccos may be invalid (< -1.0 or > 1.0).
        # This will result in NaN values in omegaS.
        omegaS = np.arccos(-np.tan(latitude * np.pi / 180.0) * np.tan(delta))  # [rad]
        # omegaS[np.logical_or(np.isnan(omegaS), np.isinf(omegaS))] = 0.0
        omegaS = rt.where(np.isnan(omegaS) | np.isinf(omegaS), 0, omegaS)
        omegaS = np.real(omegaS)
        # Day length
        DL = 24.0 / np.pi * omegaS
        # snapshot radiation
        Ra = 1333.6 * dr * np.cos(SZA * np.pi / 180.0)
        # Daily mean radiation
        RaDaily = 1333.6 / np.pi * dr * (omegaS * np.sin(latitude * np.pi / 180.0) * np.sin(delta)
                                         + np.cos(latitude * np.pi / 180.0) * np.cos(delta) * np.sin(omegaS))
        # clear-sky solar radiation
        Rgo = (0.75 + 2e-5 * elevation_m) * Ra  # [W m-2]

        # Choi et al., 2008: The Crawford and Duchon’s cloudiness factor with Brunt equation is recommended.

        # cloudy index
        cloudy = 1.0 - Rg / Rgo  # [-]
        # cloudy[cloudy < 0] = 0
        # cloudy[cloudy > 1] = 1
        cloudy = rt.clip(cloudy, 0, 1)
        # clear-sky emissivity
        epsa0 = 0.605 + 0.048 * (Ea_Pa / 100) ** 0.5  # [-]
        # all-sky emissivity
        epsa = epsa0 * (1 - cloudy) + cloudy  # [-]
        self.diagnostic(epsa, "epsa", date_UTC, target)

        # Ryu et al. 2008 2012

        # Upscaling factor
        # non0msk = RaDaily != 0
        # SFd = np.empty(np.shape(RaDaily))
        # SFd[non0msk] = 1800.0 * Ra[non0msk] / (RaDaily[non0msk] * 3600 * 24)
        # SFd[np.logical_not(non0msk)] = 1.0
        SFd = rt.where(RaDaily != 0, 1800.0 * Ra / (RaDaily * 3600 * 24), 1)
        # SFd[SZA > 89.0] = 1.0
        SFd = rt.where(SZA > 89.0, 1, SFd)
        # SFd[SFd > 1.0] = 1.0
        SFd = rt.clip(SFd, None, 1)
        self.diagnostic(SFd, "SFd", date_UTC, target)

        # bulk aerodynamic resistance
        k = 0.4  # von Karman constant
        z0 = rt.clip(canopy_height_meters * 0.05, 0.05, None)
        self.diagnostic(z0, "z0", date_UTC, target)
        ustar = wind_speed_mps * k / (np.log(10.0 / z0))  # Stability item ignored
        self.diagnostic(ustar, "ustar", date_UTC, target)
        R = wind_speed_mps / (ustar * ustar) + 2.0 / (k * ustar)  # Eq. (2-4) in Ryu et al 2008
        self.diagnostic(R, "R_raw", date_UTC, target)
        R = rt.clip(R, None, 1000)
        self.diagnostic(R, "R", date_UTC, target)
        Rs = 0.5 * R
        Rc = R  # was: Rc = 0.5 * R * 2
        self.diagnostic(Rc, "Rc", date_UTC, target)

        # Bisht et al., 2005
        DL = DL - 1.5
        # Time difference between overpass and midday
        dT = np.abs(hour_of_day - 12.0)
        # Upscaling factor for net radiation
        SFd2 = 1.5 / (np.pi * np.sin((DL - 2.0 * dT) / (2.0 * DL) * np.pi)) * DL / 24.0

        fStress = RH ** (VPD_Pa / 1000.0)

        MET.Ps = Ps_Pa
        MET.VPD = VPD_Pa
        MET.RH = RH
        MET.desTa = desTa
        MET.ddesTa = ddesTa
        MET.gamma = gamma
        MET.Cp = Cp
        MET.rhoa = rhoa
        MET.epsa = epsa
        MET.R = R
        MET.Rc = Rc
        MET.Rs = Rs
        MET.SFd = SFd
        MET.SFd2 = SFd2
        MET.DL = DL
        MET.Ra = Ra
        MET.fStress = fStress

        return MET

    def VCmax(
            self,
            date_UTC: date,
            target: str,
            peakVCmax_C3: Raster,
            peakVCmax_C4: Raster,
            LAI: Raster,
            SZA: Raster,
            LAI_minimum: Raster,
            LAI_maximum: Raster,
            fC4: Raster,
            kn: Raster) -> (Raster, Raster, Raster, Raster):
        MINIMUM_FC4 = 0.01
        A = 0.3

        # peakVCmax_C4 = rt.where(fC4 <= MINIMUM_FC4, np.nan, peakVCmax_C4)
        sf = rt.clip(rt.clip(LAI - LAI_minimum, 0, None) / rt.clip(LAI_maximum - LAI_minimum, 1, None), 0, 1)
        sf = rt.where(np.isreal(sf), sf, 0)
        sf = rt.where(np.isnan(sf), 0, sf)
        self.diagnostic(sf, "sf", date_UTC, target)
        VCmax_C3 = A * peakVCmax_C3 + (1 - A) * peakVCmax_C3 * sf
        self.diagnostic(VCmax_C3, "VCmax_C3", date_UTC, target)
        VCmax_C4 = A * peakVCmax_C4 + (1 - A) * peakVCmax_C4 * sf
        self.diagnostic(VCmax_C4, "VCmax_C4", date_UTC, target)
        # kb = 0.5 / np.cos(np.radians(SZA))
        kb = rt.where(SZA > 89, 50.0, 0.5 / np.cos(np.radians(SZA)))
        kn_kb_Lc = kn + kb * LAI
        self.diagnostic(kn_kb_Lc, "kn_kb_Lc", date_UTC, target)
        exp_neg_kn_kb_Lc = np.exp(-kn_kb_Lc)
        self.diagnostic(exp_neg_kn_kb_Lc, "exp_neg_kn_kb_Lc", date_UTC, target)
        LAI_VCmax_C3 = LAI * VCmax_C3
        self.diagnostic(LAI_VCmax_C3, "LAI_VCmax_C3", date_UTC, target)
        exp_neg_kn = np.exp(-kn)
        self.diagnostic(exp_neg_kn, "exp_neg_kn", date_UTC, target)
        VCmax_C3_tot = LAI_VCmax_C3 * (1 - exp_neg_kn) / kn
        self.diagnostic(LAI_VCmax_C3, "LAI_VCmax_C3", date_UTC, target)
        VCmax_C3_sun = LAI_VCmax_C3 * (1 - exp_neg_kn_kb_Lc) / kn_kb_Lc
        self.diagnostic(VCmax_C3_sun, "VCmax_C3_sun", date_UTC, target)
        VCmax_C3_sh = VCmax_C3_tot - VCmax_C3_sun
        self.diagnostic(VCmax_C3_sh, "VCmax_C3_sh", date_UTC, target)
        LAI_VCmax_C4 = LAI * VCmax_C4
        self.diagnostic(LAI_VCmax_C4, "LAI_VCmax_C4", date_UTC, target)
        VCmax_C4_tot = LAI_VCmax_C4 * (1 - exp_neg_kn) / kn
        self.diagnostic(VCmax_C4_tot, "VCmax_C4_tot", date_UTC, target)
        VCmax_C4_sun = LAI_VCmax_C4 * (1 - exp_neg_kn_kb_Lc) / kn_kb_Lc
        self.diagnostic(VCmax_C4_sun, "VCmax_C4_sun", date_UTC, target)
        VCmax_C4_sh = VCmax_C4_tot - VCmax_C4_sun
        self.diagnostic(VCmax_C4_sh, "VCmax_C4_sh", date_UTC, target)

        return VCmax_C3_sun, VCmax_C4_sun, VCmax_C3_sh, VCmax_C4_sh

    def canopy_shortwave_radiation(
            self,
            date_UTC: date,
            target: str,
            PARDiff: Raster,
            PARDir: Raster,
            NIRDiff: Raster,
            NIRDir: Raster,
            UV: Raster,
            SZA: Raster,
            LAI: Raster,
            CI: Raster,
            RVIS: Raster,
            RNIR: Raster) -> namedtuple:
        """
        =============================================================================

        Module     : Canopy radiative transfer
        Input      : diffuse PAR radiation (PARDiff) [W m-2],
                   : direct PAR radiation (PARDir) [W m-2],
                   : diffuse NIR radiation (NIRDiff) [W m-2],
                   : direct NIR radiation (NIRDir) [W m-2],
                   : ultroviolet radiation (UV) [W m-2],
                   : solar zenith angle (SZA) [degree],
                   : leaf area index (LAI) [-],
                   : clumping index (CI) [-],
                   : VIS albedo (ALB_VIS) [-],
                   : NIR albedo (ALB_NIR) [-],
                   : leaf maximum carboxylation rate at 25C for C3 plant (Vcmax25_C3Leaf) [umol m-2 s-1],
                   : leaf maximum carboxylation rate at 25C for C4 plant (Vcmax25_C4Leaf) [umol m-2 s-1].
        Output     : total absorbed PAR by sunlit leaves (APAR_Sun) [umol m-2 s-1],
                   : total absorbed PAR by shade leaves (APAR_Sh) [umol m-2 s-1],
                   : total absorbed SW by sunlit leaves (ASW_Sun) [W m-2],
                   : total absorbed SW by shade leaves (ASW_Sh) [W m-2],
                   : sunlit canopy maximum carboxylation rate at 25C for C3 plant (Vcmax25_C3Sun) [umol m-2 s-1],
                   : shade canopy maximum carboxylation rate at 25C for C3 plant (Vcmax25_C3Sh) [umol m-2 s-1],
                   : sunlit canopy maximum carboxylation rate at 25C for C4 plant (Vcmax25_C4Sun) [umol m-2 s-1],
                   : shade canopy maximum carboxylation rate at 25C for C4 plant (Vcmax25_C4Sh) [umol m-2 s-1],
                   : fraction of sunlit canopy (fSun) [-],
                   : ground heat storage (G) [W m-2],
                   : total absorbed SW by soil (ASW_Soil) [W m-2].
        References : Ryu, Y., Baldocchi, D. D., Kobayashi, H., Van Ingen, C., Li, J., Black, T. A., Beringer, J.,
                     Van Gorsel, E., Knohl, A., Law, B. E., & Roupsard, O. (2011).

                     Integration of MODIS land and atmosphere products with a coupled-process model i
                     to estimate gross primary productivity and evapotranspiration from 1 km to global scales.
                     Global Biogeochemical Cycles, 25(GB4017), 1-24. doi:10.1029/2011GB004053.1.


        Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
        March 2020

        =============================================================================
        """

        CSR = namedtuple('CSR',
                         'fSun, APAR_Sun, APAR_Sh, ASW_Sun, ASW_Sh, ASW_Soil, G, Vcmax25_C3Sun, Vcmax25_C3Sh, Vcmax25_C4Sun, Vcmax25_C4Sh')

        # Leaf scattering coefficients and soil reflectance (Sellers 1985)
        SIGMA_P = 0.175
        RHO_PSOIL = 0.15
        SIGMA_N = 0.825
        RHO_NSOIL = 0.30

        # Extinction coefficient for diffuse and scattered diffuse PAR
        kk_Pd = 0.72  # Table A1

        # self.diagnostic(PARDiff, "PARDiff", date_UTC, target)
        # self.diagnostic(PARDir, "PARDir", date_UTC, target)
        # self.diagnostic(NIRDiff, "NIRDiff", date_UTC, target)
        # self.diagnostic(NIRDir, "NIRDir", date_UTC, target)
        # self.diagnostic(UV, "UV", date_UTC, target)
        # self.diagnostic(SZA, "SZA", date_UTC, target)
        # self.diagnostic(LAI, "LAI", date_UTC, target)
        # self.diagnostic(CI, "CI", date_UTC, target)
        # self.diagnostic(RVIS, "RVIS", date_UTC, target)
        # self.diagnostic(RNIR, "RNIR", date_UTC, target)

        # Beam radiation extinction coefficient of canopy
        kb = rt.where(SZA > 89, 50.0, 0.5 / np.cos(np.radians(SZA)))  # Table A1
        self.diagnostic(kb, "kb", date_UTC, target)
        # Extinction coefficient for beam and scattered beam PAR
        kk_Pb = rt.where(SZA > 89, 50.0, 0.46 / np.cos(np.radians(SZA)))  # Table A1
        self.diagnostic(kk_Pb, "kk_Pb", date_UTC, target)

        # Extinction coefficient for beam and scattered beam NIR
        kk_Nb = kb * np.sqrt(1.0 - SIGMA_N)  # Table A1
        self.diagnostic(kk_Nb, "kk_Nb", date_UTC, target)
        # Extinction coefficient for diffuse and scattered diffuse NIR
        kk_Nd = 0.35 * np.sqrt(1.0 - SIGMA_N)  # Table A1
        # self.diagnostic(kk_Nd, "kk_Nd", date_UTC, target)
        # Sunlit fraction
        fSun = rt.clip(1.0 / kb * (1.0 - np.exp(-kb * LAI * CI)) / LAI, 0, 1)  # Integration of Eq. (1)
        # fSun = rt.where(LAI == 0, 0, fSun)
        self.diagnostic(fSun, "fSun", date_UTC, target)

        # For simplicity
        L_CI = LAI * CI
        self.diagnostic(L_CI, "L_CI", date_UTC, target)
        exp_kk_Pd_L_CI = np.exp(-kk_Pd * L_CI)
        self.diagnostic(exp_kk_Pd_L_CI, "exp_kk_Pd_L_CI", date_UTC, target)
        exp_kk_Nd_L_CI = np.exp(-kk_Nd * L_CI)
        self.diagnostic(exp_kk_Nd_L_CI, "exp_kk_Nd_L_CI", date_UTC, target)

        # Total absorbed incoming PAR
        Q_PDn = (1.0 - RVIS) * PARDir * (1.0 - np.exp(-kk_Pb * L_CI)) + (1.0 - RVIS) * PARDiff * (
                1.0 - exp_kk_Pd_L_CI)  # Eq. (2)
        self.diagnostic(Q_PDn, "Q_PDn", date_UTC, target)
        self.diagnostic(image=Q_PDn, variable="Q_PDn", date_UTC=date_UTC, target=target)
        # Absorbed incoming beam PAR by sunlit leaves
        Q_PbSunDn = PARDir * (1.0 - SIGMA_P) * (1.0 - np.exp(-kb * L_CI))  # Eq. (3)
        self.diagnostic(Q_PbSunDn, "Q_PbSunDn", date_UTC, target)
        # Absorbed incoming diffuse PAR by sunlit leaves
        Q_PdSunDn = PARDiff * (1.0 - RVIS) * (1.0 - np.exp(-(kk_Pd + kb) * L_CI)) * kk_Pd / (kk_Pd + kb)  # Eq. (4)
        self.diagnostic(Q_PdSunDn, "Q_PdSunDn", date_UTC, target)
        # Absorbed incoming scattered PAR by sunlit leaves
        Q_PsSunDn = PARDir * (
                (1.0 - RVIS) * (1.0 - np.exp(-(kk_Pb + kb) * L_CI)) * kk_Pb / (kk_Pb + kb) - (1.0 - SIGMA_P) * (
                1.0 - np.exp(-2.0 * kb * L_CI)) / 2.0)  # Eq. (5)
        Q_PsSunDn = rt.clip(Q_PsSunDn, 0, None)
        self.diagnostic(Q_PsSunDn, "Q_PsSunDn", date_UTC, target)
        # Absorbed incoming PAR by sunlit leaves
        Q_PSunDn = Q_PbSunDn + Q_PdSunDn + Q_PsSunDn  # Eq. (6)
        self.diagnostic(Q_PSunDn, "Q_PSunDn", date_UTC, target)
        # Absorbed incoming PAR by shade leaves
        Q_PShDn = rt.clip(Q_PDn - Q_PSunDn, 0, None)  # Eq. (7)
        self.diagnostic(Q_PShDn, "Q_PShDn", date_UTC, target)
        # Incoming PAR at soil surface
        I_PSoil = rt.clip((1.0 - RVIS) * PARDir + (1 - RVIS) * PARDiff - (Q_PSunDn + Q_PShDn), 0, None)
        self.diagnostic(I_PSoil, "I_PSoil", date_UTC, target)
        # Absorbed PAR by soil
        APAR_Soil = rt.clip((1.0 - RHO_PSOIL) * I_PSoil, 0, None)
        self.diagnostic(APAR_Soil, "APAR_Soil", date_UTC, target)
        # Absorbed outgoing PAR by sunlit leaves
        Q_PSunUp = rt.clip(I_PSoil * RHO_PSOIL * exp_kk_Pd_L_CI, 0, None)  # Eq. (8)
        self.diagnostic(Q_PSunUp, "Q_PSunUp", date_UTC, target)

        # Absorbed outgoing PAR by shade leaves
        Q_PShUp = rt.clip(I_PSoil * RHO_PSOIL * (1 - exp_kk_Pd_L_CI), 0, None)  # Eq. (9)
        self.diagnostic(Q_PShUp, "Q_PShUp", date_UTC, target)
        # Total absorbed PAR by sunlit leaves
        APAR_Sun = Q_PSunDn + Q_PSunUp  # Eq. (10)
        self.diagnostic(APAR_Sun, "APAR_Sun", date_UTC, target)
        # Total absorbed PAR by shade leaves
        APAR_Sh = Q_PShDn + Q_PShUp  # Eq. (11)
        self.diagnostic(APAR_Sh, "APAR_Sh", date_UTC, target)

        # Absorbed incoming NIR by sunlit leaves
        Q_NSunDn = NIRDir * (1.0 - SIGMA_N) * (1.0 - np.exp(-kb * L_CI)) + NIRDiff * (1 - RNIR) * (
                1.0 - np.exp(-(kk_Nd + kb) * L_CI)) * kk_Nd / (kk_Nd + kb) + NIRDir * (
                           (1.0 - RNIR) * (1.0 - np.exp(-(kk_Nb + kb) * L_CI)) * kk_Nb / (kk_Nb + kb) - (
                           1.0 - SIGMA_N) * (1.0 - np.exp(-2.0 * kb * L_CI)) / 2.0)  # Eq. (14)
        Q_NSunDn = rt.clip(Q_NSunDn, 0, None)
        self.diagnostic(Q_NSunDn, "Q_NSunDn", date_UTC, target)
        # Absorbed incoming NIR by shade leaves
        Q_NShDn = (1.0 - RNIR) * NIRDir * (1.0 - np.exp(-kk_Nb * L_CI)) + (1.0 - RNIR) * NIRDiff * (
                1.0 - exp_kk_Nd_L_CI) - Q_NSunDn  # Eq. (15)
        Q_NShDn = rt.clip(Q_NShDn, 0, None)
        self.diagnostic(Q_NShDn, "Q_NShDn", date_UTC, target)
        # Incoming NIR at soil surface
        I_NSoil = (1.0 - RNIR) * NIRDir + (1.0 - RNIR) * NIRDiff - (Q_NSunDn + Q_NShDn)
        I_NSoil = rt.clip(I_NSoil, 0, None)
        self.diagnostic(I_NSoil, "I_NSoil", date_UTC, target)
        # Absorbed NIR by soil
        ANIR_Soil = (1.0 - RHO_NSOIL) * I_NSoil
        ANIR_Soil = rt.clip(ANIR_Soil, 0, None)
        self.diagnostic(ANIR_Soil, "ANIR_Soil", date_UTC, target)
        # Absorbed outgoing NIR by sunlit leaves
        Q_NSunUp = I_NSoil * RHO_NSOIL * exp_kk_Nd_L_CI  # Eq. (16)
        Q_NSunUp = rt.clip(Q_NSunUp, 0, None)
        self.diagnostic(Q_NSunUp, "Q_NSunUp", date_UTC, target)
        # Absorbed outgoing NIR by shade leaves
        Q_NShUp = I_NSoil * RHO_NSOIL * (1.0 - exp_kk_Nd_L_CI)  # Eq. (17)
        Q_NShUp = rt.clip(Q_NShUp, 0, None)
        self.diagnostic(Q_NShUp, "Q_NShUp", date_UTC, target)
        # Total absorbed NIR by sunlit leaves
        ANIR_Sun = Q_NSunDn + Q_NSunUp  # Eq. (18)
        self.diagnostic(ANIR_Sun, "ANIR_Sun", date_UTC, target)
        # Total absorbed NIR by shade leaves
        ANIR_Sh = Q_NShDn + Q_NShUp  # Eq. (19)
        self.diagnostic(ANIR_Sh, "ANIR_Sh", date_UTC, target)

        # UV
        UVDir = UV * PARDir / (PARDir + PARDiff + 1e-5)
        self.diagnostic(UVDir, "UVDir", date_UTC, target)
        UVDiff = UV - UVDir
        self.diagnostic(UVDiff, "UVDiff", date_UTC, target)
        Q_U = (1.0 - 0.05) * UVDiff * (1.0 - np.exp(-kk_Pb * L_CI)) + (1.0 - 0.05) * UVDiff * (1.0 - exp_kk_Pd_L_CI)
        self.diagnostic(Q_U, "Q_U", date_UTC, target)
        AUV_Sun = Q_U * fSun
        self.diagnostic(AUV_Sun, "AUV_Sun", date_UTC, target)
        AUV_Sh = Q_U * (1 - fSun)
        self.diagnostic(AUV_Sh, "AUV_Sh", date_UTC, target)
        AUV_Soil = (1.0 - 0.05) * UV - Q_U
        self.diagnostic(AUV_Soil, "AUV_Soil", date_UTC, target)

        # Ground heat storage
        G = APAR_Soil * 0.28
        self.diagnostic(G, "G", date_UTC, target)

        # Summary
        ASW_Sun = APAR_Sun + ANIR_Sun + AUV_Sun
        ASW_Sun = rt.where(LAI == 0, 0, ASW_Sun)
        self.diagnostic(ASW_Sun, "ASW_Sun", date_UTC, target)
        ASW_Sh = APAR_Sh + ANIR_Sh + AUV_Sh
        ASW_Sh = rt.where(LAI == 0, 0, ASW_Sh)
        self.diagnostic(ASW_Sh, "ASW_Sh", date_UTC, target)
        ASW_Soil = APAR_Soil + ANIR_Soil + AUV_Soil
        self.diagnostic(ASW_Soil, "ASW_Soil", date_UTC, target)
        APAR_Sun = rt.where(LAI == 0, 0, APAR_Sun)
        APAR_Sun = APAR_Sun * 4.56
        self.diagnostic(APAR_Sun, "APAR_Sun", date_UTC, target)
        APAR_Sh = rt.where(LAI == 0, 0, APAR_Sh)
        APAR_Sh = APAR_Sh * 4.56
        self.diagnostic(APAR_Sh, "APAR_Sh", date_UTC, target)

        CSR.fSun = fSun
        CSR.APAR_Sun = APAR_Sun
        CSR.APAR_Sh = APAR_Sh
        CSR.ASW_Sun = ASW_Sun
        CSR.ASW_Sh = ASW_Sh
        CSR.ASW_Soil = ASW_Soil
        CSR.G = G

        return CSR

    def canopy_longwave_radiation(
            self,
            LAI: Raster,
            SZA: Raster,
            Ts_K: Raster,
            Tf_K: Raster,
            Ta_K: Raster,
            epsa: Raster,
            epsf: float,
            epss: float,
            ALW_min: float = None,
            intermediate_min: float = None,
            intermediate_max: float = None) -> namedtuple:
        """
        =============================================================================

        Module     : Canopy longwave radiation transfer
        Input      : leaf area index (LAI) [-],
                   : extinction coefficient for longwave radiation (kd) [m-1],
                   : extinction coefficient for beam radiation (kb) [m-1],
                   : air temperature (Ta) [K],
                   : soil temperature (Ts) [K],
                   : foliage temperature (Tf) [K],
                   : clear-sky emissivity (epsa) [-],
                   : soil emissivity (epss) [-],
                   : foliage emissivity (epsf) [-].
        Output     : total absorbed LW by sunlit leaves (Q_LSun),
                   : total absorbed LW by shade leaves (Q_LSh).
        References : Wang, Y., Law, R. M., Davies, H. L., McGregor, J. L., & Abramowitz, G. (2006).
                     The CSIRO Atmosphere Biosphere Land Exchange (CABLE) model for use in climate models and as an offline model.


        Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
        March 2020

        =============================================================================
        """
        CLR = namedtuple('CLR', 'ALW_Sun, ALW_Sh, ALW_Soil, Ls, La Lf')

        # SZA[SZA > 89.0] = 89.0
        SZA = rt.clip(SZA, None, 89)
        kb = 0.5 / np.cos(SZA * np.pi / 180.0)  # Table A1 in Ryu et al 2011
        kd = 0.78  # Table A1 in Ryu et al 2011

        # Stefan_Boltzmann_constant
        sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)

        # Long wave radiation flux densities from air, soil and leaf
        La = rt.clip(epsa * sigma * Ta_K ** 4, 0, None)
        Ls = rt.clip(epss * sigma * Ts_K ** 4, 0, None)
        Lf = rt.clip(epsf * sigma * Tf_K ** 4, 0, None)

        # For simplicity
        kd_LAI = kd * LAI

        # Absorbed longwave radiation by sunlit leaves
        CLR.ALW_Sun = rt.clip(
            rt.clip(Ls - Lf, intermediate_min, None) * kd * (np.exp(-kd_LAI) - np.exp(-kb * LAI)) / (
                    kd - kb) + kd * rt.clip(La - Lf,
                                            intermediate_min, intermediate_max) * (
                    1.0 - np.exp(-(kb + kd) * LAI)), ALW_min, None) / (kd + kb)  # Eq. (44)

        # Absorbed longwave radiation by shade leaves
        CLR.ALW_Sh = rt.clip(
            (1.0 - np.exp(-kd_LAI)) * rt.clip(Ls + La - 2 * Lf, intermediate_min, intermediate_max) - CLR.ALW_Sun,
            ALW_min,
            None)  # Eq. (45)

        # Absorbed longwave radiation by soil
        CLR.ALW_Soil = rt.clip((1.0 - np.exp(-kd_LAI)) * Lf + np.exp(-kd_LAI) * La, ALW_min, None)  # Eq. (41)
        CLR.Ls = Ls
        CLR.La = La
        CLR.Lf = Lf

        return CLR

    def C4_photosynthesis(self, Tf_K: Raster, Ci: Raster, APAR: Raster, Vcmax25: Raster) -> Raster:
        """
        =============================================================================
        Collatz et al., 1992

        Module     : Photosynthesis for C4 plant
        Input      : leaf temperature (Tf) [K],
                   : intercellular CO2 concentration (Ci) [umol mol-1],
                   : absorbed photosynthetically active radiation (APAR) [umol m-2 s-1],
                   : maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1].
        Output     : net assimilation (An) [umol m-2 s-1].


        Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
        March 2020

        =============================================================================
        """
        # Temperature correction
        item = (Tf_K - 298.15) / 10.0
        Q10 = 2.0
        k = 0.7 * pow(Q10, item)  # [mol m-2 s-1]
        Vcmax_o = Vcmax25 * pow(Q10, item)  # [umol m-2 s-1]
        Vcmax = Vcmax_o / (
                (1.0 + np.exp(0.3 * (286.15 - Tf_K))) * (1.0 + np.exp(0.3 * (Tf_K - 309.15))))  # [umol m-2 s-1]
        Rd_o = 0.8 * pow(Q10, item)  # [umol m-2 s-1]
        Rd = Rd_o / (1.0 + np.exp(1.3 * (Tf_K - 328.15)))  # [umol m-2 s-1]

        # Three limiting states
        Je = Vcmax  # [umol m-2 s-1]
        alf = 0.067  # [mol CO2 mol photons-1]
        Ji = alf * APAR  # [umol m-2 s-1]
        ci = Ci * 1e-6  # [umol mol-1] -> [mol CO2 mol CO2-1]
        Jc = ci * k * 1e6  # [umol m-2 s-1]

        # Colimitation (not the case at canopy level according to DePury and Farquhar)
        a = 0.83
        b = -(Je + Ji)
        c = Je * Ji
        Jei = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        Jei = np.real(Jei)
        a = 0.93
        b = -(Jei + Jc)
        c = Jei * Jc
        Jeic = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        Jeic = np.real(Jeic)

        # Net assimilation
        # An = nanmin(cat(3,Je,Ji,Jc),[],3) - Rd;    % [umol m-2 s-1]
        An = rt.clip(Jeic - Rd, 0, None)
        # An[An<0.0] = 0.0

        return An

    def C3_photosynthesis(
            self,
            Tf_K: Raster,
            Ci: Raster,
            APAR: Raster,
            Vcmax25: Raster,
            Ps_Pa: Raster,
            alf: Raster) -> Raster:
        """
        =============================================================================
        Collatz et al., 1991

        Module     : Photosynthesis for C3 plant
        Input      : leaf temperature (Tf) [K],
                   : intercellular CO2 concentration (Ci) [umol mol-1],
                   : absorbed photosynthetically active radiation (APAR) [umol m-2 s-1],
                   : maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1],
                   : surface pressure (Ps) [Pa].
        Output     : net assimilation (An) [umol m-2 s-1].


        Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
        March 2020

        =============================================================================
        """

        # TODO document the alf input

        # Gas constant
        R = 8.314e-3  # [kJ K-1 mol-1]
        # O2 concentration
        O2 = Ps_Pa * 0.21  # [Pa]
        # Unit convertion
        Pi = Ci * 1e-6 * Ps_Pa  # [umol mol-1] -> [Pa]

        # Temperature correction
        item = (Tf_K - 298.15) / 10
        KC25 = 30  # [Pa]
        KCQ10 = 2.1  # [-]
        KO25 = 30000  # [Pa]
        KOQ10 = 1.2  # [-]
        tao25 = 2600  # [Pa]
        taoQ10 = 0.57  # [-]
        # KC = KC25 * pow(KCQ10, item)  # [Pa]
        KC = KC25 * KCQ10 ** item  # [Pa]
        # KO = KO25 * pow(KOQ10, item)  # [Pa]
        KO = KO25 * KOQ10 ** item  # [Pa]
        K = KC * (1.0 + O2 / KO)  # [Pa]
        # tao = tao25 * pow(taoQ10, item)  # [Pa]
        tao = tao25 * taoQ10 ** item  # [Pa]
        GammaS = O2 / (2.0 * tao)  # [Pa]
        VcmaxQ10 = 2.4  # [-]
        # Vcmax_o = Vcmax25 * pow(VcmaxQ10, item)  # [umol m-2 s-1]
        Vcmax_o = Vcmax25 * VcmaxQ10 ** item  # [umol m-2 s-1]
        Vcmax = Vcmax_o / (1.0 + np.exp((-220.0 + 0.703 * Tf_K) / (R * Tf_K)))  # [umol m-2 s-1]
        Rd_o = 0.015 * Vcmax  # [umol m-2 s-1]
        Rd = Rd_o * 1.0 / (1.0 + np.exp(1.3 * (Tf_K - 273.15 - 55.0)))  # [umol m-2 s-1]

        # Three limiting states
        JC = Vcmax * (Pi - GammaS) / (Pi + K)
        JE = alf * APAR * (Pi - GammaS) / (Pi + 2.0 * GammaS)
        JS = Vcmax / 2.0

        # Colimitation (not the case at canopy level according to DePury and Farquhar)
        a = 0.98
        b = -(JC + JE)
        c = JC * JE
        JCE = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        JCE = np.real(JCE)
        a = 0.95
        b = -(JCE + JS)
        c = JCE * JS
        JCES = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        JCES = np.real(JCES)

        # Net assimilation
        # An = nanmin(cat(3,JC,JE,JS),[],3) - Rd    % [umol m-2 s-1]
        An = rt.clip(JCES - Rd, 0, None)
        # An[An < 0.0] = 0.0

        return An

    def energy_balance(
            self,
            date_UTC: date,
            target: str,
            An: Raster,
            ASW: Raster,
            ALW: Raster,
            Tf_K: Raster,
            Ps_Pa: Raster,
            Ca: Raster,
            Ta_K: Raster,
            RH: Raster,
            VPD_Pa: Raster,
            desTa: Raster,
            ddesTa: Raster,
            gamma: Raster,
            Cp: Raster,
            rhoa: Raster,
            Rc: Raster,
            m: Raster,
            b0: Raster,
            flgC4: bool,
            carbon: int,
            iter: int) -> namedtuple:
        """
        =============================================================================

        Module     : Canopy energy balance
        Input      : net assimulation (An) [umol m-2 s-1],
                   : total absorbed shortwave radiation by sunlit/shade canopy (ASW) [umol m-2 s-1],
                   : total absorbed longwave radiation by sunlit/shade canopy (ALW) [umol m-2 s-1],
                   : sunlit/shade leaf temperature (Tf) [K],
                   : surface pressure (Ps) [Pa],
                   : ambient CO2 concentration (Ca) [umol mol-1],
                   : air temperature (Ta) [K],
                   : relative humidity (RH) [-],
                   : water vapour deficit (VPD) [Pa],
                   : 1st derivative of saturated vapour pressure (desTa),
                   : 2nd derivative of saturated vapour pressure (ddesTa),
                   : psychrometric constant (gamma) [pa K-1],
                   : air density (rhoa) [kg m-3],
                   : aerodynamic resistance (ra) [s m-1],
                   : Ball-Berry slope (m) [-],
                   : Ball-Berry intercept (b0) [-].
        Output     : sunlit/shade canopy net radiation (Rn) [W m-2],
                   : sunlit/shade canopy latent heat (LE) [W m-2],
                   : sunlit/shade canopy sensible heat (H) [W m-2],
                   : sunlit/shade leaf temperature (Tl) [K],
                   : stomatal resistance to vapour transfer from cell to leaf surface (rs) [s m-1],
                   : inter-cellular CO2 concentration (Ci) [umol mol-1].
        References : Paw U, K. T., & Gao, W. (1988). Applications of solutions to non-linear energy budget equations.
                     Agricultural and Forest Meteorology, 43(2), 121�145. doi:10.1016/0168-1923(88)90087-1


        Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
        March 2020

        =============================================================================
        """

        EB = namedtuple('EB', 'Rn, LE, H, Tf, gs, Ci')

        # Convert factor
        cf = 0.446 * (273.15 / Tf_K) * (Ps_Pa / 101325.0)
        self.diagnostic(cf, f"cf_C{carbon}_I{iter}", date_UTC, target)
        # Stefan_Boltzmann_constant
        sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)

        # Stomatal H2O conductance
        gs1 = m * RH * An / Ca + b0  # [mol m-2 s-1]
        self.diagnostic(gs1, f"gs1_C{carbon}_I{iter}", date_UTC, target)

        # Intercellular CO2 concentration
        Ci = Ca - 1.6 * An / gs1  # [umol./mol]

        if flgC4:
            Ci = rt.clip(Ci, 0.2 * Ca, 0.6 * Ca)
        else:
            Ci = rt.clip(Ci, 0.5 * Ca, 0.9 * Ca)

        self.diagnostic(Ci, f"Ci_C{carbon}_I{iter}", date_UTC, target)

        # Stomatal resistance to vapour transfer from cell to leaf surface
        rs = 1.0 / (gs1 / cf * 1e-2)  # [s m-1]
        self.diagnostic(rs, f"rs_C{carbon}_I{iter}", date_UTC, target)

        # Stomatal H2O conductance
        gs2 = 1.0 / rs  # [m s-1]
        self.diagnostic(gs2, f"gs1_C{carbon}_I{iter}", date_UTC, target)

        # Canopy net radiation
        Rn = rt.clip(ASW + ALW - 4.0 * 0.98 * sigma * (Ta_K ** 3) * (Tf_K - Ta_K), 0, None)

        # To reduce redundant computation
        rc = rs
        ddesTa_Rc2 = ddesTa * Rc * Rc
        self.diagnostic(ddesTa_Rc2, f"ddesTa_Rc2_C{carbon}_I{iter}", date_UTC, target)
        gamma_Rc_rc = gamma * (Rc + rc)
        self.diagnostic(gamma_Rc_rc, f"gamma_Rc_rc_C{carbon}_I{iter}", date_UTC, target)
        rhoa_Cp_gamma_Rc_rc = rhoa * Cp * gamma_Rc_rc
        self.diagnostic(rhoa_Cp_gamma_Rc_rc, f"rhoa_Cp_gamma_Rc_rc_C{carbon}_I{iter}", date_UTC, target)

        # Solution (Paw and Gao 1988)
        a = 1.0 / 2.0 * ddesTa_Rc2 / rhoa_Cp_gamma_Rc_rc  # Eq. (10b)
        self.diagnostic(a, f"a_C{carbon}_I{iter}", date_UTC, target)
        b = -1.0 - Rc * desTa / gamma_Rc_rc - ddesTa_Rc2 * Rn / rhoa_Cp_gamma_Rc_rc  # Eq. (10c)
        self.diagnostic(b, f"b_C{carbon}_I{iter}", date_UTC, target)
        c = rhoa * Cp / gamma_Rc_rc * VPD_Pa + desTa * Rc / gamma_Rc_rc * Rn + 1.0 / 2.0 * ddesTa_Rc2 / rhoa_Cp_gamma_Rc_rc * Rn * Rn  # Eq. (10d) in Paw and Gao (1988)
        self.diagnostic(c, f"c_C{carbon}_I{iter}", date_UTC, target)
        LE = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)  # Eq. (10a)
        LE = np.real(LE)
        self.diagnostic(LE, f"LE_raw_C{carbon}_I{iter}", date_UTC, target)

        # Constraints
        # LE[LE > Rn] = Rn[LE > Rn]
        LE = rt.clip(LE, 0, Rn)
        # LE[Rn < 0.0] = 0.0
        # LE[LE < 0.0] = 0.0
        # LE[Ta < 273.15] = 0.0
        LE = rt.where(Ta_K < 273.15, 0, LE)

        # Update
        H = rt.clip(Rn - LE, 0, Rn)
        self.diagnostic(Rc, f"Rc_C{carbon}_I{iter}", date_UTC, target)
        self.diagnostic(rhoa, f"rhoa_C{carbon}_I{iter}", date_UTC, target)
        self.diagnostic(Cp, f"Cp_C{carbon}_I{iter}", date_UTC, target)
        dT = rt.clip(Rc / (rhoa * Cp) * H, -20, 20)  # Eq. (6)
        self.diagnostic(dT, f"dT_C{carbon}_I{iter}", date_UTC, target)
        Tf_K = Ta_K + dT

        EB.Rn = Rn
        EB.LE = LE
        EB.H = H
        EB.Tf = Tf_K
        EB.gs = gs2
        EB.Ci = Ci

        return EB

    def soil(
            self,
            Ts: Raster,
            Ta: Raster,
            G: Raster,
            VPD: Raster,
            RH: Raster,
            gamma: Raster,
            Cp: Raster,
            rhoa: Raster,
            desTa: Raster,
            Rs: Raster,
            ASW_Soil: Raster,
            ALW_Soil: Raster,
            Ls: Raster,
            epsa: Raster) -> namedtuple:
        SOIL = namedtuple('SOIL', 'Rn, LE, H, Ts')

        # Net radiation
        # Rn = Rnet - Rn_Sun - Rn_Sh
        sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)
        Rn = rt.clip(ASW_Soil + ALW_Soil - Ls - 4.0 * epsa * sigma * (Ta ** 3) * (Ts - Ta), 0, None)
        # G = Rn * 0.35

        # Latent heat
        LE = desTa / (desTa + gamma) * (Rn - G) * (RH ** (VPD / 1000.0))  # (Ryu et al., 2011)
        LE = rt.clip(LE, 0, Rn)
        # Sensible heat
        H = rt.clip(Rn - G - LE, 0, Rn)

        # Update temperature
        dT = rt.clip(Rs / (rhoa * Cp) * H, -20, 20)
        Ts = Ta + dT

        SOIL.Rn = Rn
        SOIL.LE = LE
        SOIL.H = H
        SOIL.Ts = Ts

        return SOIL

    def carbon_water_fluxes(
            self,
            date_UTC: date,
            target: str,
            ST_K: Raster,
            LAI: Raster,
            Ta_K: Raster,
            APAR_Sun: Raster,
            APAR_Sh: Raster,
            ASW_Sun: Raster,
            ASW_Sh: Raster,
            Vcmax25_Sun: Raster,
            Vcmax25_Sh: Raster,
            m: Raster,
            b0: Union[Raster, float],
            fSun: Raster,
            ASW_Soil: Raster,
            G: Raster,
            SZA: Raster,
            Ca: Raster,
            Ps_Pa: Raster,
            gamma: Raster,
            Cp: Raster,
            rhoa: Raster,
            VPD_Pa: Raster,
            RH: Raster,
            desTa: Raster,
            ddesTa: Raster,
            epsa: Raster,
            Rc: Raster,
            Rs: Raster,
            alf: Raster,
            fStress: Raster,
            FVC: Raster,
            C4: bool) -> namedtuple:
        CWF = namedtuple('CWF', 'GPP, LE, LE_soil, LE_canopy, Rn, Rn_soil, Rn_canopy')

        if C4:
            carbon = 4
        else:
            carbon = 3

            # Constraints
        if C4:
            GPP_max = 50
        else:
            GPP_max = 40

        # Initialization

        if self.initialize_Tf_with_ST:
            # re-configuring initialization of BESS soil and canopy temperatures to ECOSTRESS surface temperature
            Tf_Sun_K = ST_K
            Tf_Sh_K = ST_K
            Ts_K = ST_K
            Tf_K = ST_K
        else:
            # this model originally initialized soil and canopy temperature to air temperature
            Tf_Sun_K = Ta_K
            Tf_Sh_K = Ta_K
            Ts_K = Ta_K
            Tf_K = Ta_K

        self.diagnostic(Tf_Sun_K, f"Tf_Sun_K_C{carbon}_init", date_UTC, target)
        self.diagnostic(Tf_Sh_K, f"Tf_Sh_K_C{carbon}_init", date_UTC, target)
        self.diagnostic(Ts_K, f"Ts_K_C{carbon}_init", date_UTC, target)
        self.diagnostic(Tf_K, f"Tf_K_C{carbon}_init", date_UTC, target)

        if C4:
            chi = 0.4
        else:
            chi = 0.7

        Ci_Sun = Ca * chi
        self.diagnostic(Ci_Sun, f"Ci_Sun_C{carbon}_init", date_UTC, target)
        Ci_Sh = Ca * chi
        self.diagnostic(Ci_Sh, f"Ci_Sh_C{carbon}_init", date_UTC, target)
        b0 = b0 * fStress
        self.diagnostic(b0, f"b0_C{carbon}_init", date_UTC, target)

        epsf = 0.98
        epss = 0.96
        sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)

        # initialize sunlit partition (overwritten when iterations process)
        An_Sun = Tf_Sun_K * 0
        Rn_Sun = Tf_Sun_K * 0
        LE_Sun = Tf_Sun_K * 0
        H_Sun = Tf_Sun_K * 0
        gs_Sun = Tf_Sun_K * 0

        # initialize shaded partition (overwritten when iterations process)
        An_Sh = Tf_Sh_K * 0
        Rn_Sh = Tf_Sh_K * 0
        LE_Sh = Tf_Sh_K * 0
        H_Sh = Tf_Sh_K * 0
        gs_Sh = Tf_Sh_K * 0

        # initialize soil partition (overwritten when iterations process)
        Rn_Soil = Ts_K * 0
        LE_Soil = Ts_K * 0
        H_Soil = Ts_K * 0
        gs_Soil = Ts_K * 0

        # Iteration
        for iter in range(1, self.passes + 1):

            # Longwave radiation
            # CLR:[ALW_Sun, ALW_Sh, ALW_Soil, Ls, La]
            CLR = self.canopy_longwave_radiation(
                LAI=LAI,  # leaf area index (LAI) [-]
                SZA=SZA,  # solar zenith angle (degrees)
                Ts_K=Ts_K,  # soil temperature (Ts) [K]
                Tf_K=Tf_K,  # foliage temperature (Tf) [K]
                Ta_K=Ta_K,  # air temperature (Ta) [K]
                epsa=epsa,  # clear-sky emissivity (epsa) [-]
                epsf=epsf,  # foliage emissivity (epsf) [-]
                epss=epss  # soil emissivity (epss) [-],
            )

            ALW_Sun = CLR.ALW_Sun
            self.diagnostic(ALW_Sun, f"ALW_Sun_C{carbon}_I{iter}", date_UTC, target)
            ALW_Sh = CLR.ALW_Sh
            self.diagnostic(ALW_Sh, f"ALW_Sh_C{carbon}_I{iter}", date_UTC, target)
            ALW_Soil = CLR.ALW_Soil
            self.diagnostic(ALW_Soil, f"ALW_Soil_C{carbon}_I{iter}", date_UTC, target)
            La = CLR.La
            self.diagnostic(La, f"La_C{carbon}_I{iter}", date_UTC, target)
            Ls = CLR.Ls
            self.diagnostic(Ls, f"LS_C{carbon}_I{iter}", date_UTC, target)
            Lf = CLR.Lf
            self.diagnostic(Lf, f"Lf_C{carbon}_I{iter}", date_UTC, target)

            # Photosynthesis (sunlit)
            if C4:
                An_Sun = self.C4_photosynthesis(
                    Tf_K=Tf_Sun_K,  # sunlit leaf temperature (Tf) [K]
                    Ci=Ci_Sun,  # sunlit intercellular CO2 concentration (Ci) [umol mol-1]
                    APAR=APAR_Sun,  # sunlit absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                    Vcmax25=Vcmax25_Sun  # sunlit maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                )
            else:
                An_Sun = self.C3_photosynthesis(
                    Tf_K=Tf_Sun_K,  # sunlit leaf temperature (Tf) [K]
                    Ci=Ci_Sun,  # sunlit intercellular CO2 concentration (Ci) [umol mol-1]
                    APAR=APAR_Sun,  # sunlit absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                    Vcmax25=Vcmax25_Sun,  # sunlit maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                    Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                    alf=alf  # TODO document alf
                )

            self.diagnostic(An_Sun, f"An_Sun_C{carbon}_I{iter}", date_UTC, target)

            # Energy balance (sunlit)
            # EB:[Rn, LE, H, Tf, gs, Ci]
            EB_Sun = self.energy_balance(
                date_UTC=date_UTC,
                target=target,
                An=An_Sun,  # net assimulation (An) [umol m-2 s-1]
                ASW=ASW_Sun,  # total absorbed shortwave radiation by sunlit canopy (ASW) [umol m-2 s-1]
                ALW=ALW_Sun,  # total absorbed longwave radiation by sunlit canopy (ALW) [umol m-2 s-1]
                Tf_K=Tf_Sun_K,  # sunlit leaf temperature (Tf) [K]
                Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                Ca=Ca,  # ambient CO2 concentration (Ca) [umol mol-1]
                Ta_K=Ta_K,  # air temperature (Ta) [K]
                RH=RH,  # relative humidity (RH) [-]
                VPD_Pa=VPD_Pa,  # water vapour deficit (VPD) [Pa]
                desTa=desTa,  # 1st derivative of saturated vapour pressure (desTa)
                ddesTa=ddesTa,  # 2nd derivative of saturated vapour pressure (ddesTa)
                gamma=gamma,  # psychrometric constant (gamma) [pa K-1]
                Cp=Cp,  # TODO document specific heat
                rhoa=rhoa,  # air density (rhoa) [kg m-3]
                Rc=Rc,  # TODO is this Ra or Rc in Ball-Berry?
                m=m,  # Ball-Berry slope (m) [-]
                b0=b0,  # Ball-Berry intercept (b0) [-]
                flgC4=C4,  # process for C4 plants instead of C3
                carbon=carbon,
                iter=iter
            )

            # Get EB_Sun values
            Rn_Sun = rt.where(np.isnan(EB_Sun.Rn), Rn_Sun, EB_Sun.Rn)
            self.diagnostic(Rn_Sun, f"Rn_Sun_C{carbon}_I{iter}", date_UTC, target)
            LE_Sun = rt.where(np.isnan(EB_Sun.LE), LE_Sun, EB_Sun.LE)
            self.diagnostic(LE_Sun, f"LE_Sun_C{carbon}_I{iter}", date_UTC, target)
            H_Sun = rt.where(np.isnan(EB_Sun.H), H_Sun, EB_Sun.H)
            self.diagnostic(H_Sun, f"H_Sun_C{carbon}_I{iter}", date_UTC, target)
            Tf_Sun_K = rt.where(np.isnan(EB_Sun.Tf), Tf_Sun_K, EB_Sun.Tf)
            self.diagnostic(Tf_Sun_K, f"Tf_Sun_K_C{carbon}_I{iter}", date_UTC, target)
            # gs_Sun = rt.where(np.isnan(EB_Sun.gs), gs_Sun, EB_Sun.gs)
            # self.diagnostic(gs_Sun, f"gs_Sun_C{carbon}_I{iter}", date_UTC, target)
            Ci_Sun = rt.where(np.isnan(EB_Sun.Ci), Ci_Sun, EB_Sun.Ci)
            self.diagnostic(Ci_Sun, f"Ci_Sun_C{carbon}_I{iter}", date_UTC, target)

            # Photosynthesis (shade)
            if C4:
                An_Sh = self.C4_photosynthesis(
                    Tf_K=Tf_Sh_K,  # shaded leaf temperature (Tf) [K]
                    Ci=Ci_Sh,  # shaded intercellular CO2 concentration (Ci) [umol mol-1]
                    APAR=APAR_Sh,  # shaded absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                    Vcmax25=Vcmax25_Sh  # shaded maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                )
            else:
                An_Sh = self.C3_photosynthesis(
                    Tf_K=Tf_Sh_K,  # shaded leaf temperature (Tf) [K]
                    Ci=Ci_Sh,  # shaed intercellular CO2 concentration (Ci) [umol mol-1]
                    APAR=APAR_Sh,  # shaded absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                    Vcmax25=Vcmax25_Sh,  # shaded maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                    Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                    alf=alf  # TODO document alf
                )

            self.diagnostic(An_Sh, f"An_Sh_C{carbon}_I{iter}", date_UTC, target)

            # Energy balance (shade)
            # EB_Sh:[Rn_Sh, LE_Sh, H_Sh, Tf_Sh, gs_Sh, Ci_Sh]
            EB_Sh = self.energy_balance(
                date_UTC=date_UTC,
                target=target,
                An=An_Sh,  # net assimulation (An) [umol m-2 s-1]
                ASW=ASW_Sh,  # total absorbed shortwave radiation by shaded canopy (ASW) [umol m-2 s-1]
                ALW=CLR.ALW_Sh,  # total absorbed longwave radiation by shaded canopy (ALW) [umol m-2 s-1]
                Tf_K=Tf_Sh_K,  # shaded leaf temperature (Tf) [K]
                Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                Ca=Ca,  # ambient CO2 concentration (Ca) [umol mol-1]
                Ta_K=Ta_K,  # air temperature (Ta) [K]
                RH=RH,  # relative humidity (RH) [-]
                VPD_Pa=VPD_Pa,  # water vapour deficit (VPD) [Pa]
                desTa=desTa,  # 1st derivative of saturated vapour pressure (desTa)
                ddesTa=ddesTa,  # 2nd derivative of saturated vapour pressure (ddesTa)
                gamma=gamma,  # psychrometric constant (gamma) [pa K-1]
                Cp=Cp,  # TODO document specific heat
                rhoa=rhoa,  # air density (rhoa) [kg m-3]
                Rc=Rc,  # TODO is this Ra or Rc in Ball-Berry?
                m=m,  # Ball-Berry slope (m) [-]
                b0=b0,  # Ball-Berry intercept (b0) [-]
                flgC4=C4,  # process for C4 plants instead of C3
                carbon=carbon,
                iter=iter
            )

            # Get EB_Sh values
            Rn_Sh = rt.where(np.isnan(EB_Sh.Rn), Rn_Sh, EB_Sh.Rn)
            self.diagnostic(Rn_Sh, f"Rn_Sh_C{carbon}_I{iter}", date_UTC, target)
            LE_Sh = rt.where(np.isnan(EB_Sh.LE), LE_Sh, EB_Sh.LE)
            self.diagnostic(LE_Sh, f"LE_Sh_C{carbon}_I{iter}", date_UTC, target)
            # H_Sh = rt.where(np.isnan(EB_Sh.H), H_Sh, EB_Sh.H)
            # self.diagnostic(H_Sh, f"H_Sh_C{carbon}_I{iter}", date_UTC, target)
            Tf_Sh_K = rt.where(np.isnan(EB_Sh.Tf), Tf_Sh_K, EB_Sh.Tf)
            self.diagnostic(Tf_Sh_K, f"Tf_Sh_K_C{carbon}_I{iter}", date_UTC, target)
            # gs_Sh = rt.where(np.isnan(EB_Sh.gs), gs_Sh, EB_Sh.gs)
            # self.diagnostic(gs_Sh, f"gs_Sh_C{carbon}_I{iter}", date_UTC, target)
            Ci_Sh = rt.where(np.isnan(EB_Sh.Ci), Ci_Sh, EB_Sh.Ci)
            self.diagnostic(Ci_Sh, f"Ci_Sh_C{carbon}_I{iter}", date_UTC, target)

            # Soil
            # SOIL:[Rn, LE, H, Ts]
            SOIL = self.soil(
                Ts=Ts_K,
                Ta=Ta_K,
                G=G,
                VPD=VPD_Pa,
                RH=RH,
                gamma=gamma,
                Cp=Cp,
                rhoa=rhoa,
                desTa=desTa,
                Rs=Rs,
                ASW_Soil=ASW_Soil,
                ALW_Soil=CLR.ALW_Soil,
                Ls=CLR.Ls,
                epsa=epsa
            )

            Rn_Soil = rt.where(np.isnan(SOIL.Rn), Rn_Soil, SOIL.Rn)
            self.diagnostic(Rn_Soil, f"Rn_Soil_C{carbon}_I{iter}", date_UTC, target)
            LE_Soil = rt.where(np.isnan(SOIL.LE), LE_Soil, SOIL.LE)
            self.diagnostic(LE_Soil, f"LE_Soil_C{carbon}_I{iter}", date_UTC, target)
            # H_Soil = rt.where(np.isnan(SOIL.H), H_Soil, SOIL.H)
            # self.diagnostic(H_Soil, f"H_Soil_C{carbon}_I{iter}", date_UTC, target)
            Ts_K = rt.where(np.isnan(SOIL.Ts), Ts_K, SOIL.Ts)
            self.diagnostic(Ts_K, f"Ts_K_C{carbon}_I{iter}", date_UTC, target)

            # Composite components
            # Tf = (Tf_Sun.^4.*fSun + Tf_Sh.^4.*(1-fSun)).^0.25;
            Tf_K_new = (((Tf_Sun_K ** 4) * fSun + (Tf_Sh_K ** 4) * (1 - fSun)) ** 0.25)
            Tf_K = rt.where(np.isnan(Tf_K_new), Tf_K, Tf_K_new)
            self.diagnostic(Tf_K, f"Tf_K_C{carbon}_I{iter}", date_UTC, target)

        self.diagnostic(LE_Soil, f"LE_soil_C{carbon}", date_UTC, target)
        LE_canopy = rt.clip(LE_Sun + LE_Sh, 0, 1000)
        self.diagnostic(LE_canopy, f"LE_canopy_C{carbon}", date_UTC, target)
        LE = rt.clip(LE_Sun + LE_Sh + LE_Soil, 0, 1000)  # [W m-2]
        self.diagnostic(LE, f"LE_C{carbon}", date_UTC, target)
        GPP = rt.clip(An_Sun + An_Sh, 0, GPP_max)  # [umol m-2 s-1]
        self.diagnostic(GPP, f"GPP_C{carbon}", date_UTC, target)
        Rn_canopy = rt.clip(Rn_Sun + Rn_Sh, 0, None)
        self.diagnostic(Rn_canopy, f"Rn_canopy_C{carbon}", date_UTC, target)
        self.diagnostic(Rn_Soil, f"Rn_soil_C{carbon}", date_UTC, target)
        Rn = rt.clip(Rn_Sun + Rn_Sh + Rn_Soil, 0, 1000)  # [W m-2]
        self.diagnostic(Rn, f"Rn_C{carbon}", date_UTC, target)
        # CWF.Gs = (gs_Sun * fSun + gs_Sh * (1.0 - fSun)) * 1000.0  # [mm s-1]

        # Meng et al., 2009
        # Lsoil = (1.0 - FVC) * (1.0 - epss) * CLR.La + (1.0 - FVC) * sigma * epss * pow(Ts_K, 4)
        # Lcanopy = FVC * (1.0 - epsf) * CLR.La + FVC * sigma * epsf * pow(Tf_K, 4)
        # L = Lcanopy + Lsoil
        # E = FVC * epsf + (1.0 - FVC) * epss
        # CWF.LST = np.pow((L / E / sigma), 0.25)
        # CWF.LST = rt.where(CWF.LST <= 273.16 - 60.0, np.nan, CWF.LST)
        # CWF.LST = rt.where(CWF.LST >= 273.16 + 60.0, np.nan, CWF.LST)

        CWF.LE_canopy = LE_canopy
        CWF.LE_soil = LE_Soil
        CWF.LE = LE
        CWF.GPP = GPP
        CWF.Rn_canopy = Rn_canopy
        CWF.Rn_soil = Rn_Soil
        CWF.Rn = Rn

        return CWF

    def interpolate_fC4(self, C3: Raster, C4: Raster, fC4: Raster):
        return C3 * (1 - fC4) + C4 * fC4

    def BESS(
            self,
            geometry: RasterGeometry,
            target: str,
            time_UTC: datetime or str,
            ST_K: Raster,
            NDVI: Raster,
            albedo: Raster,
            Ta_K: Raster = None,
            RH: Raster = None,
            SM: Raster = None,
            Ca: Raster = None,
            wind_speed: Raster = None,
            COT: Raster = None,
            AOT: Raster = None,
            vapor_gccm: Raster = None,
            ozone_cm: Raster = None,
            elevation_km: Raster = None,
            SZA: Raster = None,
            Rg: Raster = None,
            VISdiff: Raster = None,
            VISdir: Raster = None,
            NIRdiff: Raster = None,
            NIRdir: Raster = None,
            UV: Raster = None,
            KG_climate: Raster = None,
            fC4: Raster = None,
            alf: Raster = None,
            kn: Raster = None,
            b0_C3: Raster = None,
            m_C3: Raster = None,
            m_C4: Raster = None,
            peakVCmax_C3: Raster = None,
            peakVCmax_C4: Raster = None,
            canopy_height_meters: Raster = None,
            CI: Raster = None,
            NDVI_minimum: Raster = None,
            NDVI_maximum: Raster = None,
            water: Raster = None,
            output_variables: List[str] = DEFAULT_OUTPUT_VARIABLES):
        b0_C4 = 0.04

        results = {}

        self.logger.info(
            f"processing {cl.name('BESS')} " +
            f"tile {cl.place(target)} {cl.val(geometry.shape)} " +
            f"at {cl.time(time_UTC)} UTC"
        )

        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        date_UTC = time_UTC.date()
        hour_of_day = self.hour_of_day(time_UTC=time_UTC, geometry=geometry)
        self.diagnostic(hour_of_day, "hour_of_day", date_UTC, target)
        day_of_year = self.day_of_year(time_UTC=time_UTC, geometry=geometry)
        self.diagnostic(ST_K, "ST_K", date_UTC, target)
        self.diagnostic(NDVI, "NDVI", date_UTC, target)
        self.diagnostic(albedo, "albedo", date_UTC, target)

        if water is None:
            water = NDVI <= 0

        self.diagnostic(water, "water", date_UTC, target)

        if elevation_km is None:
            elevation_km = self.elevation_km(geometry)

        self.diagnostic(elevation_km, "elevation_km", date_UTC, target)

        if SZA is None:
            SZA = self.SZA(day_of_year=day_of_year, hour_of_day=hour_of_day, geometry=geometry)

        self.diagnostic(SZA, "SZA", date_UTC, target)

        if Rg is None or VISdiff is None or VISdir is None or NIRdiff is None or NIRdir is None or UV is None:
            Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = self.FLiES(
                geometry=geometry,
                target=target,
                time_UTC=time_UTC,
                albedo=albedo,
                COT=COT,
                AOT=AOT,
                vapor_gccm=vapor_gccm,
                ozone_cm=ozone_cm,
                elevation_km=elevation_km,
                SZA=SZA,
                KG_climate=KG_climate
            )

            Rg = Rg.mask(~np.isnan(ST_K))

            self.diagnostic(Ra, "Ra", date_UTC, target)
            self.diagnostic(Rg, "Rg", date_UTC, target)
            self.diagnostic(UV, "UV", date_UTC, target)
            self.diagnostic(VIS, "VIS", date_UTC, target)
            self.diagnostic(NIR, "NIR", date_UTC, target)
            self.diagnostic(VISdiff, "VISdiff", date_UTC, target)
            self.diagnostic(NIRdiff, "NIRdiff", date_UTC, target)
            self.diagnostic(VISdir, "VISdir", date_UTC, target)
            self.diagnostic(NIRdir, "NIRdir", date_UTC, target)

        if canopy_height_meters is None:
            canopy_height_meters = self.canopy_height_meters(geometry=geometry)

        canopy_height_meters = rt.where(np.isnan(canopy_height_meters), 0, canopy_height_meters)

        self.diagnostic(canopy_height_meters, "canopy_height_meters", date_UTC, target)

        if Ta_K is None:
            Ta_K = self.Ta_K(
                time_UTC=time_UTC,
                geometry=geometry,
                ST_K=ST_K,
                water=water,
                apply_scale=False,
                apply_bias=True
            )

        self.diagnostic(Ta_K, "Ta_K", date_UTC, target)

        if SM is None:
            SM = self.SM(time_UTC=time_UTC, geometry=geometry, ST_fine=ST_K, NDVI_fine=NDVI, water=water)

        self.diagnostic(SM, "SM", date_UTC, target)

        if "SM" in output_variables:
            results["SM"] = SM

        if RH is None:
            RH = self.RH(time_UTC=time_UTC, geometry=geometry, SM=SM, ST_K=ST_K, water=water)

        self.diagnostic(RH, "RH", date_UTC, target)

        if "RH" in output_variables:
            results["RH"] = RH

        SVP_Pa = self.SVP_Pa_from_Ta_K(Ta_K)

        self.diagnostic(SVP_Pa, "SVP_Pa", date_UTC, target)

        if "SVP_Pa" in output_variables:
            results["SVP_Pa"] = SVP_Pa

        Ea_Pa = RH * SVP_Pa

        self.diagnostic(Ea_Pa, "Ea_Pa", date_UTC, target)

        if "Ea_Pa" in output_variables:
            results["Ea_Pa"] = Ea_Pa

        if "Ea_kPa" in output_variables:
            results["Ea_kPa"] = Ea_Pa / 1000

        if wind_speed is None:
            wind_speed = self.wind_speed(time_UTC=time_UTC, geometry=geometry)

        self.diagnostic(wind_speed, "wind_speed", date_UTC, target)

        MET = self.meteorology(
            date_UTC=date_UTC,
            target=target,
            day_of_year=day_of_year,
            hour_of_day=hour_of_day,
            latitude=geometry.lat,
            elevation_m=elevation_km * 1000,
            SZA=SZA,
            Ta_K=Ta_K,
            Ea_Pa=Ea_Pa,
            Rg=Rg,
            wind_speed_mps=wind_speed,
            canopy_height_meters=canopy_height_meters
        )

        RH = MET.RH

        if "Ta_K" in output_variables:
            results["Ta_K"] = Ta_K

        if "Ta_C" in output_variables:
            results["Ta_C"] = Ta_K - 273.15

        if "RH" in output_variables:
            results["RH"] = RH

        if fC4 is None:
            fC4 = self.fC4(geometry=geometry)

        self.diagnostic(fC4, "fC4", date_UTC, target)

        if alf is None:
            alf = self.alf(geometry=geometry)

        self.diagnostic(alf, "alf", date_UTC, target)

        if kn is None:
            kn = self.kn(geometry=geometry)

        self.diagnostic(kn, "kn", date_UTC, target)

        if b0_C3 is None:
            b0_C3 = self.b0_C3(geometry=geometry)

        self.diagnostic(b0_C3, "b0_C3", date_UTC, target)

        if m_C3 is None:
            m_C3 = self.m_C3(geometry=geometry)

        self.diagnostic(m_C3, "m_C3", date_UTC, target)

        if m_C4 is None:
            m_C4 = self.m_C4(geometry=geometry)

        self.diagnostic(m_C4, "m_C4", date_UTC, target)

        if peakVCmax_C3 is None:
            peakVCmax_C3 = self.peakVCmax_C3(geometry=geometry)

        self.diagnostic(peakVCmax_C3, "peakVCmax_C3", date_UTC, target)

        if peakVCmax_C4 is None:
            peakVCmax_C4 = self.peakVCmax_C4(geometry=geometry)

        self.diagnostic(peakVCmax_C4, "peakVCmax_C4", date_UTC, target)

        if CI is None:
            CI = self.CI(geometry=geometry)

        self.diagnostic(CI, "CI", date_UTC, target)

        if NDVI_minimum is None:
            NDVI_minimum = self.NDVI_minimum(geometry=geometry)

        self.diagnostic(NDVI_minimum, "NDVI_minimum", date_UTC, target)
        LAI_minimum = self.NDVI_to_LAI(NDVI_minimum)
        self.diagnostic(LAI_minimum, "LAI_minimum", date_UTC, target)

        if NDVI_maximum is None:
            NDVI_maximum = self.NDVI_maximum(geometry=geometry)

        self.diagnostic(NDVI_maximum, "NDVI_maximum", date_UTC, target)
        LAI_maximum = self.NDVI_to_LAI(NDVI_maximum)
        self.diagnostic(LAI_maximum, "LAI_maximum", date_UTC, target)
        self.diagnostic(NDVI, "NDVI", date_UTC, target)
        LAI = self.NDVI_to_LAI(NDVI)
        self.diagnostic(LAI, "LAI", date_UTC, target)
        FVC = self.NDVI_to_FVC(NDVI)
        self.diagnostic(FVC, "FVC", date_UTC, target)

        VCmax_C3_sun, VCmax_C4_sun, VCmax_C3_sh, VCmax_C4_sh = self.VCmax(
            date_UTC=date_UTC,
            target=target,
            peakVCmax_C3=peakVCmax_C3,
            peakVCmax_C4=peakVCmax_C4,
            LAI=LAI,
            SZA=SZA,
            LAI_minimum=LAI_minimum,
            LAI_maximum=LAI_maximum,
            fC4=fC4,
            kn=kn
        )

        albedo_NWP = self.GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)
        RVIS_NWP = self.GEOS5FP_connection.ALBVISDR(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)
        RVIS = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
        self.diagnostic(RVIS, "RVIS", date_UTC, target)
        RNIR_NWP = self.GEOS5FP_connection.ALBNIRDR(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)
        RNIR = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
        self.diagnostic(RNIR, "RNIR", date_UTC, target)
        PARDir = VISdir
        self.diagnostic(PARDir, "PARDir", date_UTC, target)

        CSR = self.canopy_shortwave_radiation(
            date_UTC=date_UTC,
            target=target,
            PARDiff=VISdiff,
            PARDir=VISdir,
            NIRDiff=NIRdiff,
            NIRDir=NIRdir,
            UV=UV,
            SZA=SZA,
            LAI=LAI,
            CI=CI,
            RVIS=RVIS,
            RNIR=RNIR
        )

        fSun = CSR.fSun
        self.diagnostic(fSun, "fSun", date_UTC, target)
        APAR_Sun = CSR.APAR_Sun
        self.diagnostic(APAR_Sun, "APAR_Sun", date_UTC, target)
        APAR_Sh = CSR.APAR_Sh
        self.diagnostic(APAR_Sh, "APAR_Sh", date_UTC, target)
        ASW_Sun = CSR.ASW_Sun
        self.diagnostic(ASW_Sun, "ASW_Sun", date_UTC, target)
        ASW_Sh = CSR.ASW_Sh
        self.diagnostic(ASW_Sh, "ASW_Sh", date_UTC, target)
        ASW_Soil = CSR.ASW_Soil
        self.diagnostic(ASW_Soil, "ASW_Soil", date_UTC, target)
        G = CSR.G
        self.diagnostic(G, "G", date_UTC, target)

        if Ca is None:
            Ca = self.Ca(time_UTC=time_UTC, geometry=geometry)

        self.diagnostic(Ca, "Ca", date_UTC, target)

        CWF_C3 = self.carbon_water_fluxes(
            date_UTC=date_UTC,
            target=target,
            ST_K=ST_K,
            LAI=LAI,
            Ta_K=Ta_K,
            APAR_Sun=APAR_Sun,
            APAR_Sh=APAR_Sh,
            ASW_Sun=ASW_Sun,
            ASW_Sh=ASW_Sh,
            Vcmax25_Sun=VCmax_C3_sun,
            Vcmax25_Sh=VCmax_C3_sh,
            m=m_C3,
            b0=b0_C3,
            fSun=fSun,
            ASW_Soil=ASW_Soil,
            G=G,
            SZA=SZA,
            Ca=Ca,
            Ps_Pa=MET.Ps,
            gamma=MET.gamma,
            Cp=MET.Cp,
            rhoa=MET.rhoa,
            VPD_Pa=MET.VPD,
            RH=MET.RH,
            desTa=MET.desTa,
            ddesTa=MET.ddesTa,
            epsa=MET.epsa,
            Rc=MET.Rc,
            Rs=MET.Rs,
            alf=alf,
            fStress=MET.fStress,
            FVC=FVC,
            C4=False
        )

        CWF_C4 = self.carbon_water_fluxes(
            date_UTC=date_UTC,
            target=target,
            ST_K=ST_K,
            LAI=LAI,
            Ta_K=Ta_K,
            APAR_Sun=APAR_Sun,
            APAR_Sh=APAR_Sh,
            ASW_Sun=ASW_Sun,
            ASW_Sh=ASW_Sh,
            Vcmax25_Sun=VCmax_C4_sun,
            Vcmax25_Sh=VCmax_C4_sh,
            m=m_C4,
            b0=b0_C4,
            fSun=fSun,
            ASW_Soil=ASW_Soil,
            G=G,
            SZA=SZA,
            Ca=Ca,
            Ps_Pa=MET.Ps,
            gamma=MET.gamma,
            Cp=MET.Cp,
            rhoa=MET.rhoa,
            VPD_Pa=MET.VPD,
            RH=MET.RH,
            desTa=MET.desTa,
            ddesTa=MET.ddesTa,
            epsa=MET.epsa,
            Rc=MET.Rc,
            Rs=MET.Rs,
            alf=alf,
            fStress=MET.fStress,
            FVC=FVC,
            C4=True
        )

        GPP = rt.clip(self.interpolate_fC4(CWF_C3.GPP, CWF_C4.GPP, fC4), 0, 50)
        GPP = GPP.mask(~np.isnan(ST_K))
        self.diagnostic(GPP, "GPP", date_UTC, target)

        if "GPP" in output_variables:
            results["GPP"] = GPP

        if "Rg" in output_variables:
            results["Rg"] = Rg

        # Upscale from snapshot to daily
        SFd = MET.SFd
        GPP_daily = 1800 * GPP / SFd * 1e-6 * 12  # Eq. (3) in Ryu et al 2008
        GPP_daily = rt.where(SFd < 0.01, 0, GPP_daily)
        GPP_daily = rt.where(SZA >= 90, 0, GPP_daily)
        self.diagnostic(GPP_daily, "GPP_daily", time_UTC, target)

        Rn = rt.clip(self.interpolate_fC4(CWF_C3.Rn, CWF_C4.Rn, fC4), 0, 1000)
        self.diagnostic(Rn, "Rn", date_UTC, target)

        if "Rn" in output_variables:
            results["Rn"] = Rn

        Rn_soil = rt.clip(self.interpolate_fC4(CWF_C3.Rn_soil, CWF_C4.Rn_soil, fC4), 0, 1000)
        self.diagnostic(Rn_soil, "Rn_soil", date_UTC, target)
        Rn_canopy = rt.clip(self.interpolate_fC4(CWF_C3.Rn_canopy, CWF_C4.Rn_canopy, fC4), 0, 1000)
        self.diagnostic(Rn_canopy, "Rn_canopy", date_UTC, target)
        LE = rt.clip(self.interpolate_fC4(CWF_C3.LE, CWF_C4.LE, fC4), 0, 1000)
        self.diagnostic(LE, "LE", date_UTC, target)

        if "LE" in output_variables:
            results["LE"] = LE

        LE_soil = rt.clip(self.interpolate_fC4(CWF_C3.LE_soil, CWF_C4.LE_soil, fC4), 0, 1000)
        self.diagnostic(LE_soil, "LE_soil", date_UTC, target)
        LE_canopy = rt.clip(self.interpolate_fC4(CWF_C3.LE_canopy, CWF_C4.LE_canopy, fC4), 0, 1000)
        self.diagnostic(LE_canopy, "LE_canopy", date_UTC, target)

        return results

    def SVP_kPa_from_Ta_C(self, Ta_C: Raster) -> Raster:
        """
        saturation vapor pressure in kPa from air temperature in celsius
        :param Ta_C: air temperature in celsius
        :return: saturation vapor pressure in kilopascals
        """
        return 0.611 * np.exp((Ta_C * 17.27) / (Ta_C + 237.7))

    def SVP_Pa_from_Ta_K(self, Ta_K: Raster) -> Raster:
        """
        saturation vapor pressure in kPa from air temperature in celsius
        :param Ta_K: air temperature in Kelvin
        :return: saturation vapor pressure in kilopascals
        """
        Ta_C = Ta_K - 273.15
        SVP_kPa = self.SVP_kPa_from_Ta_C(Ta_C)
        SVP_Pa = SVP_kPa * 1000

        return SVP_Pa

    def SM(
            self,
            time_UTC: datetime,
            geometry: RasterGrid,
            ST_fine: Raster,
            NDVI_fine: Raster,
            water: Raster,
            coarse_cell_size: int = GEOS_IN_SENTINEL_COARSE_CELL_SIZE,
            fvlim=0.5,
            a=0.5,
            smoothing="linear") -> Raster:
        if not self.downscale_moisture:
            return self.SM_coarse(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

        self.logger.info("downscaling GEOS-5 FP top level soil moisture raster in cubic meters per cubic meters")
        fine = geometry
        coarse = geometry.rescale(coarse_cell_size)
        ST_fine = ST_fine.mask(~water)
        NDVI_fine = NDVI_fine.mask(~water)
        SM_coarse = self.SM_coarse(time_UTC=time_UTC, geometry=coarse, resampling="cubic")
        FVC_fine = self.NDVI_to_FVC(NDVI_fine)
        soil_fine = FVC_fine < fvlim
        Tmin_coarse = ST_fine.to_geometry(coarse, resampling="min")
        Tmax_coarse = ST_fine.to_geometry(coarse, resampling="max")
        Ts_fine = ST_fine.mask(soil_fine)
        Tsmin_coarse = Ts_fine.to_geometry(coarse, resampling="min").fill(Tmin_coarse)
        Tsmax_coarse = Ts_fine.to_geometry(coarse, resampling="max").fill(Tmax_coarse)
        ST_coarse = ST_fine.to_geometry(coarse, resampling="average")
        SEE_coarse = (Tsmax_coarse - ST_coarse) / rt.clip(Tsmax_coarse - Tsmin_coarse, 1, None)
        SM_SEE_proportion = (SM_coarse / SEE_coarse).to_geometry(fine, resampling=smoothing)
        Tsmax_fine = Tsmax_coarse.to_geometry(geometry, resampling=smoothing)
        Tsrange_fine = (Tsmax_coarse - Tsmin_coarse).to_geometry(fine, resampling=smoothing)
        SEE_fine = (Tsmax_fine - ST_fine) / rt.clip(Tsrange_fine, 1, None)
        SM_resampled = self.GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=fine, resampling="cubic")
        SEE_mean = SEE_coarse.to_geometry(fine, resampling=smoothing)
        SM_fine = rt.clip(SM_resampled + a * SM_SEE_proportion * (SEE_fine - SEE_mean), 0, 1)
        SM_fine = SM_fine.mask(~water)

        return SM_fine

    def SM_coarse(self, time_UTC: datetime, geometry: RasterGeometry, resampling: str = None) -> Raster:
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        return self.GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
