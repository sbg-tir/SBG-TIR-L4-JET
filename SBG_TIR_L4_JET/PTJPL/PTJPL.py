"""
This module contains the core physics of the PT-JPL algorithm.
"""
import logging
import warnings
from datetime import datetime
from os.path import join, abspath, dirname, expanduser
from typing import Callable, Dict, List

import numpy as np
from scipy.stats import zscore
from datetime import date

import rasters as rt
from rasters import Raster, RasterGeometry, RasterGrid

from modisci import MODISCI
from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP

from breathing_earth_system_simulator import BESS, DEFAULT_DOWNSCALE_AIR, DEFAULT_DOWNSCALE_HUMIDITY, DEFAULT_DOWNSCALE_MOISTURE
from ..SRTM import SRTM

__author__ = "Gregory Halverson"

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_GEOS5FP_DOWNLOAD = "GEOS5FP_download_directory"
DEFAULT_GEOS5FP_PRODUCTS = "GEOS5FP_products"
DEFAULT_PTJPL_INTERMEDIATE = "PTJPL_intermediate"

DEFAULT_RESAMPLING = "cubic"
DEFAULT_PREVIEW_QUALITY = 20

DEFAULT_OUTPUT_VARIABLES = [
    "Rn",
    "LE",
    "ETc",
    "ETi",
    "ETs",
    "ET",
    "ESI",
    "WUE"
]

FLOOR_TOPT = True

# Priestley-Taylor coefficient alpha
PT_ALPHA = 1.26
BETA = 1.0

# maximum proportion of soil heat flux out of net radiation to the soil
G_MAX_PROPORTION = 0.35
# G_MAX_PROPORTION_WATER = 0.5
W_MAX_PROPORTION = 0.5

# psychrometric constant gamma in pascals over kelvin
# same as value for ventilated (Asmann type) psychrometers, with an air movement of some 5 m/s
# http://www.fao.org/docrep/x0490e/x0490e07.htm
PSYCHROMETRIC_GAMMA = 0.0662  # Pa/K

KRN = 0.6
KPAR = 0.5

STEFAN_BOLTZMAN_CONSTANT = 5.67036713e-8  # SI units watts per square meter per kelvin to the fourth

class GEOS5FPNotAvailableError(IOError):
    pass


class PTJPL(BESS):
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
            intermediate_directory=None,
            preview_quality: int = DEFAULT_PREVIEW_QUALITY,
            ANN_model: Callable = None,
            ANN_model_filename: str = None,
            resampling: str = DEFAULT_RESAMPLING,
            downscale_air: bool = DEFAULT_DOWNSCALE_AIR,
            downscale_humidity: bool = DEFAULT_DOWNSCALE_HUMIDITY,
            downscale_moisture: bool = DEFAULT_DOWNSCALE_MOISTURE,
            floor_Topt: bool = FLOOR_TOPT,
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
            intermediate_directory = join(working_directory, DEFAULT_PTJPL_INTERMEDIATE)

        super(PTJPL, self).__init__(
            working_directory=working_directory,
            static_directory=static_directory,
            SRTM_connection=SRTM_connection,
            SRTM_download=SRTM_download,
            GEOS5FP_connection=GEOS5FP_connection,
            GEOS5FP_download=GEOS5FP_download,
            GEOS5FP_products=GEOS5FP_products,
            GEDI_connection=GEDI_connection,
            GEDI_download=GEDI_download,
            ORNL_connection=ORNL_connection,
            CI_directory=CI_directory,
            intermediate_directory=intermediate_directory,
            preview_quality=preview_quality,
            ANN_model=ANN_model,
            ANN_model_filename=ANN_model_filename,
            resampling=resampling,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution,
            include_preview=include_preview,
            downscale_air=downscale_air,
            downscale_humidity=downscale_humidity,
            downscale_moisture=downscale_moisture
        )

        self.downscale_air = downscale_air
        self.downscale_humidity = downscale_humidity
        self.floor_Topt = floor_Topt

    def load_Topt(self, geometry: RasterGeometry) -> Raster:
        SCALE_FACTOR = 0.01
        filename = join(abspath(dirname(__file__)), "Topt_mean_CMG_int16.tif")
        image = rt.clip(rt.Raster.open(filename, geometry=geometry, resampling="cubic") * SCALE_FACTOR, 0, None)
        image.nodata = np.nan

        return image

    def load_fAPARmax(self, geometry: RasterGeometry) -> Raster:
        SCALE_FACTOR = 0.0001
        filename = join(abspath(dirname(__file__)), "fAPARmax_mean_CMG_int16.tif")
        image = rt.clip(rt.Raster.open(filename, geometry=geometry, resampling="cubic") * SCALE_FACTOR, 0, None)
        image.nodata = np.nan

        return image

    def SAVI_from_NDVI(self, NDVI: Raster) -> Raster:
        """
        Linearly calculates Soil-Adjusted Vegetation Index from ST_K.
        :param NDVI: normalized difference vegetation index clipped between 0 and 1
        :return: soil-adjusted vegetation index
        """
        return NDVI * 0.45 + 0.132

    def fAPAR_from_savi(self, SAVI: Raster) -> Raster:
        """
        Linearly calculates fraction of absorbed photosynthetically active radiation from SAVI.
        :param SAVI: soil adjusted vegetation index
        :return: fraction of absorbed photosynthetically active radiation
        """
        return rt.clip(SAVI * 1.3632 + -0.048, 0, 1)

    def fAPAR_from_NDVI(self, NDVI: Raster) -> Raster:
        """
        Calculates fraction of absorbed photosynthetically active radiation from ST_K.
        :param NDVI: ST_K clipped between 0 and 1
        :return:
        """
        return self.fAPAR_from_savi(self.SAVI_from_NDVI(NDVI))

    def fIPAR_from_NDVI(self, NDVI: Raster) -> Raster:
        """
        Calculate fraction of intercepted photosynthetically active radiation from ST_K
        :param NDVI: ST_K clipped between 0 and 1
        :return: fraction of intercepted photosynthetically active radiation
        """
        return rt.clip(NDVI - 0.05, 0, 1)



    def delta_from_Ta(self, Ta_C: Raster) -> Raster:
        """
        Calculates slope of saturation vapor pressure to air temperature.
        :param Ta_C: air temperature in celsius
        :return: slope of the vapor pressure curve in kilopascals/kelvin
        """
        return 4098 * (0.6108 * np.exp(17.27 * Ta_C / (237.7 + Ta_C))) / (Ta_C + 237.3) ** 2

    def kelvin_to_celsius(self, temperature_K: Raster) -> Raster:
        """
        Reduces temperature in kelvin to celsius.
        :param temperature_K: temperature in kelvin
        :return: temperature in celsius
        """
        return temperature_K - 273.15

    def pascal_to_kilopascal(self, pressure_Pa: Raster) -> Raster:
        """
        Scales order of magnitude of pressure in pascals to kilopascals.
        :param pressure_Pa: pressure in pascals
        :return: pressure in kilopascals
        """
        return pressure_Pa / 1000.0

    def fwet_from_RH(self, RH: Raster) -> Raster:
        """
        Calculates relative surface wetness from relative humidity.
        :param RH: relative humidity as a proportion between 0 and 1
        :return: relative surface wetness as a proportion between 0 and 1
        """
        return RH ** 4

    def calculate_vapor(self, LE_daily: Raster, daylight_hours: Raster) -> Raster:
        # convert length of day in hours to seconds
        daylight_seconds = daylight_hours * 3600.0

        # constant latent heat of vaporization for water: the number of joules of energy it takes to evaporate one kilogram
        LATENT_VAPORIZATION_JOULES_PER_KILOGRAM = 2450000.0

        # factor seconds out of watts to get joules and divide by latent heat of vaporization to get kilograms
        ET = rt.clip(LE_daily * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)

        return ET

    def water_heat_flux(
            self,
            WST_C: Raster,
            Td_C: Raster,
            wind_speed: Raster,
            SWnet: Raster) -> Raster:
        """
        Water heat flux method from AquaSEBS
        http://www.mdpi.com/2072-4292/8/7/583

        There’s a water-surface evaporation paper with a model called AquaSEBS:
        http://www.mdpi.com/2072-4292/8/7/583

        They have a solid methodology for G over water.

        T0: skin-surface temperature
        Td: near-surface dew-point
        u: wind-speed
        Rs: net shortwave radiation

        :param WST_C: water surface temperature Celsius
        :param Td_C: dew-point temperature Celsius
        :param wind_speed: wind speed meters per second
        :param SWnet: net shortwave Watts per square meter
        :return:
        """

        # TODO determine units for wind speed

        Tn = 0.5 * (WST_C - Td_C)
        eta = 0.35 + 0.015 * WST_C + 0.0012 * (Tn ** 2)
        S = 3.3 * wind_speed
        beta = 4.5 + 0.05 * WST_C + (eta + 0.47) * S
        Te = Td_C + SWnet * beta
        W = beta * (Te - WST_C)

        return W

    def Ta_C(self, time_UTC: datetime, geometry: RasterGeometry, ST_K: Raster = None) -> Raster:
        Ta_K = self.Ta_K(
            time_UTC=time_UTC,
            geometry=geometry,
            ST_K=ST_K
        )

        Ta_C = Ta_K - 273.15

        return Ta_C

    def Ea_kPa(self, time_UTC: datetime, geometry: RasterGeometry, ST_K: Raster = None) -> Raster:
        Ea_Pa = self.Ea_Pa(
            time_UTC=time_UTC,
            geometry=geometry,
            ST_K=ST_K
        )

        Ea_kPa = Ea_Pa / 1000

        return Ea_kPa

    def SWin(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        self.logger.info("retrieving GEOS-5 FP incoming shortwave raster in Watts per square meter")
        return self.GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

    def wind_speed(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        self.logger.info("retrieving GEOS-5 FP wind speed raster in meters per second")
        return self.GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

    def Rn(
            self,
            date_UTC: date,
            target: str,
            SWin: Raster,
            albedo: Raster,
            ST_C: Raster,
            emissivity: Raster,
            Ea_kPa: Raster,
            Ta_C: Raster,
            cloud_mask: Raster) -> Raster:
        Ea_Pa = Ea_kPa * 1000
        Ta_K = Ta_C + 273.15
        ST_K = ST_C + 273.15

        albedo = rt.clip(albedo, 0, 1)
        self.diagnostic(albedo, "albedo", date_UTC, target)

        # calculate outgoing shortwave from incoming shortwave and albedo
        SWout = rt.clip(SWin * albedo, 0, None)
        self.diagnostic(SWout, "SWout", date_UTC, target)

        # calculate instantaneous net radiation from components
        SWnet = rt.clip(SWin - SWout, 0, None)
        self.diagnostic(SWnet, "SWnet", date_UTC, target)

        # calculate atmospheric emissivity
        eta1 = 0.465 * Ea_Pa / Ta_K
        atmospheric_emissivity = (1 - (1 + eta1) * np.exp(-(1.2 + 3 * eta1) ** 0.5))

        if cloud_mask is None:
            LWin = atmospheric_emissivity * STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4
        else:
            # calculate incoming longwave for clear sky and cloudy
            LWin = rt.where(
                ~cloud_mask,
                atmospheric_emissivity * STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4,
                STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4
            )

        self.diagnostic(LWin, "LWin", date_UTC, target)

        emissivity = rt.clip(emissivity, 0, 1)
        self.diagnostic(emissivity, "emissivity", date_UTC, target)

        # calculate outgoing longwave from land surface temperature and emissivity
        LWout = emissivity * STEFAN_BOLTZMAN_CONSTANT * ST_K ** 4
        self.diagnostic(LWout, "LWout", date_UTC, target)

        # LWnet = rt.clip(LWin - LWout, 0, None)
        LWnet = LWin - LWout
        self.diagnostic(LWnet, "LWnet", date_UTC, target)

        # constrain negative values of instantaneous net radiation
        Rn = rt.clip(SWnet + LWnet, 0, None)
        self.diagnostic(Rn, "Rn", date_UTC, target)

        return Rn

    def PTJPL(
            self,
            geometry: RasterGrid,
            target: str,
            time_UTC: datetime or str,
            ST_C: Raster,
            emissivity: Raster,
            NDVI: Raster,
            albedo: Raster,
            water: Raster = None,
            cloud_mask: Raster = None,
            SWin: Raster = None,
            Ta_C: Raster = None,
            RH: Raster = None,
            Ea_kPa: Raster = None,
            Topt: Raster = None,
            fAPARmax: Raster = None,
            Rn: Raster = None,
            Rn_daily: Raster = None,
            wind_speed: Raster = None,
            output_variables: List[str] = DEFAULT_OUTPUT_VARIABLES) -> Dict[str, Raster]:
        warnings.filterwarnings('ignore')

        results = {}

        # calculate time
        hour_of_day = self.hour_of_day(time_UTC=time_UTC, geometry=geometry)
        day_of_year = self.day_of_year(time_UTC=time_UTC, geometry=geometry)
        SHA = self.SHA_deg_from_doy_lat(day_of_year, geometry.lat)
        sunrise_hour = self.sunrise_from_sha(SHA)
        daylight_hours = self.daylight_from_sha(SHA)
        date_UTC = time_UTC.date()

        self.diagnostic(ST_C, "ST_C", date_UTC, target)

        # calculate meteorology

        if Ta_C is None:
            Ta_C = self.Ta_C(time_UTC=time_UTC, geometry=geometry)

            if self.downscale_air:
                Ta_C = ST_C.contain(zscore(ST_C)) * np.nanstd(Ta_C) + np.nanmean(Ta_C)

            self.diagnostic(Ta_C, "Ta_C", date_UTC, target)

        # calculate saturation vapor pressure in kPa from air temperature in celsius
        # floor saturation vapor pressure at 1
        SVP_kPa = rt.clip(self.SVP_kPa_from_Ta_C(Ta_C), 1, None)
        self.diagnostic(SVP_kPa, "SVP_kPa", date_UTC, target)

        # if Ea_kPa is None:
        #     Ea_kPa = self.Ea_kPa(time_UTC=time_UTC, geometry=geometry)
        #
        #     if self.downscale_humidity:
        #         Ea_kPa = ST_C.contain(zscore(ST_C)) * -np.nanstd(Ea_kPa) + np.nanmean(Ea_kPa)
        #
        #     self.diagnostic(Ea_kPa, "Ea_kPa", date_UTC, target)

        ST_K = ST_C + 273.15

        if Ea_kPa is None and RH is None:
            Ea_kPa = self.Ea_kPa(time_UTC=time_UTC, geometry=geometry, ST_K=ST_K)
        elif Ea_kPa is None and RH is not None:
            RH = rt.clip(RH, 0, 1)
            Ea_kPa = RH * SVP_kPa

        # calculate vapor pressure deficit from water vapor pressure
        VPD_kPa = SVP_kPa - Ea_kPa

        # lower bound of vapor pressure deficit is zero, negative values replaced with nodata
        VPD_kPa = rt.where(VPD_kPa < 0, np.nan, VPD_kPa)

        self.diagnostic(VPD_kPa, "VPD_kPa", date_UTC, target)

        if "VPD" in output_variables:
            VPD_hPa = VPD_kPa * 10
            results["VPD"] = VPD_hPa

        # calculate relative humidity from water vapor pressure and saturation vapor pressure
        # upper bound of relative humidity is one, results higher than one are capped at one
        if RH is None:
            RH = rt.clip(Ea_kPa / SVP_kPa, 0, 1)

        self.diagnostic(RH, "RH", date_UTC, target)

        if SWin is None:
            # SWin = self.SWin(time_UTC=time_UTC, geometry=geometry)
            Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = self.FLiES(
                geometry=geometry,
                target=target,
                time_UTC=time_UTC,
                albedo=albedo
            )

            SWin = Rg

        self.diagnostic(SWin, "SWin", date_UTC, target)

        if "SWin" in output_variables:
            results["SWin"] = SWin

        albedo = rt.clip(albedo, 0, 1)
        self.diagnostic(albedo, "albedo", date_UTC, target)

        # calculate outgoing shortwave from incoming shortwave and albedo
        SWout = rt.clip(SWin * albedo, 0, None)
        self.diagnostic(SWout, "SWout", date_UTC, target)

        # calculate instantaneous net radiation from components
        SWnet = rt.clip(SWin - SWout, 0, None)
        self.diagnostic(SWnet, "SWnet", date_UTC, target)

        if Rn is None:
            Rn = self.Rn(
                date_UTC=date_UTC,
                target=target,
                SWin=SWin,
                albedo=albedo,
                ST_C=ST_C,
                emissivity=emissivity,
                Ea_kPa=Ea_kPa,
                Ta_C=Ta_C,
                cloud_mask=cloud_mask
            )

        self.diagnostic(Rn, "Rn", date_UTC, target)

        if "Rn" in output_variables:
            results["Rn"] = Rn

        if Rn_daily is None:
            # integrate net radiation to daily value
            Rn_daily = self.Rn_daily(
                Rn,
                hour_of_day,
                sunrise_hour,
                daylight_hours
            )

            # constrain negative values of daily integrated net radiation
            Rn_daily = rt.clip(Rn_daily, 0, None)

        self.diagnostic(Rn_daily, "Rn_daily", date_UTC, target)

        if "Rn_daily" in output_variables:
            results["Rn_daily"] = Rn_daily

        # calculate relative surface wetness from relative humidity
        fwet = self.fwet_from_RH(RH)
        self.diagnostic(fwet, "fwet", date_UTC, target)

        # calculate slope of saturation to vapor pressure curve Pa/K
        delta = self.delta_from_Ta(Ta_C)
        self.diagnostic(delta, "delta", date_UTC, target)

        # calculate vegetation values

        self.diagnostic(NDVI, "NDVI", date_UTC, target)

        if water is None:
            water = NDVI < 0

        # water-mask NDVI
        NDVI = rt.where(water, np.nan, NDVI)

        # calculate fAPAR from NDVI
        fAPAR = self.fAPAR_from_NDVI(NDVI)
        self.diagnostic(fAPAR, "fAPAR", date_UTC, target)

        # calculate fIPAR from NDVI
        fIPAR = self.fIPAR_from_NDVI(NDVI)
        self.diagnostic(fIPAR, "fIPAR", date_UTC, target)

        # calculate green canopy fraction (fg) from fAPAR and fIPAR, constrained between zero and one
        fg = rt.clip(fAPAR / fIPAR, 0, 1)
        self.diagnostic(fg, "fg", date_UTC, target)

        if "fg" in output_variables:
            results["fg"] = fg

        if fAPARmax is None:
            fAPARmax = self.load_fAPARmax(geometry=geometry)

        self.diagnostic(fAPARmax, "fAPARmax", date_UTC, target)

        # calculate plant moisture constraint (fM) from fraction of photosynthetically active radiation,
        # constrained between zero and one
        fM = rt.clip(fAPAR / fAPARmax, 0.0, 1.0)
        self.diagnostic(fM, "fM", date_UTC, target)

        if "fM" in output_variables:
            results["fM"] = fM

        # calculate soil moisture constraint from mean relative humidity and vapor pressure deficit,
        # constrained between zero and one
        fSM = rt.clip(RH ** (VPD_kPa / BETA), 0.0, 1.0)
        self.diagnostic(fSM, "fSM", date_UTC, target)

        if "fSM" in output_variables:
            results["fSM"] = fSM

        if Topt is None:
            Topt = self.load_Topt(geometry=geometry)

            if self.floor_Topt:
                Topt = rt.where(Ta_C > Topt, Ta_C, Topt)

        self.diagnostic(Topt, "Topt", date_UTC, target)

        if "Topt" in output_variables:
            results["Topt"] = Topt

        # calculate plant temperature constraint (fT) from optimal phenology
        fT = np.exp(-(((Ta_C - Topt) / Topt) ** 2))

        self.diagnostic(fT, "fT", date_UTC, target)

        if "fT" in output_variables:
            results["fT"] = fT

        # calculate leaf area index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LAI = -np.log(1 - fIPAR) * (1 / KPAR)

        self.diagnostic(LAI, "LAI", date_UTC, target)

        # calculate delta / (delta + gamma)
        epsilon = delta / (delta + PSYCHROMETRIC_GAMMA)
        self.diagnostic(epsilon, "epsilon", date_UTC, target)

        # soil evaporation

        # caluclate net radiation of the soil from leaf area index
        Rn_soil = Rn * np.exp(-KRN * LAI)
        self.diagnostic(Rn_soil, "Rn_soil", date_UTC, target)

        # calculate instantaneous soil heat flux from net radiation and fractional vegetation cover
        G = Rn * (0.05 + (1 - rt.clip(fIPAR, 0, 1)) * 0.265)

        # constrain soil heat flux
        G = rt.where(np.isnan(G), np.nan, rt.clip(G, 0, G_MAX_PROPORTION * Rn))
        G = G.mask(~water)
        self.diagnostic(G, "G", date_UTC, target)

        # dew-point temperature in Celsius
        Td_C = Ta_C - ((100 - RH * 100) / 5.0)
        self.diagnostic(Td_C, "Td_C", date_UTC, target)

        if wind_speed is None:
            wind_speed = self.wind_speed(time_UTC=time_UTC, geometry=geometry)

        # water heat flux
        W = self.water_heat_flux(
            ST_C,
            Td_C,
            wind_speed,
            SWnet
        )

        W = rt.clip(W, 0, W_MAX_PROPORTION * Rn)
        W = W.mask(water)
        self.diagnostic(W, "W", date_UTC, target)

        # calculate soil evaporation (LEs) from relative surface wetness, soil moisture constraint,
        # priestley taylor coefficient, epsilon = delta / (delta + gamma), net radiation of the soil,
        # and soil heat flux
        LE_soil = rt.clip((fwet + fSM * (1 - fwet)) * PT_ALPHA * epsilon * (Rn_soil - G), 0, None)
        self.diagnostic(LE_soil, "LE_soil", date_UTC, target)

        # canopy transpiration

        # calculate net radiation of the canopy from net radiation of the soil
        Rn_canopy = Rn - Rn_soil
        self.diagnostic(Rn_canopy, "Rn_canopy", date_UTC, target)

        # calculate potential evapotranspiration (pET) from net radiation, and soil heat flux

        PET_water = PT_ALPHA * epsilon * (Rn - W)
        self.diagnostic(PET_water, "PET_water", date_UTC, target)
        PET_land = PT_ALPHA * epsilon * (Rn - G)
        self.diagnostic(PET_land, "PET_land", date_UTC, target)

        PET = rt.where(
            water,
            PET_water,
            PET_land
        )

        if "PET" in output_variables:
            results["PET"] = PET

        self.diagnostic(PET, "PET", date_UTC, target)

        # calculate canopy transpiration (LEc) from priestley taylor, relative surface wetness,
        # green canopy fraction, plant temperature constraint, plant moisture constraint,
        # epsilon = delta / (delta + gamma), and net radiation of the canopy
        LE_canopy = rt.clip(PT_ALPHA * (1 - fwet) * fg * fT * fM * epsilon * Rn_canopy, 0, None)
        self.diagnostic(LE_canopy, "LE_canopy", date_UTC, target)

        # interception evaporation

        # calculate interception evaporation (LEi) from relative surface wetness and net radiation of the canopy
        LE_interception = rt.clip(fwet * PT_ALPHA * epsilon * Rn_canopy, 0, None)
        self.diagnostic(LE_interception, "LE_interception", date_UTC, target)

        # combined evapotranspiration

        # combine soil evaporation (LEs), canopy transpiration (LEc), and interception evaporation (LEi)
        # into instantaneous evapotranspiration (LE)
        LE = LE_soil + LE_canopy + LE_interception
        LEs = LE_soil / LE
        self.diagnostic(LEs, "LEs", date_UTC, target)

        if "LEs" in output_variables:
            results["LEs"] = LEs

        LEc = LE_canopy / LE
        self.diagnostic(LEc, "LEc", date_UTC, target)

        if "LEc" in output_variables:
            results["LEc"] = LEc

        LEi = LE_interception / LE
        self.diagnostic(LEi, "LEi", date_UTC, target)

        if "LEi" in output_variables:
            results["LEi"] = LEi

        LE = rt.where(water, PET, LE)
        LE = rt.where(np.isnan(LE), PET, LE)
        LE = rt.clip(LE, 0, PET)

        if "LE" in output_variables:
            results["LE"] = LE

        self.diagnostic(LE, "LE", date_UTC, target)

        # daily evapotranspiration

        # calculate evaporative fraction (EF) from evapotranspiration, net radiation, and soil heat flux

        EF = rt.clip(LE / Rn, 0, 1)
        self.diagnostic(EF, "EF", date_UTC, target)

        # calculate daily latent heat flux from evaporative fraction and daily net radiation with minimum of zero
        LE_daily = rt.clip(EF * Rn_daily, 0, Rn_daily)
        self.diagnostic(LE_daily, "LE_daily", date_UTC, target)

        # calculate instantaneous evaporative stress index as the ratio of latent heat flux to potential evapotranspiration
        ESI = rt.clip(LE / PET, 0, 1)
        ESI = ESI.mask(~water)

        if "ESI" in output_variables:
            results["ESI"] = ESI

        self.diagnostic(ESI, "ESI", date_UTC, target)

        # calculate daytime daily evapotranspiration in kilograms equivalent to millimeters
        ET = self.calculate_vapor(LE_daily, daylight_hours)

        if "ET" in output_variables:
            results["ET"] = ET

        self.diagnostic(ET, "ET", date_UTC, target)

        warnings.resetwarnings()

        return results
