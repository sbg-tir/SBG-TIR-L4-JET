"""
This module implements the MOD16 algorithm of evapotranspiration for PT-JPL uncertainty.

This implementation follows the MOD16 Version 1.5 Collection 6 algorithm described in the MOD16 user's guide.
https://landweb.nascom.nasa.gov/QA_WWW/forPage/user_guide/MOD16UsersGuide2016.pdf

Developed by Gregory Halverson in the Jet Propulsion Laboratory Year-Round Internship Program (Columbus Technologies and Services), in coordination with the ECOSTRESS mission and master's thesis studies at California State University, Northridge.
"""
import logging
from typing import Callable
from datetime import datetime
from os.path import join, abspath, dirname, expanduser
import numpy as np
import pandas as pd
from numpy import where, nan, exp, array, isnan, logical_and, clip, float32
import warnings

import rasters as rt
from rasters import Raster, RasterGrid, RasterGeometry
from geos5fp import GEOS5FP

from MCD12C1_2019_v006 import load_MCD12C1_IGBP

from ..FLiES import FLiES
# from ..MCD12.MCD12C1 import MCD12C1
from ..SRTM import SRTM
from ..model.model import DEFAULT_PREVIEW_QUALITY, DEFAULT_RESAMPLING

__author__ = 'Kaniska Mallick, Adam Purdy, Gregory Halverson'

logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_MOD16_INTERMEDIATE = "MOD16_intermediate"

DEFAULT_OUTPUT_VARIABLES = [
    'LEi',
    'LEc',
    'LEs',
    'LE',
    'LE_daily',
    'ET_daily_kg'
]

# TODO need to defend picking arbitrary maximum to avoid extreme values
MAXIMUM_RESISTANCE = 2000.0

# gas constant for dry air in joules per kilogram per kelvin
RD = 286.9

# gas constant for moist air in joules per kilogram per kelvin
RW = 461.5

# specific heat of water vapor in joules per kilogram per kelvin
CPW = 1846.0

# specific heat of dry air in joules per kilogram per kelvin
CPD = 1005.0

# psychrometric constant in pascals per kelvin
GAMMA = 67.0

# Stefan Boltzmann constant
SIGMA = 5.678e-8

# cuticular conductance in meters per second
CUTICULAR_CONDUCTANCE = 0.00001

# lookup table
LUT = pd.read_csv(join(abspath(dirname(__file__)), 'mod16.csv'))


def kelvin_to_celsius(temperature_K):
    """
    Reduces temperature in kelvin to celsius.
    :param temperature_K: temperature in kelvin
    :return: temperature in celsius
    """
    return temperature_K - 273.15


def delta_from_Ta(Ta_K):
    """
    calculates slope of saturation vapor pressure to air temperature.
    :param Ta_K: air temperature in Kelvin
    :return: slope of the vapor pressure curve in kilopascals/kelvin
    """
    # return 4098.0 * (0.6108 * exp(17.27 * Ta_C / (237.7 + Ta_C))) / (Ta_C + 237.3) ** 2
    # print("Ta_C: {}".format(np.nanmean(Ta_K)))
    t = Ta_K + 237.3
    # print("t: {}".format(np.nanmean(t)))
    delta = 4098.0 * (0.6108 * exp(17.27 * Ta_K / t)) / (t * t)
    # print("delta: {}".format(np.nanmean(delta)))

    return delta


def calculate_fwet(RH, water, use_threshold=False):
    """
    calculates relative surface wetness
    :param RH: relative humdity from 0.0 to 1.0
    :return: relative surface wetness from 0.0 to 1.0
    """
    # MOD16 uses the same humidity-based fwet as PT-JPL
    # but MOD16 constrains fwet to zero when humidity is below 70%
    if use_threshold:
        fwet = float32(where(RH >= 0.7, RH ** 4, 0.0001))
    else:
        fwet = float32(clip(RH ** 4.0, 0.0001, None))

    # water-mask relative surface wetness
    fwet = where(water, nan, fwet)

    return fwet


# saturation vapor pressure in kPa from air temperature in celsius
def SVP_from_Ta(Ta_C):
    """
    calculates saturation vapor pressure for air temperature.
    :param Ta_C: air temperature in celsius
    :return: saturation vapor pressure in kilopascals
    """
    return 0.611 * exp((Ta_C * 17.27) / (Ta_C + 237.7))


def calculate_specific_humidity(Ea_Pa, surface_pressure_Pa):
    """
    calculate specific humidity of air
    as a ratio of kilograms of water to kilograms of air and water
    from surface pressure and actual water vapor pressure.
    :param Ea_Pa: actual water vapor pressure in pascals
    :param surface_pressure_Pa: surface pressure in pascals
    :return: specific humidity in kilograms of water per kilograms of air and water
    """
    return ((0.622 * Ea_Pa) / (surface_pressure_Pa - (0.387 * Ea_Pa)))


def air_density(surface_pressure_Pa, Ta_K, specific_humidity):
    # numerator: Pa(N / m ^ 2 = kg * m / s ^ 2); denominator: J / kg / K * K)
    rhoD = surface_pressure_Pa / (RD * Ta_K)

    # calculate air density (rho)
    # in kilograms per cubic meter
    rho = rhoD * ((1.0 + specific_humidity) / (1.0 + specific_humidity * (RW / RD)))

    return rho


def wet_canopy_resistance(conductance, LAI, fwet):
    """
    calculates wet canopy resistance.
    :param conductance: leaf conductance to evaporated water vapor
    :param LAI: leaf-area index
    :param fwet: relative surface wetness
    :return: wet canopy resistance
    """
    resistance = clip(1.0 / clip(conductance * LAI * fwet, 1.0 / MAXIMUM_RESISTANCE, None), 0.0, MAXIMUM_RESISTANCE)
    # resistance = where(logical_or(LAI < 0.0001, fwet < 0.0001), nan, resistance)

    return resistance


def interception(s, Ac, rho, Cp, VPD, fc, rhrc, fwet, rvc, water):
    """
    calculates wet evaporation partition of latent heat flux using MOD16 method.
    :param s: slope of saturation to vapor pressure curve
    :param Ac: available radiation to the canopy
    :param rho: air density
    :param Cp:
    :param VPD: vapor pressure deficit
    :param fc: vegetation fraction
    :param rhrc:
    :param fwet: relative surface wetness
    :param rvc:
    :param water: water mask (true for water, false for not water)
    :return: wet evaporation in watts per square meter
    """

    numerator = (s * Ac + (rho * Cp * VPD * fc / rhrc)) * fwet
    # FIXME the denominator here does not match the formula in the MOD16 users guide
    denominator = s + GAMMA * (rvc / rhrc)
    LEi = numerator / denominator

    # fill wet latent heat flux gaps with zero
    LEi = where(isnan(LEi), 0, LEi)

    # mask wet evaporation for water bodies
    LEi = float32(where(water, nan, LEi))

    return LEi


def correctance_factor(surface_pressure_Pa, Ta_K):
    """
    calculates correctance factor (rcorr)
    for stomatal and cuticular conductances
    from surface pressure and air temperature.
    :param surface_pressure_Pa: surface pressure in pascals
    :param Ta_K: near-surface air temperature in kelvin
    :return: correctance factor (rcorr)
    """
    return 1.0 / ((101300.0 / surface_pressure_Pa) * (Ta_K / 293.15) ** 1.75)


def tmin_factor(Tmin, tmin_open, tmin_close, IGBP, water):
    """
    calculates minimum temperature factor for MOD16 equations.
    :param Tmin: minimum temperature in Celsius
    :param IGBP: IGBP land-cover code
    :return: minimum temperature factor (mTmin)
    """

    # calculate minimum temperature factor using queried open and closed minimum temperatures
    mTmin = where(Tmin >= tmin_open, 1.0, nan)

    mTmin = where(
        logical_and(
            tmin_close < Tmin,
            Tmin < tmin_open
        ),
        (Tmin - tmin_close) / (tmin_open - tmin_close),
        mTmin
    )

    mTmin = where(Tmin <= tmin_close, 0.0, mTmin)

    mTmin = where(water, nan, mTmin)

    return mTmin


def vpd_factor(vpd_open, vpd_close, VPD):
    # calculate VPD factor using queried open and closed VPD
    mVPD = where(VPD <= vpd_open, 1.0, nan)
    mVPD = where(logical_and(vpd_open < VPD, VPD < vpd_close), (vpd_close - VPD) / (vpd_close - vpd_open), mVPD)
    mVPD = where(VPD >= vpd_close, 0.0, mVPD)

    return mVPD


def canopy_conductance(LAI, fwet, gl_sh, gs1, Gcu, water):
    """
    calculate canopy conductance (Cc)
    Canopy conductance (Cc) to transpired water vapor per unit LAI is derived from stomatal
    and cuticular conductances in parallel with each other, and both in series with leaf boundary layer
    conductance (Thornton, 1998; Running & Kimball, 2005). In the case of plant transpiration,
    surface conductance is equivalent to the canopy conductance (Cc), and hence surface resistance
    (rs) is the inverse of canopy conductance (Cc).
    :param LAI: leaf-area index
    :param fwet: relative surface wetness
    :param gl_sh:
    :param gs1:
    :param Gcu:
    :return: canopy conductance
    """
    # noinspection PyTypeChecker
    Cc = where(
        logical_and(LAI > 0.0, (1.0 - fwet) > 0.0),
        gl_sh * (gs1 + Gcu) / (gs1 + gl_sh + Gcu) * LAI * (1.0 - fwet),
        0.0
    )

    Cc = clip(Cc, 1.0 / MAXIMUM_RESISTANCE, None)

    # water-mask canopy conductance
    Cc = where(water, nan, Cc)

    return Cc


def transpiration(s, Ac, rho, Cp, VPD, fc, ra, fwet, rs, water):
    # including the fc factor here produces dry results
    # AJ recommends not taking fc into account twice
    # numerator = (s * Ac + (rho * Cp * fc * VPD / ra)) * (1 - fwet)
    numerator = (s * Ac + (rho * Cp * fc * VPD / ra)) * (1.0 - fwet)
    denominator = s + GAMMA * (1.0 + (rs / ra))
    LEc = numerator / denominator

    # fill transpiration with zero
    LEc = where(isnan(LEc), 0.0, LEc)

    # mask transpiration for water bodies
    LEc = float32(where(water, nan, LEc))

    return LEc


def canopy_aerodynamic_resistance(VPD, vpd_open, vpd_close, rbl_max, rbl_min):
    """
    calculates total aerodynamic resistance to the canopy
    from vapor pressure deficit and biome-specific constraints.
    :param VPD: vapor pressure deficit
    :param vpd_open: vapor pressure deficit when stomata are open
    :param vpd_close: vapor pressure deficit when stomata are closed
    :param rbl_max:
    :param rbl_min:
    :return: aerodynamic resistance to the canopy
    """
    rtotc = where(VPD <= vpd_open, rbl_max, nan)
    rtotc = where(VPD >= vpd_close, rbl_min, rtotc)
    rtotc = where(
        logical_and(vpd_open < VPD, VPD < vpd_close),
        rbl_min + (rbl_max - rbl_min) * (vpd_close - VPD) / (vpd_close - vpd_open),
        rtotc
    )

    return rtotc


def soil_heat_flux(NDVI, Rn, use_ptjpl_g_in_mod16=False):
    """
    This function calculates soil heat flux for the JPL adaptation of the Mu et al. PM-MOD model of evapotranspiration.
    :param NDVI: normalized difference vegetation index
    :param Rn: net radiation in watts per square meter
    :param use_ptjpl_g_in_mod16: flag to use PT-JPL formula for soil heat flux
    :return:
    """

    if use_ptjpl_g_in_mod16:
        # the MATLAB formula for G in MOD16 isn't documented
        # the formula for G in Mu et al. 2011 takes average daytime and nighttime temperatures into account
        # these temperatures aren't available in the current setup
        # porting PT-JPL method for G for time being
        G = Rn * (0.05 + (1.0 - clip(NDVI - 0.05, 0.0, 1.0)) * 0.265)
    else:
        # FIXME this does not conform to the MOD16 algorithm, I don't know where Kaniska got this from
        G = float32((-0.276 * NDVI + 0.35) * Rn)

    return G


def wet_soil_evaporation(s, Asoil, rho, Cp, fc, VPD, ras, fwet, rtot):
    """
    MOD16 method for calculating wet soil evaporation
    :param s: slope of the saturation to vapor pressure curve
    :param Asoil: available radiation to the soil (soil partition of net radiation)
    :param rho:
    :param Cp: specific humidity
    :param fc: vegetation fraction
    :param VPD: vapor pressure deficit
    :param ras:
    :param fwet: relative surface wetness
    :param rtot:
    :return: wet soil evaporation in watts per square meter
    """
    numerator = (s * Asoil + rho * Cp * (1.0 - fc) * VPD / ras) * fwet
    denominator = s + GAMMA * rtot / ras
    LE_soil_wet = numerator / denominator

    return LE_soil_wet


def potential_soil_evaporation(s, Asoil, rho, Cp, fc, VPD, ras, fwet, rtot):
    numerator = (s * Asoil + rho * Cp * (1.0 - fc) * VPD / ras) * (1.0 - fwet)
    denominator = s + GAMMA * rtot / ras
    LE_soil_pot = numerator / denominator

    LE_soil_pot = clip(LE_soil_pot, 0.0, None)

    return LE_soil_pot


def soil_moisture_constraint(RH, VPD, use_ptjpl_sm_in_mod16=False):
    """
    This function calculates the soil moisture constraint for the JPL adaptation
    of the Penman-Monteith MOD16 algorithm.
    :param RH: relative humidity between zero and one
    :param VPD: vapor pressure deficit in pascals
    :return:
    """

    # constrain relative humidity between 0% and 100%
    RH = clip(RH, 0.0, 1.0)

    # there's an option here to use the PT-JPL parameterization instead, which performs better at ECOSTRESS scale
    if use_ptjpl_sm_in_mod16:
        # porting kPa BETA from PT-JPL for better results at ECOSTRESS resolution
        BETA_kPa = 1.0

        # convert vapor pressure deficit from pascals to kilopascals
        VPD_kPa = clip(VPD / 1000.0, 0.0, None)

        # soil moisture constraint should be identical to PT-JPL
        fSM = RH ** (VPD_kPa / BETA_kPa)
    else:
        # beta was 100 in the old algorithm and 200 in the new algorithm according to Mu et al. 2011
        BETA = 200.0

        fSM = RH ** (VPD / BETA)

    return fSM


def calculate_vapor(LE_daily, daylight_hours):
    # convert length of day in hours to seconds
    daylight_seconds = daylight_hours * 3600.0

    # constant latent heat of vaporization for water: the number of joules of energy it takes to evaporate one kilogram
    LATENT_VAPORIZATION_JOULES_PER_KILOGRAM = 2450000.0

    # factor seconds out of watts to get joules and divide by latent heat of vaporization to get kilograms
    ET_daily_kg = float32(clip(LE_daily * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0.0, None))

    return ET_daily_kg


# class MOD16Parameters:
#     def __init__(
#             self,
#             IGBP: Raster):
#         self.IGBP = IGBP


class MOD16(FLiES):
    def __init__(
            self,
            working_directory: str = None,
            static_directory: str = None,
            SRTM_connection: SRTM = None,
            SRTM_download: str = None,
            GEOS5FP_connection: GEOS5FP = None,
            GEOS5FP_download: str = None,
            GEOS5FP_products: str = None,
            # MCD12_connnection: MCD12C1 = None,
            # MCD12_download: str = None,
            intermediate_directory=None,
            preview_quality: int = DEFAULT_PREVIEW_QUALITY,
            ANN_model: Callable = None,
            ANN_model_filename: str = None,
            resampling: str = DEFAULT_RESAMPLING,
            downscale_air: bool = True,
            downscale_vapor: bool = True,
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
            intermediate_directory = join(working_directory, DEFAULT_MOD16_INTERMEDIATE)

        super(MOD16, self).__init__(
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

        # if MCD12_connnection is None:
        #     MCD12_connnection = MCD12C1(
        #         working_directory=static_directory,
        #         download_directory=MCD12_download
        #     )

        # self.MCD12 = MCD12_connnection
        self.downscale_air = downscale_air
        self.downscale_vapor = downscale_vapor

    def IGBP(self, geometry: RasterGeometry = None, resampling: str = None, **kwargs) -> Raster:
        # return self.MCD12.IGBP(geometry=geometry, resampling=resampling, **kwargs)
        return load_MCD12C1_IGBP(geometry=geometry)

    def IGBP_subset(self, geometry: RasterGeometry, resampling: str = None, buffer=3, **kwargs):
        return self.IGBP(geometry=geometry.bbox, resampling=resampling, buffer=buffer, **kwargs)

    def gl_sh(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        query leaf conductance to sensible heat (gl_sh)
        in seconds per meter
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['gl_sh'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                               resampling=resampling)

        return image

    def gl_e_wv(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        leaf conductance to evaporated water vapor (gl_e_wv)
        """

        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['gl_e_wv'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                 resampling=resampling)

        return image

    def CL(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        biome-specific mean potential stomatal conductance per unit leaf area
        """

        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['cl'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                            resampling=resampling)

        return image

    def tmin_open(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        open minimum temperature by land-cover
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['tmin_open'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                   resampling=resampling)

        return image

    def tmin_close(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        closed minimum temperature by land-cover
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['tmin_close'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                    resampling=resampling)

        return image

    def vpd_open(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        open vapor pressure deficit by land-cover
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['vpd_open'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                  resampling=resampling)

        return image

    def vpd_close(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        closed vapor pressure deficit by land-cover
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['vpd_close'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                   resampling=resampling)

        return image

    def rbl_max(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        maximum aerodynamic resistance
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['rbl_max'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                 resampling=resampling)

        return image

    def rbl_min(self, geometry: RasterGeometry, IGBP: Raster = None, resampling: str = None) -> Raster:
        """
        minimum aerodynamic resistance
        """
        if resampling is None:
            resampling = self.resampling

        if IGBP is None:
            IGBP = self.IGBP_subset(geometry)

        image = Raster(float32(array(LUT['rbl_min'])[IGBP]), geometry=IGBP.geometry).to_geometry(geometry,
                                                                                                 resampling=resampling)

        return image

    # TODO check units of minimum temperature
    def MOD16(
            self,
            geometry: RasterGrid,
            target: str,
            time_UTC: datetime or str,
            ST_K: Raster,
            emissivity: Raster,
            NDVI: Raster,
            albedo: Raster,
            LAI: Raster = None,
            FVC: Raster = None,
            IGBP: Raster = None,
            Ta_K: Raster = None,
            Tmin_K: Raster = None,
            Ea_Pa: Raster = None,
            elevation_km: Raster = None,
            Ps_Pa: Raster = None,
            SWin: Raster = None,
            Rn: Raster = None,
            Rn_daily: Raster = None,
            G: Raster = None,
            water=None,
            cloud_mask=None,
            output_variables=DEFAULT_OUTPUT_VARIABLES,
            results=None,
            diagnostic=False):
        """
        calculate MOD16 evapotranspiration.
        :param Rn: net radiation in watts per square meter
        :param Rn_daily: daily net radiation in watts per square meter
        :param IGBP: land cover type using MODIS landcover codes
            http://gIGBPf.umd.edu/data/IGBP/
            0: Water
            1: Evergreen Needleleaf forest
            2: Evergreen Broadleaf forest
            3: Deciduous Needleleaf forest
            4: Deciduous Broadleaf forest
            5: Mixed forest
            6: Closed shrublands
            7: Open shrublands
            8: Woody savannas
            9: Savannas
            10: Grasslands
            11: Permanent wetlands
            12: Croplands
            13: Urban and built-up
            14: Cropland/Natural vegetation mosaic
            15: Snow and ice
            16: Barren or sparsely vegetated
        :param FVC: vegetation cover fraction (fPAR can be used as substitute)
        :param LAI: leaf-area index
        :param NDVI: normalized difference vegetation index
        :param Ta_K: air temperature in kelvin
        :param Ea_Pa: actual water vapor pressure in pascals
        :param Ps_Pa: pressure in pascals
        :param Tmin_K: minimum temperature
        :param results:
        :return:
        """
        STEFAN_BOLTZMAN_CONSTANT = 5.67036713e-8  # SI units watts per square meter per kelvin to the fourth

        if results is None:
            results = {}

        warnings.filterwarnings('ignore')

        # calculate time
        hour_of_day = self.hour_of_day(time_UTC=time_UTC, geometry=geometry)
        day_of_year = self.day_of_year(time_UTC=time_UTC, geometry=geometry)
        SHA = self.SHA_deg_from_doy_lat(day_of_year, geometry.lat)
        sunrise_hour = self.sunrise_from_sha(SHA)
        daylight_hours = self.daylight_from_sha(SHA)
        date_UTC = time_UTC.date()

        NDVI = rt.clip(NDVI, -1, 1)

        if water is None:
            water = NDVI < 0

        if LAI is None:
            LAI = self.NDVI_to_LAI(NDVI)

        if FVC is None:
            FVC = self.NDVI_to_FVC(NDVI)

        if IGBP is None:
            IGBP = self.IGBP(geometry)

        self.diagnostic(IGBP, "IGBP", date_UTC, target)

        if Ta_K is None:
            Ta_K = self.Ta_K(time_UTC=time_UTC, geometry=geometry, ST_K=ST_K)

        Ta_K = Ta_K.astype(np.float32)
        Ta_K = rt.where(water, nan, Ta_K)

        self.diagnostic(Ta_K, "Ta_K", date_UTC, target)

        if 'Ta_K' in output_variables or diagnostic:
            results['Ta_K'] = Ta_K

        if Tmin_K is None:
            Tmin_K = self.Tmin_K(time_UTC=time_UTC, geometry=geometry, ST_K=ST_K)

        self.diagnostic(Tmin_K, "Tmin_K", date_UTC, target)

        if Ea_Pa is None:
            Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, ST_K=ST_K)

        Ea_Pa = Ea_Pa.astype(np.float32)
        Ea_Pa = rt.where(water, nan, Ea_Pa)

        self.diagnostic(Ea_Pa, "Ea_Pa", date_UTC, target)

        if 'Ea_Pa' in output_variables or diagnostic:
            results['Ea_Pa'] = Ea_Pa

        if Ps_Pa is None:
            if elevation_km is None:
                elevation_km = self.elevation_km(geometry)

            elevation_m = elevation_km * 1000
            Ps_Pa = 101325.0 * (1.0 - 0.0065 * elevation_m / Ta_K) ** (9.807 / (0.0065 * 287.0))  # [Pa]

        Ps_Pa = Ps_Pa.astype(np.float32)
        Ps_Pa = rt.where(water, nan, Ps_Pa)

        self.diagnostic(Ps_Pa, "Ps_Pa", date_UTC, target)

        if 'Ps_Pa' in output_variables or diagnostic:
            results['Ps_Pa'] = Ps_Pa

        albedo = rt.clip(albedo, 0, 1)
        self.diagnostic(albedo, "albedo", date_UTC, target)

        if Rn is None:
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

        if "Rn" in output_variables:
            results["Rn"] = Rn

        if "Rn_daily" in output_variables:
            results["Rn_daily"] = Rn_daily

        # remove out of bounds fAPAR
        # fc = float32(where(logical_or(fc < 0, fc > 1), nan, fc))
        FVC = float32(clip(FVC, 0.0, 1.0))
        FVC = where(water, nan, FVC)

        if 'fc' in output_variables or diagnostic:
            # print('saving fc')
            # print(nanmin(fc), nanmax(fc))
            results['fc'] = FVC

        # remove out of bounds LST
        # NDVI = float32(where(logical_or(NDVI < -1, NDVI > 1), nan, NDVI))
        NDVI = float32(clip(NDVI, 0.0, 1.0))

        # remove out of bounds LAI
        LAI = float32(clip(LAI, 0.0, 10.0))
        LAI = where(water, nan, LAI)

        if 'LAI' in output_variables or diagnostic:
            # print('saving LAI')
            # print(nanmin(LAI), nanmax(LAI))
            results['LAI'] = LAI

        logger.info("calculating PM-MOD meteorology")

        # specific humidity of air
        # as a ratio of kilograms of water to kilograms of air and water
        # from surface pressure and actual water vapor pressure
        specific_humidity = calculate_specific_humidity(Ea_Pa, Ps_Pa)

        if 'specific_humidity' in output_variables or diagnostic:
            results['specific_humidity'] = specific_humidity

        rho = air_density(Ps_Pa, Ta_K, specific_humidity)

        # calculate specific heat capacity of the air (Cp)
        # in joules per kilogram per kelvin
        # from specific heat of water vapor (CPW)
        # and specific heat of dry air (CPD)
        Cp = specific_humidity * CPW + (1 - specific_humidity) * CPD

        # calculate saturation vapor pressure from air temperature
        Ta_C = kelvin_to_celsius(Ta_K)
        SVP_kPa = SVP_from_Ta(Ta_C)
        SVP_Pa = float32(SVP_kPa * 1000.0)

        if 'SVP' in output_variables or diagnostic:
            results['SVP'] = SVP_Pa

        # relative humidity from the ratio of actual water vapor pressure to saturation vapor pressure
        RH = Ea_Pa / SVP_Pa

        # remove out of bounds RH
        RH = float32(clip(RH, 0.0, 1.0))
        RH = where(water, nan, RH)

        if 'RH' in output_variables or diagnostic:
            # print('saving RH')
            # print(nanmin(RH), nanmax(RH))
            results['RH'] = RH

        # slope of saturation vapor pressure curve in pascals per degree
        s = delta_from_Ta(Ta_K) * 1000.0

        # print("s: {}".format(np.nanmean(s)))

        # vapor pressure deficit in pascals
        VPD = SVP_Pa - Ea_Pa

        # remove negative VPD values
        # replacing negative VPD with zero instead of nan to avoid holes in the map
        VPD = float32(where(VPD < 0.0, 0.0, VPD))
        VPD = where(water, nan, VPD)

        if 'VPD' in output_variables or diagnostic:
            # print('saving VPD')
            # print(nanmin(VPD), nanmax(VPD))
            results['VPD'] = VPD

        # calculate relative surface wetness (fwet)
        # from relative humidity
        fwet = calculate_fwet(RH, water)

        if 'fwet' in output_variables or diagnostic:
            # print('saving fwet')
            # print(nanmin(fwet), nanmax(fwet))
            results['fwet'] = fwet

        logger.info("calculating PM-MOD resistances")

        IGBP_subset = self.IGBP_subset(geometry=geometry)

        # query leaf conductance to sensible heat (gl_sh)
        # in seconds per meter
        # gl_sh = float32(array(LUT['gl_sh'])[IGBP])
        gl_sh = self.gl_sh(geometry=geometry, IGBP=IGBP_subset)

        if 'gl_sh' in output_variables or diagnostic:
            # print('saving gl_sh')
            # print(nanmin(gl_sh), nanmax(gl_sh))
            results['gl_sh'] = gl_sh

        # calculate wet canopy resistance to sensible heat (rhc)
        # in seconds per meter
        # from leaf conductance to sensible heat (gl_sh), LAI, and relative surface wetness (fwet)
        rhc = wet_canopy_resistance(gl_sh, LAI, fwet)

        if 'rhc' in output_variables or diagnostic:
            # print('saving rhc')
            # print(nanmin(rhc), nanmax(rhc))
            results['rhc'] = rhc

        # calculate resistance to radiative heat transfer through air (rrc)
        rrc = float32(rho * Cp / (4.0 * SIGMA * Ta_K ** 3.0))

        if 'rrc' in output_variables or diagnostic:
            # print('saving rrc')
            # print(nanmin(rrc), nanmax(rrc))
            results['rrc'] = rrc

        # calculate aerodynamic resistance (rhrc)
        # in seconds per meter
        # from wet canopy resistance to sensible heat
        # and resistance to radiative heat transfer through air
        rhrc = float32((rhc * rrc) / (rhc + rrc))

        if 'rhrc' in output_variables or diagnostic:
            # print('saving rhrc')
            # print(nanmin(rhrc), nanmax(rhrc))
            results['rhrc'] = rhrc

        # calculate leaf conductance to evaporated water vapor (gl_e_wv)
        # gl_e_wv = float32(array(LUT['gl_e_wv'])[IGBP])
        gl_e_wv = self.gl_e_wv(geometry=geometry, IGBP=IGBP_subset)

        if 'gl_e_wv' in output_variables or diagnostic:
            # print('saving gl_e_wv')
            # print(nanmin(gl_e_wv), nanmax(gl_e_wv))
            results['gl_e_wv'] = gl_e_wv

        rvc = wet_canopy_resistance(gl_e_wv, LAI, fwet)

        if 'rvc' in output_variables or diagnostic:
            # print('saving rvc')
            # print(nanmin(rvc), nanmax(rvc))
            results['rvc'] = rvc

        # caluclate available radiation to the canopy (Ac)
        # in watts per square meter
        # this is the same as net radiation to the canopy in PT-JPL
        Ac = float32(Rn * FVC)

        if 'Ac' in output_variables or diagnostic:
            # print('saving Ac')
            # print(nanmin(Ac), nanmax(Ac))
            results['Ac'] = Ac

        # calculate wet latent heat flux (LEi)
        # in watts per square meter
        LEi = interception(s, Ac, rho, Cp, VPD, FVC, rhrc, fwet, rvc, water)

        if 'LEi' in output_variables or diagnostic:
            # print('saving LEi')
            # print(nanmin(LEi), nanmax(LEi))
            results['LEi'] = LEi

        # calculate correctance factor (rcorr)
        # for stomatal and cuticular conductances
        # from surface pressure and air temperature
        rcorr = correctance_factor(Ps_Pa, Ta_K)

        if 'rcorr' in output_variables or diagnostic:
            # print('saving rcorr')
            # print(nanmin(rcorr), nanmax(rcorr))
            results['rcorr'] = rcorr

        # query biome-specific mean potential stomatal conductance per unit leaf area
        # CL = array(LUT['cl'])[IGBP]
        CL = self.CL(geometry=geometry, IGBP=IGBP_subset)

        CL = where(water, nan, CL)

        if 'CL' in output_variables or diagnostic:
            # print('saving CL')
            # print(nanmin(CL), nanmax(CL))
            results['CL'] = CL

        # query open minimum temperature by land-cover
        # tmin_open = float32(array(LUT['tmin_open'])[IGBP])
        tmin_open = self.tmin_open(geometry=geometry, IGBP=IGBP_subset)

        if 'tmin_open' in output_variables or diagnostic:
            results['tmin_open'] = tmin_open

        # query closed minimum temperature by land-cover
        # tmin_close = float32(array(LUT['tmin_close'])[IGBP])
        tmin_close = self.tmin_close(geometry=geometry, IGBP=IGBP_subset)

        if 'tmin_close' in output_variables or diagnostic:
            results['tmin_close'] = tmin_close

        Tmin_C = Tmin_K - 273.15

        if 'Tmin_C' in output_variables or diagnostic:
            results['Tmin_C'] = Tmin_C

        # calculate minimum temperature factor for stomatal conductance
        mTmin = tmin_factor(Tmin_C, tmin_open, tmin_close, IGBP, water)

        if 'mTmin' in output_variables or diagnostic:
            results['mTmin'] = mTmin

        # query open vapor pressure deficit by land-cover
        # vpd_open = float32(array(LUT['vpd_open'])[IGBP])
        vpd_open = self.vpd_open(geometry=geometry, IGBP=IGBP_subset)

        vpd_open = where(water, nan, vpd_open)

        if 'vpd_open' in output_variables or diagnostic:
            results['vpd_open'] = vpd_open

        # query closed vapor pressure deficit by land-cover
        # vpd_close = float32(array(LUT['vpd_close'])[IGBP])
        vpd_close = self.vpd_close(geometry=geometry, IGBP=IGBP_subset)

        vpd_close = where(water, nan, vpd_close)

        if 'vpd_close' in output_variables or diagnostic:
            results['vpd_close'] = vpd_close

        # calculate vapor pressure deficit factor for stomatal conductance
        mVPD = vpd_factor(vpd_open, vpd_close, VPD)

        if 'mVPD' in output_variables or diagnostic:
            results['mVPD'] = mVPD

        # calculate stomatal conductance (gs1)
        gs1 = CL * mTmin * mVPD * rcorr

        if 'gs1' in output_variables or diagnostic:
            results['gs1'] = gs1

        # correct cuticular conductance constant to leaf cuticular conductance (Gcu) using correction factor (rcorr)
        Gcu = CUTICULAR_CONDUCTANCE * rcorr

        if 'Gcu' in output_variables or diagnostic:
            results['Gcu'] = Gcu

        # calculate canopy conductance
        Cc = canopy_conductance(LAI, fwet, gl_sh, gs1, Gcu, water)

        if 'Cc' in output_variables or diagnostic:
            results['Cc'] = Cc

        # calculate surface resistance to evapotranspiration (rs)
        # as inverse of canopy conductance (Cc)
        rs = 1.0 / Cc

        rs = clip(rs, 0.0, MAXIMUM_RESISTANCE)

        # water-mask surface resistance to evapotranspiration
        rs = where(water, nan, rs)

        if 'rs' in output_variables or diagnostic:
            results['rs'] = rs

        # calculate convective heat transfer (rh)
        # as inverse of leaf conductance to sensible heat (gl_sh)
        rh = 1.0 / gl_sh

        if 'rh' in output_variables or diagnostic:
            results['rh'] = rs

        # calculate radiative heat transfer (rr)
        rr = rho * Cp / (4.0 * SIGMA * Ta_K ** 3)

        if 'rr' in output_variables or diagnostic:
            results['rr'] = rs

        # calculate parallel resistance (ra)
        # MOD16 user guide is not clear about what to call this
        ra = (rh * rr) / (rh + rr)

        logger.info("calculating PM-MOD fluxes")

        if 'ra' in output_variables or diagnostic:
            results['ra'] = ra

        if 'Cp' in output_variables or diagnostic:
            results['Cp'] = Cp

        if 'rho' in output_variables or diagnostic:
            results['rho'] = rho

        if 's' in output_variables or diagnostic:
            results['s'] = s

        # calculate transpiration
        LEc = transpiration(s, Ac, rho, Cp, VPD, FVC, ra, fwet, rs, water)

        # store transpiration in results
        if 'LEc' in output_variables or diagnostic:
            # print('saving LEc')
            # print(nanmin(LEc), nanmax(LEc))
            results['LEc'] = LEc

        # soil evaporation

        # query aerodynamic resistant constraints from land-cover
        # rbl_max = float32(array(LUT['rbl_max'])[IGBP])
        rbl_max = self.rbl_max(geometry=geometry, IGBP=IGBP_subset)
        # rbl_min = float32(array(LUT['rbl_min'])[IGBP])
        rbl_min = self.rbl_min(geometry=geometry, IGBP=IGBP_subset)

        rbl_max = where(water, nan, rbl_max)
        rbl_min = where(water, nan, rbl_min)

        if 'rbl_max' in output_variables or diagnostic:
            results['rbl_max'] = rbl_max

        if 'rbl_min' in output_variables or diagnostic:
            results['rbl_min'] = rbl_min

        rtotc = canopy_aerodynamic_resistance(VPD, vpd_open, vpd_close, rbl_max, rbl_min)

        if 'rtotc' in output_variables or diagnostic:
            # print('saving rtotc')
            # print(nanmin(rtotc), nanmax(rtotc))
            results['rtotc'] = rtotc

        # calculate total aerodynamic resistance
        # by applying correction to total canopy resistance
        rtot = rcorr * rtotc

        if 'rtot' in output_variables or diagnostic:
            # print('saving rtot')
            # print(nanmin(rtot), nanmax(rtot))
            results['rtot'] = rtot

        # calculate resistance to radiative heat transfer
        rrs = float32(rho * Cp / (4.0 * SIGMA * Ta_K ** 3))
        rrs = where(water, nan, rrs)

        if 'rrs' in output_variables or diagnostic:
            # print('saving rrs')
            # print(nanmin(rrs), nanmax(rrs))
            results['rrs'] = rrs

        # calculate aerodynamic resistance at the soil surface
        ras = float32((rtot * rrs) / (rtot + rrs))

        if 'ras' in output_variables or diagnostic:
            # print('saving ras')
            # print(nanmin(ras), nanmax(ras))
            results['ras'] = ras

        # calculate soil heat flux
        if G is None:
            G = soil_heat_flux(NDVI, Rn)

        if 'G' in output_variables or diagnostic:
            # print('saving G')
            # print(nanmin(G), nanmax(G))
            results['G'] = G

        # calculate available radiation at the soil
        Asoil = (1.0 - FVC) * Rn - G
        Asoil = clip(Asoil, 0.0, None)

        if 'Asoil' in output_variables or diagnostic:
            # print('saving Asoil')
            # print(nanmin(Asoil), nanmax(Asoil))
            results['Asoil'] = Asoil

        # separate wet soil evaporation and potential soil evaporation

        # calculate wet soil evaporation
        LE_soil_wet = wet_soil_evaporation(s, Asoil, rho, Cp, FVC, VPD, ras, fwet, rtot)

        if 'LE_soil_wet' in output_variables or diagnostic:
            # print('saving LE_soil_wet')
            # print(nanmin(LE_soil_wet), nanmax(LE_soil_wet))
            results['LE_soil_wet'] = LE_soil_wet

        LE_soil_pot = potential_soil_evaporation(s, Asoil, rho, Cp, FVC, VPD, ras, fwet, rtot)

        if 'LE_soil_pot' in output_variables or diagnostic:
            # print('saving LE_soil_pot')
            # print(nanmin(LE_soil_pot), nanmax(LE_soil_pot))
            results['LE_soil_pot'] = LE_soil_pot

        # calculate soil moisture constraint
        fSM = soil_moisture_constraint(RH, VPD)

        if 'fSM' in output_variables or diagnostic:
            # print('saving fSM')
            # print(nanmin(fSM), nanmax(fSM))
            results['fSM'] = fSM

        # calculate soil evaporation
        LEs = LE_soil_wet + LE_soil_pot * fSM

        # zero out negative values of soil evaporation
        LEs = clip(LEs, 0.0, None)

        # fill soil evaporation with zero
        LEs = where(isnan(LEs), 0.0, LEs)

        # mask soil evaporation for water bodies
        LEs = where(water, nan, LEs)

        # store soil evaporation in results
        if 'LEs' in output_variables or diagnostic:
            # print('saving LEs')
            # print(nanmin(LEs), nanmax(LEs))
            results['LEs'] = LEs

        # sum partitions into total latent heat flux
        LE = LEi + LEc + LEs

        # constrain latent heat flux to net radiation
        LE = clip(LE, 0.0, Rn)

        # remove negative values of latent heat flux
        # LE = where(LE <= 0, nan, LE)

        # store latent heat flux in results
        if 'LE' in output_variables or diagnostic:
            # print('saving LE')
            # print(nanmin(LE), nanmax(LE))
            results['LE'] = LE

        # save sensible heat flux
        if 'H' in output_variables or diagnostic:
            H = Rn - G - LE
            results['H'] = H

        # calculate evaporative fraction
        EF = LE / (Rn - G)
        EF = where((Rn - G) < 0.0001, 1.0, EF)
        EF = float32(clip(EF, 0.0, 1.0))
        # EF = clip(where(Rn - G == 0, 1, LE / (Rn - G)), 0, 1)

        # remove evaporative fractions greater than 100%
        # EF = where(EF > 1, nan, EF)

        if 'EF' in output_variables or diagnostic:
            # print('saving EF')
            # print(nanmin(EF), nanmax(EF))
            results['EF'] = EF

        # calculate daily latent heat flux from evaporative fraction and daily net radiation with minimum of zero
        LE_daily = EF * Rn_daily
        LE_daily = clip(LE_daily, 0.0, None)

        # store daily latent heat flux in results
        if 'LE_daily' in output_variables or diagnostic:
            results['LE_daily'] = LE_daily

        # calculate daytime daily evapotranspiration in kilograms equivalent to millimeters
        ET_daily_kg = calculate_vapor(LE_daily, daylight_hours)

        if 'ET_daily_kg' in output_variables or diagnostic:
            results['ET_daily_kg'] = float32(ET_daily_kg)

        warnings.resetwarnings()

        return results
