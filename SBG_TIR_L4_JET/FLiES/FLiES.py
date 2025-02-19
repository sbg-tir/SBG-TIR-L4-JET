"""
Forest Light Environmental Simulator (FLiES)
Artificial Neural Network Implementation
for the Breathing Earth Systems Simulator (BESS)
"""

import logging
import warnings
from datetime import datetime, date, timedelta
from os.path import join, abspath, dirname, exists, expanduser
from time import process_time
from typing import Callable, Union

import numpy as np
import pandas as pd
from dateutil import parser
from scipy.stats import zscore

import colored_logging as cl
from .daylight_hours import day_angle_rad_from_doy, solar_dec_deg_from_day_angle_rad
from .solar_zenith_angle import sza_deg_from_lat_dec_hour
from ..model.model import Model

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # from keras.engine.saving import load_model
    from keras.models import load_model
    from keras.saving import register_keras_serializable

import rasters as rt
from rasters import Raster, RasterGeometry

from geos5fp import GEOS5FP
from koppengeiger import load_koppen_geiger

from ..SRTM import SRTM


__author__ = "Gregory Halverson, Robert Freepartner"

MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_FLIES_INTERMEDIATE = "FLiES_intermediate"
DEFAULT_MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_INCLUDE_PREVIEW = True
DEFAULT_RESAMPLING = "cubic"
DEFAULT_SAVE_INTERMEDIATE = True
DEFAULT_SHOW_DISTRIBUTION = True
DEFAULT_DYNAMIC_ATYPE_CTYPE = False


class SentinelNotAvailable(IOError):
    pass


class SRTMNotAvailableError(IOError):
    pass


class GEOS5FPNotAvailableError(IOError):
    pass


class BlankOutputError(Exception):
    pass

@register_keras_serializable()
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

class FLiES(Model):
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
            intermediate_directory: str = None,
            preview_quality: int = DEFAULT_PREVIEW_QUALITY,
            ANN_model: Callable = None,
            ANN_model_filename: str = MODEL_FILENAME,
            resampling: str = DEFAULT_RESAMPLING,
            save_intermediate: bool = DEFAULT_SAVE_INTERMEDIATE,
            show_distribution: bool = DEFAULT_SHOW_DISTRIBUTION,
            include_preview: bool = DEFAULT_INCLUDE_PREVIEW,
            dynamic_atype_ctype: bool = DEFAULT_DYNAMIC_ATYPE_CTYPE):

        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        if static_directory is None:
            static_directory = working_directory

        static_directory = abspath(expanduser(static_directory))

        self.logger.info(f"FLiES working directory: {cl.dir(working_directory)}")

        if SRTM_connection is None:
            try:
                self.logger.info("connecting to SRTM")
                SRTM_connection = SRTM(
                    working_directory=static_directory,
                    download_directory=SRTM_download,
                    offline_ok=True
                )
            except Exception as e:
                self.logger.exception(e)
                raise SRTMNotAvailableError()

        self.SRTM_connection = SRTM_connection

        if GEOS5FP_connection is None:
            try:
                self.logger.info(f"connecting to GEOS-5 FP")
                GEOS5FP_connection = GEOS5FP(
                    working_directory=working_directory,
                    download_directory=GEOS5FP_download,
                    products_directory=GEOS5FP_products
                )
            except Exception as e:
                self.logger.exception(e)
                raise GEOS5FPNotAvailableError()

        self.GEOS5FP_connection = GEOS5FP_connection

        if intermediate_directory is None:
            intermediate_directory = join(working_directory, DEFAULT_FLIES_INTERMEDIATE)

        intermediate_directory = abspath(expanduser(intermediate_directory))

        self.logger.info(f"FLiES intermediate directory: {cl.dir(intermediate_directory)}")

        if ANN_model_filename is None:
            ANN_model_filename = DEFAULT_MODEL_FILENAME

        if ANN_model is None:
            ANN_model = load_model(ANN_model_filename, custom_objects={'mae': mae})

        super(FLiES, self).__init__(
            working_directory=working_directory,
            static_directory=static_directory,
            intermediate_directory=intermediate_directory,
            preview_quality=preview_quality,
            resampling=resampling,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution,
            include_preview=include_preview
        )

        self.ANN_model = ANN_model
        self.dynamic_atype_ctype = dynamic_atype_ctype

    def UTC_to_solar(self, time_UTC: datetime, lon: float) -> datetime:
        return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))

    def UTC_offset_hours(self, geometry: RasterGeometry) -> Raster:
        return Raster(np.radians(geometry.lon) / np.pi * 12, geometry=geometry)

    def day_of_year(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        doy_UTC = time_UTC.timetuple().tm_yday
        hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
        UTC_offset_hours = self.UTC_offset_hours(geometry=geometry)
        hour_of_day = hour_UTC + UTC_offset_hours
        doy = doy_UTC
        doy = rt.where(hour_of_day < 0, doy - 1, doy)
        doy = rt.where(hour_of_day > 24, doy + 1, doy)

        return doy

    def hour_of_day(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
        UTC_offset_hours = self.UTC_offset_hours(geometry=geometry)
        hour_of_day = hour_UTC + UTC_offset_hours
        hour_of_day = rt.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
        hour_of_day = rt.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

        return hour_of_day

    def SZA(self, day_of_year: Raster, hour_of_day: Raster, geometry: RasterGeometry) -> Raster:
        """
        Calculates solar zenith angle at latitude and solar apparent time.
        :param lat: latitude in degrees
        :param dec: solar declination in degrees
        :param hour: hour of day
        :return: solar zenith angle in degrees
        """
        # day_angle = np.radians((2 * np.pi * (day_of_year - 1)) / 365)
        # day_angle_rad = (2 * np.pi * (day_of_year - 1)) / 365
        # lat = np.radians(geometry.lat)
        # solar_dec_rad = np.radians((0.006918 - 0.399912 * np.cos(day_angle_rad) + 0.070257 * np.sin(day_angle_rad) - 0.006758 * np.cos(
        #     2 * day_angle_rad) + 0.000907 * np.sin(2 * day_angle_rad) - 0.002697 * np.cos(3 * day_angle_rad) + 0.00148 * np.sin(
        #     3 * day_angle_rad)) * (180 / np.pi))
        # hour_angle_rad = np.radians(hour_of_day * 15.0 - 180.0)
        # SZA = np.degrees(np.arccos(np.sin(lat) * np.sin(solar_dec_rad) + np.cos(lat) * np.cos(solar_dec_rad) * np.cos(hour_angle_rad)))

        latitude = geometry.lat
        # print("lat: {}".format(np.nanmean(latitude)))
        day_angle_rad = day_angle_rad_from_doy(day_of_year)
        # print("day angle: {}".format(np.nanmean(day_angle_rad)))
        solar_dec_deg = solar_dec_deg_from_day_angle_rad(day_angle_rad)
        # print("solar declination: {}".format(np.nanmean(solar_dec_deg)))
        SZA_deg = sza_deg_from_lat_dec_hour(latitude, solar_dec_deg, hour_of_day)
        # print("SZA: {}".format(np.nanmean(SZA_deg)))

        SZA = Raster(SZA_deg, geometry=geometry)

        return SZA

    def generate_atype_ctype(self, COT: Raster, KG_climate: Raster, geometry: RasterGeometry, dynamic_atype_ctype: bool = True) -> (Raster, Raster):
        atype = np.full(geometry.shape, 1, dtype=np.uint16)
        ctype = np.full(geometry.shape, 0, dtype=np.uint16)

        if dynamic_atype_ctype:
            atype = np.where((COT == 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, atype)
            ctype = np.where((COT == 0) & ((KG_climate == 5) | (KG_climate == 6)), 0, ctype)

            atype = np.where((COT == 0) & ((KG_climate == 3) | (KG_climate == 4)), 2, atype)
            ctype = np.where((COT == 0) & ((KG_climate == 3) | (KG_climate == 4)), 0, ctype)

            atype = np.where((COT == 0) & (KG_climate == 1), 4, atype)
            ctype = np.where((COT == 0) & (KG_climate == 1), 0, ctype)

            atype = np.where((COT == 0) & (KG_climate == 2), 5, atype)
            ctype = np.where((COT == 0) & (KG_climate == 2), 0, ctype)

            atype = np.where((COT > 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, atype)
            ctype = np.where((COT > 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, ctype)

            atype = np.where((COT > 0) & ((KG_climate == 3) | (KG_climate == 4)), 2, atype)
            ctype = np.where((COT > 0) & ((KG_climate == 3) | (KG_climate == 4)), 1, ctype)

            atype = np.where((COT > 0) & (KG_climate == 2), 5, atype)
            ctype = np.where((COT > 0) & (KG_climate == 2), 1, ctype)

            atype = np.where((COT > 0) & (KG_climate == 1), 4, atype)
            ctype = np.where((COT > 0) & (KG_climate == 1), 3, ctype)

        atype = Raster(atype, geometry=geometry)
        ctype = Raster(ctype, geometry=geometry)

        return atype, ctype

    def FLiES_ANN(
            self,
            geometry: RasterGeometry,
            atype: Raster,
            ctype: Raster,
            COT: Raster,
            AOT: Raster,
            vapor_gccm: Raster,
            ozone_cm: Raster,
            albedo: Raster,
            elevation_km: Raster,
            SZA: Raster) -> (Raster, Raster, Raster, Raster, Raster, Raster, Raster):

        ctype_flat = np.array(ctype).flatten()
        atype_flat = np.array(atype).flatten()
        COT_flat = np.array(COT).flatten()
        AOT_flat = np.array(AOT).flatten()
        vapor_gccm_flat = np.array(vapor_gccm).flatten()
        ozone_cm_flat = np.array(ozone_cm).flatten()
        albedo_flat = np.array(albedo).flatten()
        elevation_km_flat = np.array(elevation_km).flatten()
        SZA_flat = np.array(SZA).flatten()

        inputs = pd.DataFrame({
            "ctype": ctype_flat,
            "atype": atype_flat,
            "COT": COT_flat,
            "AOT": AOT_flat,
            "vapor_gccm": vapor_gccm_flat,
            "ozone_cm": ozone_cm_flat,
            "albedo": albedo_flat,
            "elevation_km": elevation_km_flat,
            "SZA": SZA_flat
        })

        inputs["ctype0"] = np.float32(inputs.ctype == 0)
        inputs["ctype1"] = np.float32(inputs.ctype == 1)
        inputs["ctype3"] = np.float32(inputs.ctype == 3)
        inputs["atype1"] = np.float32(inputs.ctype == 1)
        inputs["atype2"] = np.float32(inputs.ctype == 2)
        inputs["atype4"] = np.float32(inputs.ctype == 4)
        inputs["atype5"] = np.float32(inputs.ctype == 5)

        inputs = inputs[
            ["ctype0", "ctype1", "ctype3", "atype1", "atype2", "atype4", "atype5", "COT", "AOT", "vapor_gccm",
             "ozone_cm", "albedo", "elevation_km", "SZA"]]
        outputs = self.ANN_model.predict(inputs)
        shape = COT.shape
        tm = Raster(np.clip(outputs[:, 0].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)
        puv = Raster(np.clip(outputs[:, 1].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)
        pvis = Raster(np.clip(outputs[:, 2].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)
        pnir = Raster(np.clip(outputs[:, 3].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)
        fduv = Raster(np.clip(outputs[:, 4].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)
        fdvis = Raster(np.clip(outputs[:, 5].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)
        fdnir = Raster(np.clip(outputs[:, 6].reshape(shape), 0, 1).astype(np.float32), geometry=geometry, nodata=np.nan)

        return tm, puv, pvis, pnir, fduv, fdvis, fdnir

    def AOT(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        self.logger.info("retrieving GEOS-5 FP aerosol optical thickness raster")
        return self.GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry)

    def COT(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        self.logger.info("generating GEOS-5 FP cloud optical thickness raster")
        return self.GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry)

    def vapor_gccm(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        self.logger.info("generating GEOS5-FP water vapor raster in grams per square centimeter")
        return self.GEOS5FP_connection.vapor_gccm(time_UTC=time_UTC, geometry=geometry)

    def ozone_cm(self, time_UTC: datetime, geometry: RasterGeometry) -> Raster:
        self.logger.info("generating GEOS5-FP ozone raster in grams per square centimeter")
        return self.GEOS5FP_connection.ozone_cm(time_UTC=time_UTC, geometry=geometry)

    def elevation_km(self, geometry: RasterGeometry) -> Raster:
        self.logger.info("retrieving SRTM elevation raster in kilometers")
        return self.SRTM_connection.elevation_km(geometry)

    def FLiES(
            self,
            geometry: RasterGeometry,
            target: str,
            time_UTC: datetime or str,
            albedo: Raster,
            COT: Raster = None,
            AOT: Raster = None,
            vapor_gccm: Raster = None,
            ozone_cm: Raster = None,
            elevation_km: Raster = None,
            SZA: Raster = None,
            KG_climate: Raster = None):
        self.logger.info(f"processing FLiES tile {cl.place(target)} {cl.val(geometry.shape)} at " + cl.time(
            f"{time_UTC:%Y-%m-%d} UTC"))

        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        date_UTC = time_UTC.date()
        hour_of_day = self.hour_of_day(time_UTC=time_UTC, geometry=geometry)
        day_of_year = self.day_of_year(time_UTC=time_UTC, geometry=geometry)

        if elevation_km is None:
            elevation_km = self.elevation_km(geometry)

        self.diagnostic(elevation_km, "elevation_km", date_UTC, target)

        if SZA is None:
            SZA = self.SZA(day_of_year=day_of_year, hour_of_day=hour_of_day, geometry=geometry)

        self.diagnostic(SZA, "SZA", date_UTC, target)

        if AOT is None:
            AOT = self.AOT(time_UTC=time_UTC, geometry=geometry)

        self.diagnostic(AOT, "AOT", date_UTC, target)

        if COT is None:
            COT = self.COT(time_UTC=time_UTC, geometry=geometry)

        COT = np.clip(COT, 0, None)
        COT = rt.where(COT < 0.001, 0, COT)
        self.diagnostic(COT, "COT", date_UTC, target)

        if vapor_gccm is None:
            vapor_gccm = self.vapor_gccm(time_UTC=time_UTC, geometry=geometry)

        self.diagnostic(vapor_gccm, "vapor_gccm", date_UTC, target)

        if ozone_cm is None:
            ozone_cm = self.ozone_cm(time_UTC=time_UTC, geometry=geometry)

        self.diagnostic(ozone_cm, "ozone_cm", date_UTC, target)

        if KG_climate is None:
            self.logger.info("generating Koppen Geiger top-level climate classification raster")
            KG_climate = load_koppen_geiger(geometry)

        self.diagnostic(KG_climate, "KG_climate", date_UTC, target)


        atype, ctype = self.generate_atype_ctype(COT=COT, KG_climate=KG_climate, geometry=geometry, dynamic_atype_ctype=self.dynamic_atype_ctype)
        self.diagnostic(atype, "atype", date_UTC, target)
        self.diagnostic(ctype, "ctype", date_UTC, target)

        self.logger.info(
            "started neural network processing " +
            f"at tile {cl.place(target)} {cl.val(geometry.shape)} " +
            "at " + cl.time(f"{time_UTC:%Y-%m-%d} UTC")
        )

        prediction_start_time = process_time()
        tm, puv, pvis, pnir, fduv, fdvis, fdnir = self.FLiES_ANN(
            geometry=geometry,
            atype=atype,
            ctype=ctype,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            albedo=albedo,
            elevation_km=elevation_km,
            SZA=SZA
        )

        prediction_end_time = process_time()
        prediction_duration = prediction_end_time - prediction_start_time
        self.logger.info(f"finished neural network processing ({prediction_duration:0.2f}s)")

        self.diagnostic(tm, "tm", date_UTC, target)
        self.diagnostic(puv, "puv", date_UTC, target)
        self.diagnostic(pvis, "pvis", date_UTC, target)
        self.diagnostic(pnir, "pnir", date_UTC, target)
        self.diagnostic(fduv, "fduv", date_UTC, target)
        self.diagnostic(fdvis, "fdvis", date_UTC, target)
        self.diagnostic(fdnir, "fdnir", date_UTC, target)

        ##  Correction for diffuse PAR
        COT = rt.where(COT == 0.0, np.nan, COT)
        COT = rt.where(np.isfinite(COT), COT, np.nan)
        x = np.log(COT)
        p1 = 0.05088
        p2 = 0.04909
        p3 = 0.5017
        corr = np.array(p1 * x * x + p2 * x + p3)
        corr[np.logical_or(np.isnan(corr), corr > 1.0)] = 1.0
        fdvis = fdvis * corr * 0.915

        ## Radiation components
        dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * day_of_year)
        Ra = 1333.6 * dr * np.cos(SZA * np.pi / 180.0)
        Ra = rt.where(SZA > 90.0, 0, Ra)
        Rg = Ra * tm
        UV = Rg * puv
        VIS = Rg * pvis
        NIR = Rg * pnir
        # UVdiff = SSR.UV * fduv
        VISdiff = VIS * fdvis
        NIRdiff = NIR * fdnir
        # UVdir = SSR.UV - UVdiff
        VISdir = VIS - VISdiff
        NIRdir = NIR - NIRdiff

        self.diagnostic(Ra, "Ra", date_UTC, target)
        self.diagnostic(Rg, "Rg", date_UTC, target)
        self.diagnostic(UV, "UV", date_UTC, target)
        self.diagnostic(VIS, "VIS", date_UTC, target)
        self.diagnostic(NIR, "NIR", date_UTC, target)
        self.diagnostic(VISdiff, "VISdiff", date_UTC, target)
        self.diagnostic(NIRdiff, "NIRdiff", date_UTC, target)
        self.diagnostic(VISdir, "VISdir", date_UTC, target)
        self.diagnostic(NIRdir, "NIRdir", date_UTC, target)

        return Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir

    def SHA_deg_from_doy_lat(self, doy: Raster, latitude: np.ndarray) -> Raster:
        """
        This function calculates sunrise hour angle in degrees from latitude in degrees and day of year between 1 and 365.
        """
        # calculate day angle in radians
        day_angle_rad = self.day_angle_rad_from_doy(doy)

        # calculate solar declination in degrees
        solar_dec_deg = self.solar_dec_deg_from_day_angle_rad(day_angle_rad)

        # convert latitude to radians
        latitude_rad = np.radians(latitude)

        # convert solar declination to radians
        solar_dec_rad = np.radians(solar_dec_deg)

        # calculate cosine of sunrise angle at latitude and solar declination
        # need to keep the cosine for polar correction
        sunrise_cos = -np.tan(latitude_rad) * np.tan(solar_dec_rad)

        # calculate sunrise angle in radians from cosine
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sunrise_rad = np.arccos(sunrise_cos)

        # convert to degrees
        sunrise_deg = np.degrees(sunrise_rad)

        # apply polar correction
        sunrise_deg = rt.where(sunrise_cos >= 1, 0, sunrise_deg)
        sunrise_deg = rt.where(sunrise_cos <= -1, 180, sunrise_deg)

        return sunrise_deg

    def sunrise_from_sha(self, sha_deg: Raster) -> Raster:
        """
        This function calculates sunrise hour from sunrise hour angle in degrees.
        """
        return 12.0 - (sha_deg / 15.0)

    def daylight_from_sha(self, sha_deg: Raster) -> Raster:
        """
        This function calculates daylight hours from sunrise hour angle in degrees.
        """
        return (2.0 / 15.0) * sha_deg

    def NDVI_to_LAI(self, NDVI: Raster) -> Raster:
        KPAR = 0.5
        fIPAR = rt.clip(NDVI - 0.05, 0, 1)
        LAI = rt.clip(-np.log(1 - fIPAR) * (1 / KPAR), 0, None)

        return LAI

    def Tmin_K(self, time_UTC: datetime, geometry: RasterGeometry, ST_K: Raster = None) -> Raster:
        self.logger.info("retrieving GEOS-5 FP minimum air temperature raster in Kelvin")
        Tmin_K = self.GEOS5FP_connection.Tmin_K(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

        return Tmin_K

    def Ta_K(self, time_UTC: datetime, geometry: RasterGeometry, ST_K: Raster = None) -> Raster:
        # TODO option to sharpen air temperature to surface temperature
        self.logger.info("retrieving GEOS-5 FP air temperature raster in Kelvin")
        Ta_K = self.GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

        if self.downscale_air and ST_K is not None:
            mean = np.nanmean(Ta_K)
            sd = np.nanstd(Ta_K)
            self.logger.info(f"downscaling air temperature with mean {mean} and sd {sd}")
            Ta_K = ST_K.contain(zscore(ST_K, nan_policy="omit") * sd + mean)

        return Ta_K

    def Ea_Pa(self, time_UTC: datetime, geometry: RasterGeometry, ST_K: Raster = None) -> Raster:
        # TODO option to sharpen vapor pressure to NDVI
        self.logger.info("retrieving GEOS-5 FP vapor pressure raster in Pascals")
        Ea_Pa = self.GEOS5FP_connection.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=self.resampling)

        if self.downscale_vapor and ST_K is not None:
            mean = np.nanmean(Ea_Pa)
            sd = np.nanstd(Ea_Pa)
            self.logger.info(f"downscaling vapor pressure with mean {mean} and sd {sd}")
            Ea_Pa = ST_K.contain(zscore(ST_K, nan_policy="omit") * -sd + mean)

        return Ea_Pa

    def NDVI_to_FVC(self, NDVI: Raster) -> Raster:
        NDVIv = 0.52  # +- 0.03
        NDVIs = 0.04  # +- 0.03
        FVC = rt.clip((NDVI - NDVIs) / (NDVIv - NDVIs), 0, 1)

        return FVC

    def day_angle_rad_from_doy(self, doy: Raster) -> Raster:
        """
        This function calculates day angle in radians from day of year between 1 and 365.
        """
        return (2 * np.pi * (doy - 1)) / 365

    def solar_dec_deg_from_day_angle_rad(self, day_angle_rad: float) -> float:
        """
        This function calculates solar declination in degrees from day angle in radians.
        """
        return (0.006918 - 0.399912 * np.cos(day_angle_rad) + 0.070257 * np.sin(day_angle_rad) - 0.006758 * np.cos(
            2 * day_angle_rad) + 0.000907 * np.sin(2 * day_angle_rad) - 0.002697 * np.cos(
            3 * day_angle_rad) + 0.00148 * np.sin(
            3 * day_angle_rad)) * (180 / np.pi)

    def Rn_daily(
            self,
            Rn: Raster,
            hour_of_day: Raster,
            sunrise_hour: Raster,
            daylight_hours: Raster) -> Raster:
        """
        calculate daily net radiation using solar parameters
        this is the average rate of energy transfer from sunrise to sunset
        in watts per square meter
        watts are joules per second
        to get the total amount of energy transferred, factor seconds out of joules
        the number of seconds for which this average is representative is (daylight_hours * 3600)
        documented in verma et al, bisht et al, and lagouARDe et al
        :param Rn:
        :param hour_of_day:
        :param sunrise_hour:
        :param daylight_hours:
        :return:
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return 1.6 * Rn / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))
