import logging
from typing import Callable
from datetime import datetime, timedelta
from os.path import join, abspath, expanduser
from typing import Dict, List

import numpy as np
import warnings

import colored_logging as cl
import rasters as rt
from rasters import Raster, RasterGrid
from geos5fp import GEOS5FP

from ..SRTM import SRTM
from ..model.model import DEFAULT_PREVIEW_QUALITY, DEFAULT_RESAMPLING, Model

from ..timer import Timer

__author__ = 'Kaniska Mallick, Madeleine Pascolini-Campbell, Gregory Halverson'

logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_STIC_INTERMEDIATE = "STIC_intermediate"

DEFAULT_OUTPUT_VARIABLES = [
    "LE",
    "LE_change",
    "LEt",
    "PT",
    "H",
    "PET",
    "G"
]

# constants
SIGMA = 5.67e-8  # Stefann Boltzmann constant
RHO = 1.2  # Air density (kg m-3)
CP = 1013  # Specific heat of air at constant pressure (J/kg/K)
GAMMA = 0.67  # Psychrometric constant (hpa/K)
ALPHA = 1.26  # Priestley-Taylor coefficient


def STIC_closure(delta, PHI, Es, Ea, Estar, M, rho=RHO, CP=CP, gamma=GAMMA, alpha=ALPHA):
    """
    STIC closure equations with modified Priestley Taylor and Penman Monteith
    (Mallick et al., 2015, Water Resources research)
    """
    gB = ((2 * PHI * alpha * delta * gamma) / (
            2 * CP * delta * Es * rho - 2 * CP * delta * Ea * rho - 2 * CP * Ea * gamma * rho + CP * Es * gamma * rho + CP * Estar * gamma * rho - CP * M * Es * gamma * rho + CP * M * Estar * gamma * rho))
    gB = rt.where(gB < 0, 0.0001, gB)
    gB = rt.where(gB > 0.2, 0.2, gB)
    gS = (-(2 * (PHI * alpha * delta * Ea * gamma - PHI * alpha * delta * Es * gamma)) / (
            CP * Estar ** 2 * gamma * rho - CP * Es ** 2 * gamma * rho - 2 * CP * delta * Es ** 2 * rho + 2 * CP * delta * Ea * Es * rho - 2 * CP * delta * Ea * Estar * rho + 2 * CP * delta * Es * Estar * rho + 2 * CP * Ea * Es * gamma * rho - 2 * CP * Ea * Estar * gamma * rho + CP * M * Es ** 2 * gamma * rho + CP * M * Estar ** 2 * gamma * rho - 2 * CP * M * Es * Estar * gamma * rho))
    gS = rt.where(gS < 0, 0.0001, gS)
    gS = rt.where(gS > 0.2, 0.2, gS)
    dT = rt.clip(((
                          2 * delta * Es - 2 * delta * Ea - 2 * Ea * gamma + Es * gamma + Estar * gamma - M * Es * gamma + M * Estar * gamma + 2 * alpha * delta * Ea - 2 * alpha * delta * Es) / (
                          2 * alpha * delta * gamma)), -10, 50)

    EF = rt.clip((-(2 * alpha * delta * Ea - 2 * alpha * delta * Es) / (
            2 * delta * Es - 2 * delta * Ea - 2 * Ea * gamma + Es * gamma + Estar * gamma - M * Es * gamma + M * Estar * gamma)),
                 0, 1)

    return gB, gS, dT, EF


def NDVI_to_FVC(NDVI):
    NDVIv = 0.52  # +- 0.03
    NDVIs = 0.04  # +- 0.03
    FVC = rt.clip((NDVI - NDVIs) / (NDVIv - NDVIs), 0, 1)

    return FVC


def f_G_PHI_actualsurface(Rn, ttSEC, M):
    # ttsec time in seconds of the day at ECOSTRESS overpass time.
    # Rn Net Rad
    # M soil moisture
    cgMIN = 0.05  # for wet surface
    cgMAX = 0.35  # for dry surface
    tgMIN = 74000  # for wet surface
    tgMAX = 100000  # for dry surface
    solNooN = 12. * 60. * 60
    tg0 = solNooN - ttSEC

    # Estimating GHF according to Santanello and Friedl (2003)
    cg = (1 - M) * cgMAX + M * cgMIN
    tg = (1 - M) * tgMAX + M * tgMIN

    G = Rn * cg * np.cos(2 * np.pi * (tg0 + 10800) / tg)
    # G(RN<0)                     = -G(Rn<0);
    G = rt.where(Rn < 0, -G, G)

    phi = Rn - G

    return G


def f_NetRadiation(SIGMA, Ta_C, Ea_hPa, ST_C, emissivity, RG, albedo):
    etaa = 1.24 * (Ea_hPa / (Ta_C + 273.15)) ** (1. / 7)  # air emissivity
    Lin = SIGMA * etaa * (Ta_C + 273.15) ** 4
    Lout = SIGMA * emissivity * (ST_C + 273.15) ** 4
    Lnet = emissivity * Lin - Lout
    # Rn_calc              = RG*(1 - albedo) + Lnet;

    # VARIABLES
    # TA    = air temperature
    # ea    = actual vapor pressure at air temperature (hPa)
    # TS    = surface temperature
    # Lin   = downwelling longwave radiation
    # Lout  = upwelling longwave radiation
    # Lnet  = net longwave radiation

    return Lnet


def f_SoilMoisture_INITIALIZE(GAMMA, delta, ST_C, Ta_C, Td_C, dTS, Rn, Lnet, NDVI, VPD_hPa, SVP_hPa, Ea_hPa, Estar):
    # COMMENT: This functions estimates the soil moisture availability (M) (or
    # wetness) (value 0 to 1) based on thermal IR and meteorological
    # information. However, this M will be treated as initial M, which will be
    # later on estimated through iteration in the actual ET estimation loop to
    # establish feedback between M and biophysical states

    # FC FROM NDVI
    # how do i calculate FC from NDVI

    fc = NDVI_to_FVC(NDVI)

    # TU COMPUTATION (Surface dewpoint temperature)
    s11 = (45.03 + 3.014 * Td_C + 0.05345 * Td_C ** 2 + 0.00224 * Td_C ** 3) * 1e-2;  # slope of SVP at TD (hpa/K)
    s22 = (Estar - Ea_hPa) / (ST_C - Td_C);
    s33 = (45.03 + 3.014 * ST_C + 0.05345 * ST_C ** 2 + 0.00224 * ST_C ** 3) * 1e-2;  # slope of SVP at TS (hpa/K)
    s44 = (SVP_hPa - Ea_hPa) / (Ta_C - Td_C);
    # s55        = (esstar - eastar)./(TS - TA);

    # Here s33 is approximated in TS for high dTS because the Taylor-Series linearity
    # asssumption gets violated when the difference between surface and air
    # temperature becomes very high

    # s33(TS-TA>5) = s55(TS-TA>5);

    # Surface dewpoint temperature (degC)
    Tsd_C = (Estar - Ea_hPa - s33 * ST_C + s11 * Td_C) / (s11 - s33);

    # Surface moisture (Msurf)
    Msurf = (s11 / s22) * ((Tsd_C - Td_C) / (ST_C - Td_C));  # Surface wetness
    Msurf = rt.where(Msurf > 1, 0.9999, Msurf);
    Msurf = rt.where(Msurf < 0, 0.0001, Msurf);

    # Surface vapor pressure and deficit
    esurf = Ea_hPa + Msurf * (Estar - Ea_hPa);
    # esurf   = ea + Msurf.*(eastar + slope.*(TS - TA)- ea);
    Dsurf = esurf - Ea_hPa;

    # Separating soil and canopy wetness to form a composite surface moisture
    Ms = Msurf;
    Mcan = fc * Msurf;
    Msoil = (1 - fc) * Msurf;

    TdewIndex = (ST_C - Tsd_C) / (Ta_C - Td_C);  # % TdewIndex > 1 signifies super dry condition
    Ep_PT = (1.26 * delta * Rn) / (
                delta + GAMMA);  # Potential evaporation (Priestley-Taylor eqn.)

    # surface wetness comes from the soil, vegetation contribution is neglegible
    # fix this

    # Ms(fc<=0.25 & TdewIndex<1)               = Msoil (fc<=0.25 & TdewIndex<1);
    Ms = rt.where((fc <= 0.25) & (TdewIndex < 1), Msoil, Ms)

    # Mcan(fc<=0.25 & TdewIndex<1)             = 0;
    Mcan = rt.where((fc <= 0.25) & (TdewIndex < 1), 0, Mcan)

    # Ms(fc<=0.25 & TA>10 & TD<0 & Lnet<-125)             = Msoil (fc<=0.25 & TA>10 & TD<0 & Lnet<-125);
    Ms = rt.where((fc <= 0.25) & (Ta_C > 10) & (Td_C < 0) & (Lnet < -125), Msoil, Ms)
    # Mcan(fc<=0.25 & TA>10 & TD<0 & Lnet<-125)           = 0;
    Mcan = rt.where((fc <= 0.25) & (Ta_C > 10) & (Td_C < 0) & (Lnet < -125), 0, Mcan)
    # es                                                  = ea + Ms.*(eastar + slope.*(TS - TA)- ea);

    # Rootzone Moisture (Mrz)
    Mrz = (GAMMA * s11 * (Tsd_C - Td_C)) / (
                delta * s33 * (ST_C - Td_C) + GAMMA * s44 * (Ta_C - Td_C) - delta * s11 * (Tsd_C - Td_C));

    # Mrz(Mrz>1)  = 0.9999; Mrz(Mrz<0) = 0.0001;
    Mrz = rt.where(Mrz > 1, 0.9999, Mrz)
    Mrz = rt.where(Mrz < 0, 0.0001, Mrz)

    # COMBINE M to account for Hysteresis and initial estimation of surface vapor pressure
    M = Ms
    # M(Ep_PT>RN & dTS>0)                             = Mrz(Ep_PT>RN & dTS>0) ;
    M = rt.where((Ep_PT > Rn) & (dTS > 0), Mrz, M)
    # M(Ep_PT>RN & fc<=0.25)                          = Mrz(Ep_PT>RN & fc<=0.25) ;
    M = rt.where((Ep_PT > Rn) & (fc <= 0.25), Mrz, M)

    # M(Ep_PT>RN & Dsurf>DA)                          = Mrz(Ep_PT>RN & Dsurf>DA) ;
    M = rt.where((Ep_PT > Rn) & (Dsurf > VPD_hPa), Mrz, M)
    # M(Ep_PT>RN & Dsurf>DA & TdewIndex<1)            = Mrz(Ep_PT>RN & Dsurf>DA & TdewIndex<1) ;
    # M(Ep_PT>RN & dTS>0 & fc<=0.25 & Dsurf>DA & TdewIndex<1) = Mrz(Ep_PT>RN & dTS>0 & fc<=0.25 & Dsurf>DA & TdewIndex<1) ;

    # M(fc<=0.25 & dTS>0 & TA>10 & TD<0 & Lnet<-125)  = Mrz(fc<=0.25 & dTS>0 & TA>10 & TD<0 & Lnet<-125) ;
    M = rt.where((fc <= 0.25) & (dTS > 0) & (Ta_C > 10) & (Td_C < 0) & (Lnet < -125), Mrz, M)

    # M(fc<=0.25 & dTS>0 & TA>10 & TD<0 & Dsurf>DA)   = Mrz(fc<=0.25 & dTS>0 & TA>10 & TD<0 & Dsurf>DA) ;
    M = rt.where((fc <= 0.25) & (dTS > 0) & (Ta_C > 10) & (Td_C < 0) & (Dsurf > VPD_hPa), Mrz, M)

    # M(Ep_PT<RN & fc<=0.25 & Dsurf>DA)               = Mrz(Ep_PT<RN & fc<=0.25 & Dsurf>DA) ;
    M = rt.where((Ep_PT < Rn) & (fc <= 0.25) & (Dsurf > VPD_hPa), Mrz, M)
    # M(RN>0 & dTS>0 & TD<=0)                         = Mrz(RN>0 & dTS>0 & TD<=0) ;

    es = Ea_hPa + M * (Estar - Ea_hPa)
    # es          = ea + M.*(eastar + slope.*(TS - TA)- ea);
    Ds = (Estar - es)  # vapor pressure deficit at surface

    s1 = s11
    s3 = s33

    return M, Mrz, Ms, Ep_PT, Ds, s1, s3, Tsd_C


def f_SoilMoisture_ITERATE2(
        GAMMA,
        delta,
        s1,
        s3,
        ST_C,
        Ta_C,
        dTS,
        TD,
        TSD,
        RG,
        Rn,
        Lnet,
        fc,
        DA,
        D0,
        SVP_hPa,
        Ea_hPa,
        T0):
    # COMMENT: This functions estimates the soil moisture availability (M) (or
    # wetness) (value 0 to 1) based on thermal IR and meteorological
    # information. However, this M will be treated as initial M, which will be
    # later on estimated through iteration in the actual ET estimation loop to
    # establish feedback between M and biophysical states

    # call this in the iterative portion

    #########################################################################
    # % Surface Moisture (Msurf)
    # #########################################################################
    kTSTD = (T0 - TD) / (ST_C - TD)
    Msurf = (s1 / s3) * ((TSD - TD) / (kTSTD * (ST_C - TD)))  # Surface wetness

    # Msurf(RN<0 & dTS<0 & Msurf<0) = abs(Msurf(RN<0 & dTS<0 & Msurf<0));
    Msurf = rt.where((Rn < 0) & (dTS < 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf(RN<0 & dTS>0 & Msurf<0) = abs(Msurf(RN<0 & dTS>0 & Msurf<0));
    Msurf = rt.where((Rn < 0) & (dTS > 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf(RN>0 & dTS<0 & Msurf<0) = abs(Msurf(RN>0 & dTS<0 & Msurf<0));
    Msurf = rt.where((Rn > 0) & (dTS < 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf(RN>0 & dTS>0 & Msurf<0) = abs(Msurf(RN>0 & dTS>0 & Msurf<0));
    Msurf = rt.where((Rn > 0) & (dTS > 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf(RG>0 & Msurf<0) = abs(Msurf(RG>0 & Msurf<0));
    Msurf = rt.where((RG > 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf(RG<0 & Msurf<0) = abs(Msurf(RG<0 & Msurf<0));
    Msurf = rt.where((RG < 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf(TD<0 & Msurf<0) = abs(Msurf(TD<0 & Msurf<0));
    Msurf = rt.where((TD < 0) & (Msurf < 0), np.abs(Msurf), Msurf)
    # Msurf (Msurf>1) = 1; Msurf(Msurf<0) = 0.0001;
    Msurf = rt.where(Msurf > 1, 1, Msurf)
    Msurf = rt.where(Msurf < 0, 0.0001, Msurf)

    # Separating soil and canopy wetness to form a composite surface moisture
    Ms = Msurf
    Mcan = fc * Msurf
    Msoil = (1 - fc) * Msurf

    TdewIndex = (ST_C - TSD) / (Ta_C - TD)
    Ep_PT = (1.26 * delta * Rn) / (
                delta + GAMMA)  # Potential evaporation (Priestley-Taylor eqn.)

    # Ms(Ep_PT>RN & fc<=0.25 & TdewIndex<1)               = Msoil (Ep_PT>RN & fc<=0.25 & TdewIndex<1);
    Ms = rt.where((Ep_PT > Rn) & (fc <= 0.25) & (TdewIndex < 1), Msoil, Ms)
    # Ms(fc<=0.25 & TA>10 & TD<0 & Lnet<-125)             = Msoil (fc<=0.25 & TA>10 & TD<0 & Lnet<-125);
    Ms = rt.where((fc <= 0.25) & (Ta_C > 10) & (TD < 0) & (Lnet < -125), Msoil, Ms)
    # Ms(RN>Ep_PT & fc<=0.25 & TdewIndex<1 & TD<=0)       = Msoil (RN>Ep_PT & fc<=0.25 & TdewIndex<1 & TD<=0);
    Ms = rt.where((Rn > Ep_PT) & (fc <= 0.25) & (TdewIndex < 1) & (TD <= 0), Msoil, Ms)
    # Ms(RN>Ep_PT & fc<=0.25 & TdewIndex<1)               = Msoil (RN>Ep_PT & fc<=0.25 & TdewIndex<1);
    Ms = rt.where((Rn > Ep_PT) & (fc <= 0.25) & (TdewIndex < 1), Msoil, Ms)
    # Ms(D0>DA & fc<=0.25 & TdewIndex<1)                  = Msoil (D0>DA & fc<=0.25 & TdewIndex<1);
    Ms = rt.where((D0 > DA) & (fc <= 0.25) & (TdewIndex < 1), Msoil, Ms)
    #  Mcan(Ep_PT>RN & fc<=0.25 & TdewIndex<1)             = 0;
    Mcan = rt.where((Ep_PT > Rn) & (fc <= 0.25) & (TdewIndex < 1), 0, Mcan)
    # Mcan(fc<=0.25 & TA>10 & TD<0 & Lnet<-125)           = 0;
    Mcan = rt.where((fc <= 0.25) & (Ta_C > 10) & (TD < 0) & (Lnet < -125), 0, Mcan)
    # Mcan(RN>Ep_PT & fc<=0.25 & TdewIndex<1 & TD<=0)     = 0;
    Mcan = rt.where((Rn > Ep_PT) & (fc <= 0.25) & (TdewIndex < 1) & (TD <= 0), 0, Mcan)
    # Mcan(RN>Ep_PT & fc<=0.25 & TdewIndex<1)             = 0;
    Mcan = rt.where((Rn > Ep_PT) & (fc <= 0.25) & (TdewIndex < 1), 0, Mcan)
    # Mcan(D0>DA & fc<=0.25 & TdewIndex<1)                = 0;
    Mcan = rt.where((D0 > DA) & (fc <= 0.25) & (TdewIndex < 1), 0, Mcan)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Rootzone Moisture (Mrz)
    #########################################################################
    s44 = (SVP_hPa - Ea_hPa) / (Ta_C - TD)

    Mrz = (GAMMA * s1 * (TSD - TD)) / (
                delta * s3 * kTSTD * (ST_C - TD) + GAMMA * s44 * (Ta_C - TD) - delta * s1 * (TSD - TD))

    Mrz = rt.where((Rn < 0) & (dTS < 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where((Rn < 0) & (dTS > 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where((Rn > 0) & (dTS < 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where((Rn > 0) & (dTS > 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where((RG > 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where((RG < 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where((TD < 0) & (Mrz < 0), np.abs(Mrz), Mrz)
    Mrz = rt.where(Mrz > 1, 1, Mrz)
    Mrz = rt.where(Mrz < 0, 0.0001, Mrz)

    TdewIndex = (ST_C - TSD) / (Ta_C - TD)
    Ep_PT = (1.26 * delta * Rn) / (
                delta + GAMMA)  # Potential evaporation (Priestley-Taylor eqn.)

    # COMBINE M to account for Hysteresis and initial estimation of surface vapor pressure
    M = Msurf
    M = rt.where((Ep_PT > Rn) & (dTS > 0) & (fc <= 0.25) & (D0 > DA) & (TdewIndex < 1), Mrz, M)
    M = rt.where((fc <= 0.25) & (dTS > 0) & (Ta_C > 10) & (TD < 0) & (Lnet < -125) & (D0 > DA), Mrz, M)

    return M


def calculate_G_SEBAL(Rn, ST_C, NDVI, albedo):
    return Rn * ST_C * (0.0038 + 0.0074 * albedo) * (1 - 0.98 * NDVI ** 4)


class STIC(Model):
    def __init__(
            self,
            working_directory: str = None,
            static_directory: str = None,
            SRTM_connection: SRTM = None,
            SRTM_download: str = None,
            GEOS5FP_connection: GEOS5FP = None,
            GEOS5FP_download: str = None,
            GEOS5FP_products: str = None,
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
            intermediate_directory = join(working_directory, DEFAULT_STIC_INTERMEDIATE)

        super(STIC, self).__init__(
            working_directory=working_directory,
            static_directory=static_directory,
            intermediate_directory=intermediate_directory,
            preview_quality=preview_quality,
            resampling=resampling,
            save_intermediate=save_intermediate,
            show_distribution=show_distribution,
            include_preview=include_preview
        )

        self.downscale_air = downscale_air
        self.downscale_vapor = downscale_vapor

    def STIC(
            self,
            geometry: RasterGrid,
            target: str,
            time_UTC: datetime or str,
            Rn: Raster,
            RH: Raster,
            Ta_C: Raster,
            ST_C: Raster,
            albedo: Raster,
            emissivity: Raster,
            NDVI: Raster,
            G: Raster = None,
            Rg: Raster = None,
            water: Raster = None,
            output_variables: List[str] = None,
            LE_convergence_target: float = 1.0,
            max_iterations: int = 3,
            results: Dict = None):
        if results is None:
            results = {}

        if output_variables is None:
            output_variables = DEFAULT_OUTPUT_VARIABLES

        date_UTC = time_UTC.date()

        centroid_lon = geometry.centroid_latlon.x
        time_solar = time_UTC + timedelta(hours=(np.radians(centroid_lon) / np.pi * 12))
        ttSEC = time_solar.hour * 3600 + time_solar.minute * 60 + time_solar.second

        warnings.filterwarnings('ignore')

        Rn = Rn.mask(~water)
        ST_C = ST_C.mask(~water)

        self.diagnostic(Rn, "Rn", date_UTC, target)
        self.diagnostic(RH, "RH", date_UTC, target)
        self.diagnostic(Ta_C, "Ta_C", date_UTC, target)
        self.diagnostic(ST_C, "ST_C", date_UTC, target)
        self.diagnostic(albedo, "albedo", date_UTC, target)
        self.diagnostic(emissivity, "emissivity", date_UTC, target)
        self.diagnostic(NDVI, "NDVI", date_UTC, target)

        if G is None:
            G = calculate_G_SEBAL(Rn, ST_C, NDVI, albedo)

        self.diagnostic(G, "G", date_UTC, target)
        # net available energy
        phi = Rn - G
        self.diagnostic(phi, "phi", date_UTC, target)
        # psychrometric computation
        # saturation vapor pressure at given air temperature (hpa/K) (1hPa = 1mb)
        SVP_hPa = 6.13753 * (np.exp((17.27 * Ta_C) / (Ta_C + 237.3)))
        self.diagnostic(SVP_hPa, "SVP_hPa", date_UTC, target)
        # actual vapor pressure at TA (hpa/K)
        Ea_hPa = SVP_hPa * (RH)
        self.diagnostic(Ea_hPa, "Ea_hPa", date_UTC, target)
        # vapor pressure deficit (hPa)
        VPD_hPa = SVP_hPa - Ea_hPa
        self.diagnostic(VPD_hPa, "VPD_hPa", date_UTC, target)
        # slope of saturation vapor pressure to air temperature (hpa/K)
        delta = 4098 * SVP_hPa / (Ta_C + 237.3) ** 2
        self.diagnostic(delta, "delta", date_UTC, target)
        # swapping in the dew-point calculation from PT-JPL
        Td_C = Ta_C - ((100 - RH * 100) / 5.0)
        self.diagnostic(Td_C, "Td_C", date_UTC, target)
        # surface air temperature difference
        dTS = ST_C - Ta_C
        self.diagnostic(dTS, "dTS", date_UTC, target)
        # saturation vapor pressure at surface temperature (hPa/K)
        Estar = 6.13753 * np.exp((17.27 * ST_C) / (ST_C + 237.3))
        self.diagnostic(Estar, "Estar", date_UTC, target)
        # Surface Dewpoint Temperature

        if Rg is None:

            s33 = (45.03 + 3.014 * ST_C + 0.05345 * ST_C ** 2 + 0.00224 * ST_C ** 3) * 1e-2  # hpa/K
            self.diagnostic(s33, "s33", date_UTC, target)
            s1 = (45.03 + 3.014 * Td_C + 0.05345 * Td_C ** 2 + 0.00224 * Td_C ** 3) * 1e-2  # hpa/K
            self.diagnostic(s1, "s1", date_UTC, target)
            # Surface dewpoint (Celsius)
            Tsd_C = ((Estar - Ea_hPa) - (s33 * ST_C) + (s1 * Td_C)) / (s1 - s33)
            self.diagnostic(Tsd_C, "Tsd_C", date_UTC, target)
            # slope of saturation vapor pressure and temperature
            s3 = rt.where((dTS > -20) & (dTS < 5), (Estar - Ea_hPa) / (ST_C - Td_C),
                          (45.03 + 3.014 * ST_C + 0.05345 * ST_C ** 2 + 0.00224 * ST_C ** 3) * 1e-2)  # hpa/K
            self.diagnostic(s3, "s3", date_UTC, target)
            # Surface Moisture (Ms)
            # Surface wetness
            Ms = (s1 / s3) * ((Tsd_C - Td_C) / (ST_C - Td_C))
            Ms = rt.clip(rt.where((dTS < 0) & (Ms < 0) & (phi < 0), np.abs(Ms), Ms), 0, 1)
            self.diagnostic(Ms, "Ms", date_UTC, target)
            # Rootzone Moisture (Mrz)
            s44 = (SVP_hPa - Ea_hPa) / (Ta_C - Td_C)
            self.diagnostic(s44, "s44", date_UTC, target)
            Mrz = (GAMMA * s1 * (Tsd_C - Td_C)) / (
                        delta * s3 * (ST_C - Td_C) + GAMMA * s44 * (Ta_C - Td_C) - delta * s1 * (
                        Tsd_C - Td_C))  # rootzone wetness
            Mrz = rt.clip(rt.where((dTS < 0) & (Mrz < 0) & (phi < 0), np.abs(Mrz), Mrz), 0, 1)
            self.diagnostic(Mrz, "Mrz", date_UTC, target)
            # now the limits of both Ms and Mrz are consistent
            # combine M to account for Hysteresis and initial estimation of surface vapor pressure
            # Potential evaporation (Priestley-Taylor eqn.)
            Ep_PT = (ALPHA * delta * phi) / (delta + GAMMA)
            self.diagnostic(Ep_PT, "Ep_PT", date_UTC, target)
            Es = rt.where((Ep_PT > phi) & (dTS > 0) & (Td_C <= 0), Ea_hPa + Mrz * (Estar - Ea_hPa),
                          Ea_hPa + Ms * (Estar - Ea_hPa))
            self.diagnostic(Es, "Es", date_UTC, target)
            M = rt.where((Ep_PT > phi) & (dTS > 0) & (Td_C <= 0), Mrz, Ms)
            self.diagnostic(M, "M", date_UTC, target)
            # hysteresis logic
            # vapor pressure deficit at surface (Ds is later replaced by D0)
            Ds = (Estar - Es)
            self.diagnostic(Ds, "Ds", date_UTC, target)

        else:

            # FC FROM NDVI
            fc = NDVI_to_FVC(NDVI)

            # Rn SOIL
            #  kPAR = 0.5
            kRN = 0.6
            lai = -np.log(1 - fc) / 0.5
            Rnsoil = Rn * np.exp(-kRN * lai)

            Lnet = f_NetRadiation(SIGMA, Ta_C, Ea_hPa, ST_C, emissivity, Rg, albedo)
            # Get M from f_soilmoisture initialize
            M, Mrz, Ms, Ep_PT, Ds, s1, s3, Tsd_C = f_SoilMoisture_INITIALIZE(GAMMA, delta, ST_C, Ta_C, Td_C, dTS, Rn,
                                                                             Lnet, NDVI, VPD_hPa, SVP_hPa, Ea_hPa,
                                                                             Estar)
            # get G from new function
            # G = f_G_PHI_actualsurface(Rn, Rnsoil, ttSEC, M)

            G = f_G_PHI_actualsurface(Rn, ttSEC, M)
            # get phi with new comp
            phi = Rn - G

            Es = rt.where((Ep_PT > phi) & (dTS > 0) & (Td_C <= 0), Ea_hPa + Mrz * (Estar - Ea_hPa),
                          Ea_hPa + Ms * (Estar - Ea_hPa))

        # STIC analytical equations (convergence on LE)
        [gB, gS, dT, EF] = STIC_closure(delta, phi, Es, Ea_hPa, Estar, M)

        gBB = gB
        gSS = gS

        gBB_by_gSS = rt.where(gSS == 0, 0, gBB / gSS)

        self.diagnostic(gB, "gB", date_UTC, target)
        self.diagnostic(gS, "gS", date_UTC, target)
        self.diagnostic(dT, "dT", date_UTC, target)
        self.diagnostic(EF, "EF", date_UTC, target)
        gB_by_gS = rt.where(gS == 0, 0, gB / gS)
        self.diagnostic(gB_by_gS, "gBB_by_gSS", date_UTC, target)
        dT = dT
        T0 = dT + Ta_C
        self.diagnostic(T0, "T0", date_UTC, target)
        PET = ((delta * phi + RHO * CP * gB * VPD_hPa) / (delta + GAMMA))  # Penman potential evaporation
        self.diagnostic(PET, "PET", date_UTC, target)
        # Hysteresis logic (when ET is governed by root zone radiative conductance takes the main role)
        gR = (4 * SIGMA * (Ta_C + 273) ** 3 * emissivity) / (RHO * CP)
        self.diagnostic(gR, "gR", date_UTC, target)
        # normal no water stress initial LE
        omega = ((delta / GAMMA) + 1) / ((delta / GAMMA) + 1 + gB_by_gS)
        LE_eq = (phi * (delta / GAMMA)) / ((delta / GAMMA) + 1)
        LE_imp = (CP * 0.0289644 / GAMMA) * gS * 40 * VPD_hPa
        self.diagnostic(LE_imp, "LE_imp", date_UTC, target)
        LE_init = omega * LE_eq + (1 - omega) * LE_imp
        # super dry conditions (I Put this condition back) (COMMENT by K.M on 21/02/2017)
        dry = (Ds > VPD_hPa) & (PET > phi) & (dTS > 0) & (Td_C <= 0)
        self.diagnostic(dry, "dry", date_UTC, target)
        omega = rt.where(dry,
                         ((delta / GAMMA) + 1 + gR / gB) / ((delta / GAMMA) + 1 + gB / gS + gR / gS + gR / gB),
                         omega)
        self.diagnostic(omega, "omega", date_UTC, target)
        LE_eq = rt.where(dry, (phi * (delta / GAMMA)) / ((delta / GAMMA) + 1 + gR / gB), LE_eq)
        self.diagnostic(LE_eq, "LE_eq", date_UTC, target)
        LE_init = rt.where(dry, omega * LE_eq + (1 - omega * LE_imp), LE_init)
        H = ((GAMMA * phi * (1 + gB_by_gS) - RHO * CP * gB * VPD_hPa) / (delta + GAMMA * (1 + (gB_by_gS))))
        self.diagnostic(H, "H", date_UTC, target)
        LE_new = LE_init
        LE_change = LE_convergence_target
        LE_old = LE_new
        LEt = None
        PT = None
        iteration = 1
        LE_max_change = 0
        t = Timer()

        while (np.nanmax(LE_change) >= LE_convergence_target and iteration <= max_iterations):
            logger.info(f"running STIC iteration {cl.val(iteration)} / {cl.val(max_iterations)}")

            if Rg is None:

                # canopy air stream vapor pressures
                e0star = Ea_hPa + (GAMMA * LE_new * (gB + gS)) / (RHO * CP * gB * gS)
                e0star = rt.where(e0star < 0, Estar, e0star)
                e0star = rt.where(e0star > 250, Estar, e0star)
                self.diagnostic(e0star, f"e0star_{iteration}", date_UTC, target)
                D0 = VPD_hPa + (delta * phi - (delta + GAMMA) * LE_new) / (
                        RHO * CP * gB)  # vapor pressure deficit at source
                D0 = rt.where(D0 < 0, Ds, D0)
                self.diagnostic(D0, f"D0_{iteration}", date_UTC, target)
                e0 = e0star - D0
                e0 = rt.where(e0 < 0, Es, e0)
                e0 = rt.where(e0 > e0star, e0star, e0)
                self.diagnostic(e0, f"e0_{iteration}", date_UTC, target)
                # re-estimating M (direct LST feedback into M computation)
                s1 = (45.03 + 3.014 * Td_C + 0.05345 * Td_C ** 2 + 0.00224 * Td_C ** 3) * 1e-2
                self.diagnostic(s1, f"s1_{iteration}", date_UTC, target)
                Tsd_C = Td_C + (GAMMA * LE_new) / (RHO * CP * gB * s1)
                self.diagnostic(Tsd_C, f"Tsd_C_{iteration}", date_UTC, target)
                # Surface Moisture Ms
                Ms = rt.clip(s1 * (Tsd_C - Td_C) / (s3 * (ST_C - Td_C)), 0, 1)
                self.diagnostic(Ms, f"Ms_{iteration}", date_UTC, target)
                # Root zone moisture Mrz
                Mrz = rt.clip(((GAMMA * s1 * (Tsd_C - Td_C)) / (
                        delta * s3 * (ST_C - Td_C) + GAMMA * s44 * (Ta_C - Td_C) - delta * s1 * (Tsd_C - Td_C))), 0, 1)
                self.diagnostic(Mrz, f"Mrz_{iteration}", date_UTC, target)
                # combining hysteresis logic to differentiate surface vs. rootzone water control
                M = rt.where((D0 > VPD_hPa) & (PET > phi) & (dTS > 0), Mrz, M)
                M = rt.where((phi > 0) & (dTS > 0) & (Td_C <= 0), Mrz, M)
                M = rt.clip(M, 0, 1)
                self.diagnostic(M, f"M_{iteration}", date_UTC, target)
                # checking convergence
                # re-estimating alpha
                alphaN = ((gS * (e0star - Ea_hPa) * (2 * delta + 2 * GAMMA + GAMMA * gB_by_gS * (1 + M))) / (
                        2 * delta * (GAMMA * (T0 - Ta_C) * (gB + gS) + gS * (e0star - Ea_hPa))))
                self.diagnostic(alphaN, f"alphaN_{iteration}", date_UTC, target)



            else:

                # =============================================================================
                # Call these modules to use the NEW version of STIC.

                # =============================================================================
                # CANOPY-AIR STREAM vapor pressures
                e0star = Ea_hPa + (GAMMA * LE_new * (gBB + gSS)) / (RHO * CP * gBB * gSS)
                e0star = rt.where(e0star < 0, Estar, e0star)
                e0star = rt.where(e0star > 250, Estar, e0star)

                D0 = VPD_hPa + (delta * phi - (delta + GAMMA) * LE_new) / (
                            RHO * CP * gBB)  # vapor pressure deficit at source/

                e0 = e0star - D0
                e0 = rt.where(e0 < 0, Es, e0)
                e0 = rt.where(e0 > e0star, e0star, e0)

                # Get M from f_soilmoisture initialize
                M = f_SoilMoisture_ITERATE2(GAMMA, delta, s1, s3, ST_C, Ta_C, dTS, Td_C, Tsd_C, Rg, Rn, Lnet, fc,
                                            VPD_hPa, D0, SVP_hPa, Ea_hPa, T0)

                # get G from new function
                # G = f_G_PHI_actualsurface(Rn,ttSEC,M)
                # re-calculate G!

                # update this
                # G = f_G_PHI_actualsurface(Rn, Rnsoil, ttSEC, M)
                G = f_G_PHI_actualsurface(Rn, ttSEC, M)
                # recompute phi
                phi = Rn - G

                alphaN = ((gSS * (e0star - Ea_hPa) * (2 * delta + 2 * GAMMA + GAMMA * gBB_by_gSS * (1 + M))) / (
                            2 * delta * (GAMMA * (T0 - Ta_C) * (gBB + gSS) + gSS * (e0star - Ea_hPa))))

            # re-estimated conductances and states
            [gB, gS, dT, EF] = STIC_closure(delta, phi, e0, Ea_hPa, e0star, M, RHO, CP, GAMMA, alphaN)

            self.diagnostic(gB, f"gB_{iteration}", date_UTC, target)
            self.diagnostic(gS, f"gS_{iteration}", date_UTC, target)
            self.diagnostic(dT, f"dT_{iteration}", date_UTC, target)
            self.diagnostic(EF, f"EF_{iteration}", date_UTC, target)
            gB_by_gS = rt.where(gS == 0, 0, gB / gS)
            self.diagnostic(gB_by_gS, f"gB_by_gS_{iteration}", date_UTC, target)
            T0 = dT + Ta_C
            self.diagnostic(T0, f"T0_{iteration}", date_UTC, target)
            # latent heat flux
            LE_new = ((delta * phi + RHO * CP * gB * VPD_hPa) / (delta + GAMMA * (1 + gB_by_gS)))
            LE_new = rt.where(LE_new > phi, phi, LE_new)
            self.diagnostic(LE_new, f"LE_new_{iteration}", date_UTC, target)
            # Sensible Heat Flux
            H = ((GAMMA * phi * (1 + gB_by_gS) - RHO * CP * gB * VPD_hPa) / (delta + GAMMA * (1 + (gB_by_gS))))
            self.diagnostic(H, f"H_{iteration}", date_UTC, target)
            # potential evaporation (Penman)
            PET = ((delta * phi + RHO * CP * gB * VPD_hPa) / (delta + GAMMA))
            self.diagnostic(PET, f"PET_{iteration}", date_UTC, target)
            # Potential Transpiration
            PT = (delta * phi + RHO * CP * gB * VPD_hPa) / (
                    delta + GAMMA * (1 + M * gB_by_gS))  # potential transpiration
            PT = PT.mask(~water)
            self.diagnostic(PT, f"PT_{iteration}", date_UTC, target)
            # ET PARTITIONING
            LEs = rt.clip(M * PET, 0, None).mask(~water)
            self.diagnostic(LEs, f"LEs_{iteration}", date_UTC, target)
            LEt = rt.clip(LE_new - LEs, 0, None).mask(~water)
            self.diagnostic(LEt, f"LEt_{iteration}", date_UTC, target)
            # change in latent heat flux estimate
            LE_change = np.abs(LE_old - LE_new)
            self.diagnostic(LE_change, f"LE_change_{iteration}", date_UTC, target)
            LE_new = rt.where(np.isnan(LE_new), LE_old, LE_new)
            LE_old = LE_new
            LE_max_change = np.nanmax(LE_change)
            logger.info(
                f"completed STIC iteration {cl.val(iteration)} / {cl.val(max_iterations)} with max LE change: {cl.val(LE_max_change)} ({t} seconds)")
            iteration += 1

        iteration -= 1
        results["LE_max_change"] = LE_max_change
        results["iteration"] = iteration

        LE = LE_new
        self.diagnostic(LE, "LE", date_UTC, target)

        if "LE" in output_variables:
            results["LE"] = LE

        if "LE_change" in output_variables:
            results["LE_change"] = LE_change

        if "LEt" in output_variables:
            results["LEt"] = LEt

        if "PT" in output_variables:
            results["PT"] = PT

        if "PET" in output_variables:
            results["PET"] = PET

        if "G" in output_variables:
            results["G"] = G

        warnings.resetwarnings()

        return results