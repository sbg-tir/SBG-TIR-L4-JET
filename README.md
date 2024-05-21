# SBG-TIR OTTER L3T ET, L4T ESI, and L4T WUE Data Products

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

Madeleine Pascolini-Campbell (she/her)<br>
[madeleine.a.pascolini-campbell@jpl.nasa.gov](mailto:madeleine.a.pascolini-campbell@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329F

Claire Villanueva-Weeks (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

This is the main repository for the Suface Biology and Geology Thermal Infrared (SBG-TIR) level 3 & 4 evapotranspiration data product generation software. 

This software will produce estimates of:
- evapotranspiration (ET)
- evaporative stress index (ESI)
- water use efficiency (WUE)

The SBG evapotranspiration product combines the [surface temperature and emissivity observations from the OTTER sensor](https://github.com/sbg-tir/SBG-TIR-L2-LSTE) with the [NDVI and albedo estimated by STARS](https://github.com/sbg-tir/SBG-TIR-L2-STARS), estimates near-surface meteorology by downscaling GEOS-5 FP to these three high resolution images, and runs these variables through a set of surface energy balance models.

## Meteorology

Coarse resolution near-surface air temperature (Ta) and relative
humidity (RH) are taken from the GEOS-5 FP tavg1_2d_slv_Nx product. Ta
and RH are down-scaled using a linear regression between up-sampled ST,
NDVI, and albedo as predictor variables to Ta or RH from GEOS-5 FP as a
response variable, within each Sentinel tile. These regression
coefficients are then applied to the 60 m ST, NDVI, and albedo, and this
first-pass estimate is then bias-corrected to the coarse image from
GEOS-5 FP. Areas of cloud are filled in with
bi-cubically resampled GEOS-5 FP. These downscaled meteorology estimates include:

L3T ET Meteorology Data Layers:
- Near-Surface Air Temperature in Celcius (Ta)
- Near-Surface Relative Humidity [0-1] (RH)

## Soil Moisture

This same down-scaling procedure is applied to soil moisture (SM) from
the GEOS-5 FP tavg1_2d_lnd_Nx product. This is included as a single data layer:

L3T ET Soil Moisture Data Layers:
- Soil Moisture [0-1] (SM)

## Surface Energy Balance

The SBG surface energy balance workflow begins with an artificial neural network (ANN) implementation of the Forest Light Environmental Simulator (FLiES) radiative transfer algorithm, following the workflow established by Dr. Hideki Kobayashi and Dr. Youngryel Ryu. GEOS-5 FP provides sub-daily Cloud Optical Thickness (COT) in the tavg1_2d_rad_Nx product and Aerosol Optical Thickness (AOT) from tavg3_2d_aer_Nx. Together with STARS albedo, these variables are run through the ANN implementation of FLiES to estimate incoming shortwave radiation (Rg), bias-corrected to Rg from the GEOS-5 FP tavg1_2d_rad_Nx product.

The Breathing Earth System Simulator (BESS) algorithm, contributed by Dr. Youngryel Ryu, iteratively calculates net radiation (Rn), ET, and Gross Primary Production (GPP) estimates. The BESS Rn is used as the Rn input to the remaining ET models. 

Net Radiation Data Layers:
- Net Radiation in Watts Per Square Meter (Rn)

## Evapotranspiration

The PT-JPL-SM model, developed by Dr. Adam Purdy and Dr. Joshua Fisher was designed as a SM-sensitive evapotranspiration product for the Soil Moisture Active-Passive (SMAP) mission, and then reimplemented as an ET model in the SBG ensemble, using the downscaled soil moisture from the L3T SM product. Similar to the PT-JPL model used in SBG Collection 1, The PT-JPL-SM model estimates instantaneous canopy transpiration, leaf surface evaporation, and soil moisture evaporation using the Priestley-Taylor formula with a set of constraints. The total instantaneous ET estimate combining these three partitions is recorded in the L3T ET product as PTJPLSMinst. The proportion of instantaneous canopy transpiration is recorded as PTJPLSMcanopy, leaf surface evaporation as PTJPLSMinterception, and soil moisture as PTJPLSMsoil.

The Surface Temperature Initiated Closure (STIC) model, contributed by Dr. Kaniska Mallick, was designed as an surface temperature sensitive ET model, adopted by SBG for improved diurnal estimates of ET. The STIC instantaneous ET is recorded in the L3T ET product as STICinst.

The MOD16 algorithm was designed as the ET product for the Moderate Resolution Imaging Spectroradiometer (MODIS). MOD16 uses a similar approach to PT-JPL and PT-JPL-SM to independently estimate vegetation and soil components of instantaneous ET, but using the Penman-Monteith formula instead of Priestley-Taylor. It is provided here as an additional estimate in the L3T ET product, MOD16inst.

The ET estimate from BESS is recorded in the L3T ET product as BESSinst. The median of PTJPLSMinst, STICinst, MOD16inst, and BESSinst is upscaled to a daily ET estimate in millimeters per day and recorded in the L3T ET product as ETdaily. The standard deviation between these multiple estimates of ET is considered the uncertainty for the SBG evapotranspiration product, as ETinstUncertainty. Note that the ETdaily product represents the integrated ET between sunrise and sunset.

L3T ET evapotranspiration data layers:
- Daily Evapotranspiration in mm/day (ETdaily)
- Evapotranspiration Uncertainty in mm/day (ETdailyUncertainty)

## Evaporative Stress Index

The PT-JPL-SM model generates estimates of both actual and potential instantaneous ET. The potential evapotranspiration (PET) estimate represents the maximum expected ET if there were no water stress to plants on the ground. The ratio of the actual ET estimate to the PET estimate forms an index representing the water stress of plants, with zero being fully stressed with no observable ET and one being non-stressed with ET reaching PET. 

L4T ESI Data Layers:
- Evaporative Stress Index [0-1] (ESI)
- Potential Evapotranspiration in mm/day (ETo)

## Water Use Efficiency

The BESS GPP estimate represents the amount of carbon that plants are taking in. The transpiration component of PT-JPL-SM represents the amount of water that plants are releasing. The BESS GPP is divided by the PT-JPL-SM transpiration to estimate water use efficiency (WUE), the ratio of grams of carbon that plants take in to kilograms of water that plants release. 

L4T WUE Data Layers:
- Water Use Efficiency g C kg^-1^ H~2~O (WUE)
- Gross Primary Production $\mu$mol m^-2^ s^-1^ (GPP)
