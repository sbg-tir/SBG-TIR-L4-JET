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

The repositories for the evapotranspiration algorithms are located in the [JPL-Evapotranspiration-Algorithms](https://github.com/JPL-Evapotranspiration-Algorithms) organization.

## Introduction to Data Products

This is the user guide for the SBG tiled products. SBG acquires data within an orbit, and this orbit path is divided into scenes roughly 935 x 935 km in size. The SBG orbit/scene/tile products are distributed in Cloud-Optimized GeoTIFF (COG) format. The tiled products are listed in Table 1.

| **Product Long Name** | **Product Short Name** |
| --- | --- |
| STARS NDVI/Albedo | L2T STARS |
| Surface Energy Balance | L3T SEB |
| Soil Moisture | L3T SM |
| Meteorology | L3T MET |
| Evapotranspiration Ensemble | L3T ET |
| DisALEXI-JPL Evapotranspiration | L3T ET ALEXI |
| Evaporative Stress Index | L4T ESI |
| DisALEXI-JPL Evaporative Stress Index | L4T ESI ALEXI |
| Water Use Efficiency | L4T WUE |
*Table 1. Listing of SBG tiled products long names and short names.*


### Cloud-Optimized GeoTIFF Orbit/Scene/Tile Products 

To provide an analysis-ready format, the SBG products are distributed in a tiled form and using the COG format. The tiled products include the letter T in their level identifiers: L1CT, L2T, L3T, and L4T. The tiling system used for SBG is borrowed from the modified Military Grid Reference System (MGRS) tiling scheme used by Sentinel 2. These tiles divide the Universal Transverse Mercator (UTM) zones into square tiles 109760 m across. SBG uses a 60 m cell size with 1800 rows by 1800 columns in each tile, totaling 3.24 million pixels per tile. This allows the end user to assume that each 60 m SBG pixel will remain in the same location at each timestep observed in analysis. The COG format also facilitates end-user analysis as a universally recognized and supported format, compatible with open-source software, including QGIS, ArcGIS, GDAL, the Raster package in R, `rioxarray` in Python, and `Rasters.jl` in Julia.

Each `float32` data layer occupies 4 bytes of storage per pixel, which amounts to an uncompressed size of 12.96 mb for each tiled data layer. The `uint8` quality flag layers occupy a single byte per pixel, which amounts to an uncompressed size of 3.24 mb per tiled data quality layer.

Each `.tif` COG data layer in each L2T/L3T/L4T product additionally contains a rendered browse image in GeoJPEG format with a `.jpeg` extension. This image format is universally recognized and supported, and these files are compatible with Google Earth. Each L2T/L3T/L4T tile granule includes a `.json` file containing the Product Metadata and Standard Metadata in JSON format.


### Quality Flags

Two high-level quality flags are provided in all gridded and tiled products as thematic/binary masks encoded to zero and one in unsigned 8-bit integer layers. The cloud layer represents the final cloud test from L2 CLOUD. The water layer represents the surface water body in the Shuttle Radar Topography Mission (SRTM) Digital Elevation Model. For both layers, zero means absence, and one means presence. Pixels with the value 1 in the cloud layer represent detection of cloud in that pixel. Pixels with the value 1 in the water layer represent open water surface in that pixel. All tiled product data layers written in `float32` contain a standard not-a-number (`NaN`) value at each pixel that could not be retrieved. The cloud and water layers are provided to explain these missing values.

### Product Availability

The SBG products are available at the NASA Land Processes Distribution Active Archive Center (LP-DAAC), https://earthdata.nasa.gov/ and can be accessed via the Earthdata search engine. 

## L2T STARS NDVI and Albedo Product

NDVI and albedo are estimated at 60 m SBG standard resolution for each daytime SBG overpass by fusing temporally sparse but fine spatial resolution images from the Harmonized Landsat Sentinel (HLS) 2.0 product with daily, moderate spatial resolution images from the Suomi NPP Visible Infrared Imaging Radiometer Suite (VIIRS) VNP09GA product. The data fusion is performed using a variant of the Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) algorithm developed by Dr. Margaret Johnson and Gregory Halverson at the Jet Propulsion Laboratory. STARS is a Bayesian timeseries methodology that provides streaming data fusion and uncertainty quantification through efficient Kalman filtering. 

Operationally, each L2T STARS tile run loads the means and covariances of the STARS model saved from the most recent tile run, then iteratively advances the means and covariances forward each day updating with fine imagery from HLS and/or moderate resolution imagery from VIIRS up to the day of the target SBG overpass. A pixelwise, lagged 16-day implementation of the VNP43 algorithm (Schaaf, 2017) is used for a near-real-time BRDF correction on the VNP09GA products to produce VIIRS NDVI and albedo. 

Operationally, each L2T STARS tile run loads the means and covariances of the STARS model saved from the most recent tile run, then iteratively advances the means and covariances forward each day updating with fine imagery from HLS and/or moderate resolution imagery from VIIRS up to the day of the target SBG overpass. A pixelwise, lagged 16-day implementation of the VNP43 algorithm (Schaaf, 2017) is used for a near-real-time BRDF correction on the VNP09GA products to produce VIIRS NDVI and albedo. The layers of the L2T STARS product are listed in Table 2. All layers of this product are represented by 32-bit floating point arrays. The NDVI estimates and 1σ uncertainties (-UQ) are unitless from -1 to 1. The albedo estimates and 1σ uncertainties (-UQ) are proportions from 0 to 1. 


| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** |**Scale Factor** | **Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| NDVI | Normalized Difference Vegetation Index | float32 | Index | NaN | N/A | -1 | 1 | N/A | 12.06 mb |
| NDVI-UQ | Normalized Difference Vegetation Index Uncertainty | float32 | Index | NaN | N/A | -1 | 1 | N/A | 12.06 mb |
| albedo | Albedo | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| albedo-UQ | Albedo Uncertainty | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
*Table 2. Listing of L2T STARS data layers.*


## L3T AUX Ecosystem Auxiliary Inputs Product

The SBG ecosystem processing chain is designed to be independently reproducible. To facilitate open science, the auxiliary data inputs that are produced for evapotranspiration processing are distributed as a data product, such that the end user has the ability to run their own evapotranspiration model using SBG data. The data layers of the L3T AUX product are described in Table 3.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Ta | Near-surface air temperature | float32 | Celsius | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| RH | Relative Humidity | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| SM | Soil Moisture | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| Rn | Net Radiation | float32 | Ratio | NaN | N/A | 0 | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
*Table 3. Listing of the L3T AUX data layers.*

## Downscaled Meteorology

Coarse resolution near-surface air temperature (Ta) and relative humidity (RH) are taken from the GEOS-5 FP `tavg1_2d_slv_Nx` product. Ta and RH are down-scaled using a linear regression between up-sampled ST, NDVI, and albedo as predictor variables to Ta or RH from GEOS-5 FP as a response variable, within each Sentinel tile. These regression coefficients are then applied to the 60 m ST, NDVI, and albedo, and this first-pass estimate is then bias-corrected to the coarse image from GEOS-5 FP. These downscaled meteorology estimates are recorded in the L3T AUX product listed in Table . Areas of cloud are filled in with bi-cubically resampled GEOS-5 FP.

## Downscaled Soil Moisture

This same down-scaling procedure is applied to soil moisture (SM) from the GEOS-5 FP `tavg1_2d_lnd_Nx` product, which is recorded in the L3T AUX product listed in Table .

## Surface Energy Balance

The surface energy balance processing for SBG begins with an artificial neural network (ANN) implementation of the Forest Light Environmental Simulator (FLiES) radiative transfer algorithm, following the workflow established by Dr. Hideki Kobayashi and Dr. Youngryel Ryu. GEOS-5 FP provides sub-daily Cloud Optical Thickness (COT) in the `tavg1_2d_rad_Nx` product and Aerosol Optical Thickness (AOT) from `tavg3_2d_aer_Nx`. Together with STARS albedo, these variables are run through the ANN implementation of FLiES to estimate incoming shortwave radiation (Rg), bias-corrected to Rg from the GEOS-5 FP `tavg1_2d_rad_Nx` product.

The Breathing Earth System Simulator (BESS) algorithm, contributed by Dr. Youngryel Ryu, iteratively calculates net radiation (Rn), ET, and Gross Primary Production (GPP) estimates. The BESS Rn is used as the Rn input to the remaining ET models and is recorded in the L3T AUX product listed in Table 3.


## L3T ET Evapotranspiration Product

Following design of the L3T JET product from ECOSTRESS Collection 2, the SBG L3T ET product uses an ensemble of evapotranspiration models to produce an evapotranspiration estimate.

The PT-JPL-SM model, developed by Dr. Adam Purdy and Dr. Joshua Fisher was designed as a SM-sensitive evapotranspiration product for the Soil Moisture Active-Passive (SMAP) mission, and then reimplemented as an ET model in the ECOSTRESS and SBG processing chain, using the downscaled soil moisture from the L3T AUX product. Similar to the PT-JPL model used in ECOSTRESS Collection 1, The PT-JPL-SM model estimates instantaneous canopy transpiration, leaf surface evaporation, and soil moisture evaporation using the Priestley-Taylor formula with a set of constraints. These three partitions are combined into total latent heat flux in watts per square meter for the ensemble estimate. 


The Surface Temperature Initiated Closure (STIC) model, contributed by Dr. Kaniska Mallick, was designed as a ST-sensitive ET model, adopted by ECOSTRESS and SBG for improved estimates of ET reflecting mid-day heat stress. The STIC model estimates total latent heat flux directly. This instantaneous estimate of latent heat flux is included in the ensemble estimate.

The MOD16 algorithm was designed as the ET product for the Moderate Resolution Imaging Spectroradiometer (MODIS) and then continued as a Visible Infrared Imaging Radiometer Suite (VIIRS) product. MOD16 uses a similar approach to PT-JPL and PT-JPL-SM to independently estimate vegetation and soil components of instantaneous ET, but using the Penman-Monteith formula instead of the Priestley-Taylor. The MOD16 latent heat flux partitions are summed to total latent heat flux for the ensemble estimate.

The BESS model is a coupled surface energy balance and photosynthesis model. The latent heat flux component of BESS is also included in the ensemble estimate.

The median of total latent heat flux in watts per square meter from the PT-JPL, STIC, MOD16, and BESS models is upscaled to a daily ET estimate in millimeters per day and recorded in the L3T ET product as `ETdaily`. The standard deviation between these multiple estimates of ET is considered the uncertainty for the SBG evapotranspiration product, as `ETinstUncertainty`. The layers for the L3T ET products are listed in Table 6 Note that the ETdaily product represents the integrated ET between sunrise and sunset.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ETdaily | Daily Evapotranspiration | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| ETdailyUncertainty | Daily Evapotranspiration Uncertainty | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
*Table 4. Listing of the L3T ET data layers.*


## L4T ESI and WUE Products

The PT-JPL-SM model generates estimates of both actual and potential instantaneous ET. The potential evapotranspiration (PET) estimate represents the maximum expected ET if there were no water stress to plants on the ground. The ratio of the actual ET estimate to the PET estimate forms an index representing the water stress of plants, with zero being fully stressed with no observable ET and one being non-stressed with ET reaching PET. These ESI and PET estimates are distributed in the L4T ESI product as listed in Table 5.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ESI | Evaporative Stress Index | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| PET | Potential Evapotranspiration | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
*Table 5. Listing of the L4T ESI data layers.*


The BESS GPP estimate represents the amount of carbon that plants are taking in. The transpiration component of PT-JPL-SM represents the amount of water that plants are releasing. The BESS GPP is divided by the PT-JPL-SM transpiration to estimate water use efficiency (WUE), the ratio of grams of carbon that plants take in to kilograms of water that plants release. These WUE and GPP estimates are distributed in the L4T WUE product as listed in Table 6.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| WUE | Water Use Efficiency | float32 | $$\text{g C kg}^{-1} \text{H}_2\text{O}$$ | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| GPP | Gross Primary Production | float32 | $$\mu\text{mol m}^{-2} \text{s}^{-1}$$ | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
*Table 6. Listing of the L3T WUE data layers.*


## L3T ETLL Low Latency Evapotranspiration Product

In addition to the standard product, there will also be a low latency (< 24 hour) ET product, produced with low latency L2 LSTE, and ancillary inputs (NDVI) from STARS from 3 days prior. The low latency ET product involves a daily ET estimate in millimeters per day, as listed in Table 7. 

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ETdaily | Evapotranspiration Daily | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.96 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
*Table 7. Listing of the L3T ETLL data layers.*

## Standard Metadata

Each SBG product bundle contains two sets of product metadata:
-   ProductMetadata
-   StandardMetadata

Each product contains a custom set of `ProductMetadata` attributes, as listed in Table 8. The `StandardMetadata` attributes are consistent across products at each orbit/scene, as listed in Table 9.
| **Name** | **Type** |
| --- | --- |
| AncillaryInputPointer | string |
| AutomaticQualityFlag | string |
| AutomaticQualityFlagExplanation | string |
| BuildID | string |
| CRS | string |
| CampaignShortName | string |
| CollectionLabel | string |
| DataFormatType | string |
| DayNightFlag | string |
| EastBoundingCoordinate | float |
| FieldOfViewObstruction | string |
| ImageLines | float |
| ImageLineSpacing | integer |
| ImagePixels | float |
| ImagePixelSpacing | integer |
| InputPointer | string |
| InstrumentShortName | string |
| LocalGranuleID | string |
| LongName | string |
| NorthBoundingCoordinate | float |
| PGEName | string |
| PGEVersion | string |
| PlatformLongName | string |
| PlatformShortName | string |
| PlatformType | string |
| ProcessingEnvironment | string |
| ProcessingLevelDescription | string |
| ProcessingLevelID | string |
| ProducerAgency | string |
| ProducerInstitution | string |
| ProductionDateTime | string |
| ProductionLocation | string |
| RangeBeginningDate | string |
| RangeBeginningTime | string |
| RangeEndingDate | string |
| RangeEndingTime | string |
| RegionID | string |
| SISName | string |
| SISVersion | string |
| SceneBoundaryLatLonWKT | string |
| SceneID | string |
| ShortName | string |
| SouthBoundingCoordinate | float |
| StartOrbitNumber | string |
| StopOrbitNumber | string |
| WestBoundingCoordinate | float |
*Table 8. Name and type of metadata fields contained in the common StandardMetadata group in each L2T/L3T/L4T product.*

| **Name** | **Type** |
| --- | --- |
| BandSpecification | float |
| NumberOfBands | integer |
| OrbitCorrectionPerformed | string |
| QAPercentCloudCover | float |
| QAPercentGoodQuality | float |
| AuxiliaryNWP | string |
*Table 9. Name and type of metadata fields contained in the common ProductMetadata group in each L2T/L3T/L4T product.*

## Acknowledgements 

We would like to thank Joshua Fisher as the initial science lead of the SBG mission and PI of the ROSES project to re-design the SBG products.

We would like to thank Adam Purdy for contributing the PT-JPL-SM model.

We would like to thank Kaniska Mallick for contributing the STIC model.

We would like to thank Martha Anderson for contributing the DisALEXI-JPL algorithm.

##  Bibliography 

Schaaf, C. (2017). *VIIRS BRDF, Albedo, and NBAR Product Algorithm Theoretical Basis Document (ATBD).* NASA Goddard Space Flight Center.
