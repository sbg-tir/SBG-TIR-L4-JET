# SBG-TIR OTTER L3T ET, L4T ESI, and L4T WUE Data Products

This is the main repository for the Suface Biology and Geology Thermal Infrared (SBG-TIR) level 3 & 4 evapotranspiration data product generation software. 

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Kerry Cawse-Nicholson](https://github.com/kcawse) (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

[Madeleine Pascolini-Campbell](https://github.com/madeleinepc) (she/her)<br>
[madeleine.a.pascolini-campbell@jpl.nasa.gov](mailto:madeleine.a.pascolini-campbell@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329F

[Claire Villanueva-Weeks](https://github.com/clairesvw) (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## 1. Introduction 

This software will produce estimates of:
- evapotranspiration (ET)
- evaporative stress index (ESI)
- water use efficiency (WUE)

Evapotranspiration (ET) is one of the main science outputs from the Surface Biology and Geology (SBG) Mission. ET is a Level-3 (L3) product constructed from a combination of the SBG Level-2 (L2) Land Surface Temperature (LST) product and auxiliary data sources. Accurate modelling of ET requires consideration of many environmental and biological controls including: incoming radiation, the atmospheric water vapor deficit, soil water availability, vegetation physiology and phenology (Brutsaert, 1982; Monteith, 1965; Penman, 1948). Scientists develop models that ingest global satellite observations to capture these environmental and biological controls on ET. LST holds the unique ability to capture when and where plants experience stress, as observed by elevated temperatures which can idenitfy areas that have a reduced capacity to evaporate or transpire water to the atmosphere (Allen et al., 2007). The SBG evapotranspiration product combines the [surface temperature and emissivity observations from the OTTER sensor](https://github.com/sbg-tir/SBG-TIR-L2-LSTE) with the [NDVI and albedo estimated by STARS](https://github.com/sbg-tir/SBG-TIR-L2-STARS), estimates near-surface meteorology by downscaling GEOS-5 FP to these three high resolution images, and runs these variables through a set of surface energy balance models.

The repositories for the evapotranspiration algorithms are located in the [JPL-Evapotranspiration-Algorithms](https://github.com/JPL-Evapotranspiration-Algorithms) organization.

## 2. Data Products 

### 2.1. Metadata

SBG-TIR standards incorporate additional metadata that describe each GeoTIFF Dataset within the GeoTIFF file. Each of these metadata elements appear in an GeoTIFF Attribute that is directly associated with the GeoTIFF Dataset. Wherever possible, these GeoTIFF Attributes employ names that conform to the Climate and Forecast (CF) conventions. 

Each SBG product bundle contains two sets of product metadata:
-   ProductMetadata
-   StandardMetadata

#### 2.1.1. Standard Metadata
Information on the `StandardMetadata` is included on the [SBG-TIR github landing page](https://github.com/sbg-tir)

#### 2.1.2. Product Metadata

| **Name** | **Type** |
| --- | --- |
| BandSpecification | float |
| NumberOfBands | integer |
| OrbitCorrectionPerformed | string |
| QAPercentCloudCover | float |
| QAPercentGoodQuality | float |
| AuxiliaryNWP | string |

*Table 9. Name and type of metadata fields contained in the common ProductMetadata group in each L2T/L3T/L4T product.*

| **Product Long Name** | **Product Short Name** |
| --- | --- |
| STARS NDVI/Albedo | L2T STARS |
| Ecosystem Auxiliary Inputs | L3T AUX |
| Evapotranspiration | L3T ET |
| Evaporative Stress Index | L4T ESI |
| Water Use Efficiency | L4T WUE |

*Table 1. Listing of SBG ecosystem products long names and short names.*


### 2.2. Quality Flags

Two high-level quality flags are provided in all gridded and tiled products as thematic/binary masks encoded to zero and one in unsigned 8-bit integer layers. The cloud layer represents the final cloud test from L2 CLOUD. The water layer represents the surface water body in the Shuttle Radar Topography Mission (SRTM) Digital Elevation Model. For both layers, zero means absence, and one means presence. Pixels with the value 1 in the cloud layer represent detection of cloud in that pixel. Pixels with the value 1 in the water layer represent open water surface in that pixel. All tiled product data layers written in `float32` contain a standard not-a-number (`NaN`) value at each pixel that could not be retrieved. The cloud and water layers are provided to explain these missing values.
  
### 2.3. L2T STARS NDVI and Albedo Product
The STARS data product is produced with a separate Product Generating Executable (PGE) [SBG-TIR-L2-STARS](https://github.com/sbg-tir/SBG-TIR-L2-STARS).

### 2.4. L3T AUX Ecosystem Auxiliary Inputs Product

The SBG ecosystem processing chain is designed to be independently reproducible. To facilitate open science, the auxiliary data inputs that are produced for evapotranspiration processing are distributed as a data product, such that the end user has the ability to run their own evapotranspiration model using SBG data. The data layers of the L3T AUX product are described in Table 3.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Ta | Near-surface air temperature | float32 | Celsius | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| RH | Relative Humidity | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| SM | Soil Moisture | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| Rn | Net Radiation | float32 | Ratio | NaN | N/A | 0 | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 2. Listing of the L3T AUX data layers.*

### 2.5. Downscaled Meteorology & Soil Moisture

```mermaid
flowchart TB
    subgraph SBG_L2[SBG-TIR OTTER L2]
        direction TB
        SBG_L2T_STARS[SBG-TIR<br>OTTER<br>L2T_STARS<br>NDVI<br>&<br>Albedo<br>Product]
        SBG_L2T_LSTE[SBG-TIR<br>OTTER<br>L2T_LSTE<br>Surface Temperature<br>&<br>Emissivity<br>Product]
        ST[Surface Temperature 60m]
        NDVI[NDVI 60m]
        albedo[Albedo 60m]
        SBG_L2T_LSTE --> ST
        SBG_L2T_STARS --> NDVI
        SBG_L2T_STARS --> albedo
    end

    subgraph GEOS5FP[GEOS-5 FP]
        direction TB
        GEOS5FP_Ta[GEOS-5 FP<br>Air<br>Temperature]
        GEOS5FP_RH[GEOS-5 FP<br>Humidity]
        GEOS5FP_SM[GEOS-5 FP<br>Soil<br>Moisture]
    end

    subgraph downscaling[Downscaling]
        direction TB
        downscale_Ta[Air<br>Temperature<br>Downscaling]
        downscale_RH[Humidity<br>Downscaling]
        downscale_SM[Soil<br>Moisture<br>Downscaling]
    end

    subgraph downscaled_meteorology[Downscaled Meteorology]
        direction TB
        downscaled_Ta[Downscaled<br>60m<br>Air<br>Temperature]
        downscaled_RH[Downscaled<br>60m<br>Humidity]
        downscaled_SM[Downscaled<br>60m<br>Soil<br>Moisture]
    end

    GEOS5FP_Ta --> downscale_Ta
    ST --> downscale_Ta
    NDVI --> downscale_Ta
    albedo --> downscale_Ta

    GEOS5FP_RH --> downscale_RH
    ST --> downscale_RH
    NDVI --> downscale_RH
    albedo --> downscale_RH

    GEOS5FP_SM --> downscale_SM
    ST --> downscale_SM
    NDVI --> downscale_SM
    albedo --> downscale_SM

    downscale_Ta --> downscaled_Ta
    downscale_RH --> downscaled_RH
    downscale_SM --> downscaled_SM
```

Coarse resolution near-surface air temperature (Ta) and relative humidity (RH) are taken from the GEOS-5 FP `tavg1_2d_slv_Nx` product. Ta and RH are down-scaled using a linear regression between up-sampled ST, NDVI, and albedo as predictor variables to Ta or RH from GEOS-5 FP as a response variable, within each Sentinel tile. These regression coefficients are then applied to the 60 m ST, NDVI, and albedo, and this first-pass estimate is then bias-corrected to the coarse image from GEOS-5 FP. These downscaled meteorology estimates are recorded in the L3T AUX product listed in Table . Areas of cloud are filled in with bi-cubically resampled GEOS-5 FP. This same down-scaling procedure is applied to soil moisture (SM) from the GEOS-5 FP `tavg1_2d_lnd_Nx` product, which is recorded in the L3T AUX product listed in Table .

### 2.6. Surface Energy Balance

```mermaid
flowchart TB
    subgraph SBG_L2[SBG-TIR OTTER L2]
        direction TB
        SBG_L2T_STARS[SBG-TIR<br>OTTER<br>L2T_STARS<br>NDVI<br>&<br>Albedo<br>Product]
        SBG_L2T_LSTE[SBG-TIR<br>OTTER<br>L2T_LSTE<br>Surface Temperature<br>&<br>Emissivity<br>Product]
        ST[Surface Temperature 60m]
        NDVI[NDVI 60m]
        albedo[Albedo 60m]
        SBG_L2T_LSTE --> ST
        SBG_L2T_STARS --> NDVI
        SBG_L2T_STARS --> albedo
    end

    subgraph downscaled_meteorology[Downscaled Meteorology]
        direction TB
        downscaled_Ta[Downscaled<br>60m<br>Air<br>Temperature]
        downscaled_RH[Downscaled<br>60m<br>Humidity]
        downscaled_SM[Downscaled<br>60m<br>Soil<br>Moisture]
    end

    subgraph GEOS5FP[GEOS-5 FP]
        direction TB
        GEOS5FP_AOT[GEOS-5 FP AOT]
        GEOS5FP_COT[GEOS-5 FP COT]
    end

    BESS_Rn[BESS<br>60m<br>Net<br>Radiation]
    BESS_GPP[BESS<br>60m<br>GPP]
    BESS_ET[BESS<br>60m<br>ET]

    GEOS5FP_AOT --> FLiES
    GEOS5FP_COT --> FLiES
    albedo --> FLiES
    
    FLiES --> BESS
    ST --> BESS
    NDVI --> BESS
    albedo --> BESS
    downscaled_Ta --> BESS
    downscaled_RH --> BESS
    downscaled_SM --> BESS

    BESS --> BESS_Rn
    BESS --> BESS_GPP
    BESS --> BESS_ET
```

The surface energy balance processing for SBG begins with an artificial neural network (ANN) implementation of the Forest Light Environmental Simulator (FLiES) radiative transfer algorithm, following the workflow established by Dr. Hideki Kobayashi and Dr. Youngryel Ryu. GEOS-5 FP provides sub-daily Cloud Optical Thickness (COT) in the `tavg1_2d_rad_Nx` product and Aerosol Optical Thickness (AOT) from `tavg3_2d_aer_Nx`. Together with STARS albedo, these variables are run through the ANN implementation of FLiES to estimate incoming shortwave radiation (Rg), bias-corrected to Rg from the GEOS-5 FP `tavg1_2d_rad_Nx` product.

The Breathing Earth System Simulator (BESS) algorithm, contributed by Dr. Youngryel Ryu, iteratively calculates net radiation (Rn), ET, and Gross Primary Production (GPP) estimates. The BESS Rn is used as the Rn input to the remaining ET models and is recorded in the L3T AUX product listed in Table 3.


### 2.7. L3T ET Evapotranspiration Product

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

*Table 3. Listing of the L3T ET data layers.*

### 2.8. L4T ESI and WUE Products

The PT-JPL-SM model generates estimates of both actual and potential instantaneous ET. The potential evapotranspiration (PET) estimate represents the maximum expected ET if there were no water stress to plants on the ground. The ratio of the actual ET estimate to the PET estimate forms an index representing the water stress of plants, with zero being fully stressed with no observable ET and one being non-stressed with ET reaching PET. These ESI and PET estimates are distributed in the L4T ESI product as listed in Table 5.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ESI | Evaporative Stress Index | float32 | Ratio | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| PET | Potential Evapotranspiration | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 4. Listing of the L4T ESI data layers.*

The BESS GPP estimate represents the amount of carbon that plants are taking in. The transpiration component of PT-JPL-SM represents the amount of water that plants are releasing. The BESS GPP is divided by the PT-JPL-SM transpiration to estimate water use efficiency (WUE), the ratio of grams of carbon that plants take in to kilograms of water that plants release. These WUE and GPP estimates are distributed in the L4T WUE product as listed in Table 6.

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| WUE | Water Use Efficiency | float32 | $$\text{g C kg}^{-1} \text{H}_2\text{O}$$ | NaN | N/A | 0 | 1 | N/A | 12.06 mb |
| GPP | Gross Primary Production | float32 | $$\mu\text{mol m}^{-2} \text{s}^{-1}$$ | NaN | N/A | N/A | N/A | N/A | 12.06 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

*Table 5. Listing of the L3T WUE data layers.*

### 2.9. L3T ETLL Low Latency Evapotranspiration Product

In addition to the standard product, there will also be a low latency (< 24 hour) ET product, produced with low latency L2 LSTE, and ancillary inputs (NDVI) from STARS from 3 days prior. The low latency ET product involves a daily ET estimate in millimeters per day, as listed in Table 7. 

| **Name** | **Description** | **Type** | **Units** | **Fill Value** | **No Data Value** | **Valid Min** | **Valid Max** | **Scale Factor** |**Size** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ETdaily | Evapotranspiration Daily | float32 | mm/day | NaN | N/A | N/A | N/A | N/A | 12.96 mb |
| cloud | Cloud mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |
| water | Water mask | float32 | Mask | 255 | N/A | 0 | 1 | N/A | 3.24 mb |

## 3. Theory

The JPL EvapoTranspiration (ET) data ensemble provides a robust estimation of ET from multiple ET models. The ET ensemble incorporates ET data from four algorithms: Priestley Taylor-Jet Propulsion Laboratory model with soil moisture (PT-JPLSM), the Penman Monteith MODIS Global Evapotranspiration Model (MOD16), Soil Temperature Initiated Closure (STIC) model, and the Breathing Earth System Simulator (BESS) model. We present descriptions of these models here, inherited from the ECOSTRESS mission, as candidates for SBG L3 evapotranspiration processing.

## 4. Uncertainty Analysis


## 5. Cal/Val


#### Acknowledgements 

We would like to thank Joshua Fisher as the initial science lead of the SBG mission and PI of the ROSES project to re-design the SBG products.

We would like to thank Adam Purdy for contributing the PT-JPL-SM model.

We would like to thank Kaniska Mallick for contributing the STIC model.

We would like to thank Martha Anderson for contributing the DisALEXI-JPL algorithm.

####  Bibliography 

Schaaf, C. (2017). *VIIRS BRDF, Albedo, and NBAR Product Algorithm Theoretical Basis Document (ATBD).* NASA Goddard Space Flight Center.
