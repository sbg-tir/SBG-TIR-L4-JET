# %% [markdown]
# Using `ECOv002-CMR` package to retrieve ECOSTRESS granules as inputs using the Common Metadata Repository (CMR) API. Using `ECOv002-L3T-L4T-JET` package to run the product generating executable (PGE).

# %%
import numpy as np
from ECOv002_CMR import download_ECOSTRESS_granule
from SBG_TIR_L4_JET import generate_L3T_L4T_JET_runconfig, L3T_L4T_JET

# %% [markdown]
# Disable logger output in notebook

# %%
import logging

logging.getLogger().handlers = []

# %% [markdown]
# Set working directory

# %%
working_directory = "~/data/ECOSTRESS_example"
static_directory = "~/data/L3T_L4T_static"

# %% [markdown]
# Retrieve LST LSTE granule from CMR API for target date

# %%
L2T_LSTE_granule = download_ECOSTRESS_granule(
    product="L2T_LSTE", 
    orbit=35698,
    scene=14,
    tile="11SPS", 
    aquisition_date="2024-10-22",
    parent_directory=working_directory
)

L2T_LSTE_granule

# %% [markdown]
# Load and display preview of surface temperature

# %%
L2T_LSTE_granule.ST_C

# %% [markdown]
# Retrieve L2T STARS granule from CMR API as prior

# %%
L2T_STARS_granule = download_ECOSTRESS_granule(
    product="L2T_STARS", 
    tile="11SPS", 
    aquisition_date="2024-10-22",
    parent_directory=working_directory
)

L2T_STARS_granule

# %% [markdown]
# Load and display preview of vegetation index

# %%
L2T_STARS_granule.NDVI

# %% [markdown]
# Generate XML run-config file for L3T L4T JET PGE run

# %%
runconfig_filename = generate_L3T_L4T_JET_runconfig(
    L2T_LSTE_filename=L2T_LSTE_granule.product_filename,
    L2T_STARS_filename=L2T_STARS_granule.product_filename,
    working_directory=working_directory,
    static_directory=static_directory
)

runconfig_filename

# %%
with open(runconfig_filename, "r") as f:
    print(f.read())

# %%
exit_code = L3T_L4T_JET(runconfig_filename=runconfig_filename)
exit_code

# %%



