from sys import argv, exit
import logging
import yaml
import colored_logging as cl

logger = logging.getLogger(__name__)

def SBGv001_L4T_JET(runconfig_filename = argv[1]) -> int:
    """
    entry point for SBG collection 1 level 4 evapotranspiration PGE
    """
    with open(runconfig_filename, "r") as file:
        runconfig_dict = yaml.safe_load(file)
    
    logger.info(f"run-config file: {runconfig_filename}")
    logger.info(f"run-config:")
    logger.info(yaml.dump(runconfig_dict))

    exit_code = 0

    return exit_code

if __name__ == "__main__":
    exit(SBGv001_L4T_JET(runconfig_filename=argv[1]))
