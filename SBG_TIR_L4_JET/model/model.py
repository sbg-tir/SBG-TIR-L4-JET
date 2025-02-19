import logging
from os.path import abspath, expanduser, join, exists
from datetime import date
from typing import Union

import numpy as np
from rasters import Raster

import colored_logging as cl


DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_INTERMEDIATE = "intermediate"
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_INCLUDE_PREVIEW = True
DEFAULT_RESAMPLING = "cubic"
DEFAULT_SAVE_INTERMEDIATE = True
DEFAULT_SHOW_DISTRIBUTION = True

logger = logging.getLogger(__name__)

class BlankOutputError(Exception):
    pass

def check_distribution(
        image: Raster,
        variable: str,
        date_UTC: date or str,
        target: str):
    unique = np.unique(image)
    nan_proportion = np.count_nonzero(np.isnan(image)) / np.size(image)

    if len(unique) < 10:
        logger.info(f"variable {cl.name(variable)} ({image.dtype}) on {cl.time(f'{date_UTC:%Y-%m-%d}')} at {cl.place(target)} with {cl.val(unique)} unique values")

        for value in unique:
            if np.isnan(value):
                count = np.count_nonzero(np.isnan(image))
            else:
                count = np.count_nonzero(image == value)

            if value == 0 or np.isnan(value):
                logger.info(f"* {cl.colored(value, 'red')}: {cl.colored(count, 'red')}")
            else:
                logger.info(f"* {cl.val(value)}: {cl.val(count)}")
    else:
        minimum = np.nanmin(image)

        if minimum < 0:
            minimum_string = cl.colored(f"{minimum:0.3f}", "red")
        else:
            minimum_string = cl.val(f"{minimum:0.3f}")

        maximum = np.nanmax(image)

        if maximum <= 0:
            maximum_string = cl.colored(f"{maximum:0.3f}", "red")
        else:
            maximum_string = cl.val(f"{maximum:0.3f}")

        if nan_proportion > 0.5:
            nan_proportion_string = cl.colored(f"{(nan_proportion * 100):0.2f}%", "yellow")
        elif nan_proportion == 1:
            nan_proportion_string = cl.colored(f"{(nan_proportion * 100):0.2f}%", "red")
        else:
            nan_proportion_string = cl.val(f"{(nan_proportion * 100):0.2f}%")

        message = "variable " + cl.name(variable) + \
            " on " + cl.time(f"{date_UTC:%Y-%m-%d}") + \
            " at " + cl.place(target) + \
            " min: " + minimum_string + \
            " mean: " + cl.val(f"{np.nanmean(image):0.3f}") + \
            " max: " + maximum_string + \
            " nan: " + nan_proportion_string + f" ({cl.val(image.nodata)})"

        if np.all(image == 0):
            message += " all zeros"
            logger.warning(message)
        else:
            logger.info(message)

    if nan_proportion == 1:
        raise BlankOutputError(f"variable {variable} on {date_UTC:%Y-%m-%d} at {target} is a blank image")

class Model:
    def __init__(
            self,
            working_directory: str = None,
            static_directory: str = None,
            intermediate_directory: str = None,
            preview_quality: int = DEFAULT_PREVIEW_QUALITY,
            resampling: str = DEFAULT_RESAMPLING,
            save_intermediate: bool = DEFAULT_SAVE_INTERMEDIATE,
            show_distribution: bool = DEFAULT_SHOW_DISTRIBUTION,
            include_preview: bool = DEFAULT_INCLUDE_PREVIEW):

        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        if static_directory is None:
            static_directory = working_directory

        static_directory = abspath(expanduser(static_directory))

        logger.info(f"working directory: {cl.dir(working_directory)}")

        if intermediate_directory is None:
            intermediate_directory = join(working_directory, DEFAULT_INTERMEDIATE)

        intermediate_directory = abspath(expanduser(intermediate_directory))

        logger.info(f"intermediate directory: {cl.dir(intermediate_directory)}")

        self.working_directory = working_directory
        self.static_directory = static_directory
        self.intermediate_directory = intermediate_directory
        self.preview_quality = preview_quality
        self.resampling = resampling
        self.show_distribution = show_distribution
        self.save_intermediate = save_intermediate
        self.include_preview = include_preview

    def intermediate_filename(
            self,
            variable_name: str,
            acquisition_date: date or str,
            target_name: str) -> str:
        filename = join(
            self.intermediate_directory,
            f"{acquisition_date:%Y.%m.%d}",
            f"{acquisition_date:%Y.%m.%d}_{target_name}_{variable_name}.tif"
        )

        return filename

    def intermediate_available(
            self,
            variable_name: str,
            date_UTC: date or str,
            target: str) -> bool:
        filename = self.intermediate_filename(
            variable_name=variable_name,
            acquisition_date=date_UTC,
            target_name=target
        )

        available = exists(filename)

        if available:
            logger.info(f"{variable_name} cache found: {filename}")

        return available

    def check_distribution(
            self,
            image: Raster,
            variable: str,
            date_UTC: date or str,
            target: str,
            blank_OK: bool = False):
        if self.show_distribution:
            unique = np.unique(image)
            nan_proportion = np.count_nonzero(np.isnan(image)) / np.size(image)

            if len(unique) < 10:
                logger.info(
                    "variable " + cl.name(variable) + " on " + cl.time(f"{date_UTC:%Y-%m-%d}") + " at " + cl.place(
                        target))

                for value in unique:
                    count = np.count_nonzero(image == value)

                    if value == 0:
                        logger.info(f"* {cl.colored(value, 'red')}: {cl.colored(count, 'red')}")
                    else:
                        logger.info(f"* {cl.val(value)}: {cl.val(count)}")
            else:
                minimum = np.nanmin(image)

                if minimum < 0:
                    minimum_string = cl.colored(f"{minimum:0.3f}", "red")
                else:
                    minimum_string = cl.val(f"{minimum:0.3f}")

                maximum = np.nanmax(image)

                if maximum <= 0:
                    maximum_string = cl.colored(f"{maximum:0.3f}", "red")
                else:
                    maximum_string = cl.val(f"{maximum:0.3f}")

                if nan_proportion > 0.5:
                    nan_proportion_string = cl.colored(f"{(nan_proportion * 100):0.2f}%", "yellow")
                elif nan_proportion == 1:
                    nan_proportion_string = cl.colored(f"{(nan_proportion * 100):0.2f}%", "red")
                else:
                    nan_proportion_string = cl.val(f"{(nan_proportion * 100):0.2f}%")

                message = "variable " + cl.name(variable) + \
                    " on " + cl.time(f"{date_UTC:%Y-%m-%d}") + \
                    " at " + cl.place(target) + \
                    " min: " + minimum_string + \
                    " mean: " + cl.val(f"{np.nanmean(image):0.3f}") + \
                    " max: " + maximum_string + \
                    " nan: " + nan_proportion_string + f" ({cl.val(image.nodata)})"

                if np.all(image == 0):
                    message += " all zeros"
                    logger.warning(message)
                else:
                    logger.info(message)

            if nan_proportion == 1 and not blank_OK:
                raise BlankOutputError(f"variable {variable} on {date_UTC:%Y-%m-%d} at {target} is a blank image")

    def load_intermediate(
            self,
            variable_name: str,
            date_UTC: date or str,
            target: str) -> Raster or None:
        filename = self.intermediate_filename(
            variable_name=variable_name,
            acquisition_date=date_UTC,
            target_name=target
        )

        if exists(filename):
            logger.info(f"loading {cl.name(variable_name)} cache: {cl.file(filename)}")
            image = Raster.open(filename)
            return image
        else:
            return None

    def write_intermediate(
            self,
            image: Raster,
            variable: str,
            date_UTC: date or str,
            target: str) -> Union[str, None]:
        if not self.save_intermediate:
            return None

        filename = self.intermediate_filename(
            variable_name=variable,
            acquisition_date=date_UTC,
            target_name=target
        )

        logger.info(f"writing {cl.name(variable)} intermediate: {cl.file(filename)}")

        image.to_geotiff(filename, include_preview=True, preview_quality=self.preview_quality)

        return filename

    def diagnostic(
            self,
            image: Raster,
            variable: str,
            date_UTC: date or str,
            target: str,
            blank_OK: bool = False):
        self.check_distribution(image=image, variable=variable, date_UTC=date_UTC, target=target, blank_OK=blank_OK)
        self.write_intermediate(image=image, variable=variable, date_UTC=date_UTC, target=target)
