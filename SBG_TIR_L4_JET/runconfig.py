import logging
from os.path import exists

import untangle

from .exit_codes import UnableToOpenRunConfig, UnableToParseRunConfig, ECOSTRESSExitCodeException

logger = logging.getLogger(__name__)

__author__ = 'Gregory Halverson'


def parse_scalar(scalar):
    return scalar.cdata


def parse_vector(vector):
    data = [
        element.cdata
        for element
        in vector.children
    ]

    return data


def parse_group(group):
    result = {}

    if hasattr(group, 'scalar'):
        for element in group.scalar:
            result[element['name']] = parse_scalar(element)

    if hasattr(group, 'vector'):
        for element in group.vector:
            result[element['name']] = parse_vector(element)

    if hasattr(group, 'group'):
        for element in group.group:
            result[element['name']] = parse_group(element)

    return result


def parse_runconfig(text: str) -> dict:
    tree = untangle.parse(text)
    config = parse_group(tree.input)

    return config


def read_runconfig(filename: str) -> dict:
    if not exists(filename):
        raise UnableToOpenRunConfig(f"run-config file does not exist: {filename}")

    try:
        tree = untangle.parse(filename)
        config = parse_group(tree.input)
    except ECOSTRESSExitCodeException as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise UnableToParseRunConfig(f"unable to parse run-config file: {filename}")

    return config


class ECOSTRESSRunConfig:
    def read_runconfig(self, filename: str) -> dict:
        return read_runconfig(filename)
