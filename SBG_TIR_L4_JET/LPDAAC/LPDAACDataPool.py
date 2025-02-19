import netrc
import hashlib
import logging
import os
import posixpath
import re
import shutil
import urllib
from datetime import date
from fnmatch import fnmatch
from http.cookiejar import CookieJar
from os import makedirs, remove
from os.path import dirname
from os.path import exists
from os.path import getsize
from os.path import isdir
from os.path import join
from time import sleep
from typing import List, OrderedDict

import requests
import xmltodict
from bs4 import BeautifulSoup
from dateutil import parser
from pycksum import cksum

import colored_logging as cl
from ..exit_codes import DownloadFailed

CONNECTION_CLOSE = {
    "Connection": "close",
}

DEFAULT_REMOTE = "https://e4ftl01.cr.usgs.gov"
RETRIES = 6
WAIT_SECONDS = 360
XML_RETRIES = RETRIES
XML_TIMEOUT = WAIT_SECONDS
DOWNLOAD_RETRIES = RETRIES
DOWNLOAD_WAIT_SECONDS = WAIT_SECONDS

__author__ = "Gregory Halverson"

logger = logging.getLogger(__name__)


class LPDAACServerUnreachable(ConnectionError):
    pass


class LPDAACDataPool:
    logger = logging.getLogger(__name__)
    DEFAULT_CHUNK_SIZE = 2 ** 20
    DATE_REGEX = re.compile('^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$')
    DEFAULT_REMOTE = DEFAULT_REMOTE

    def __init__(self, username: str = None, password: str = None, remote: str = None, offline_ok: bool = False):
        if remote is None:
            remote = DEFAULT_REMOTE

        if username is None or password is None:
            try:
                netrc_file = netrc.netrc()
                username, _, password = netrc_file.authenticators("urs.earthdata.nasa.gov")
            except Exception as e:
                logger.warning("netrc credentials not found for urs.earthdata.nasa.gov")

        if username is None or password is None:
            if not "LPDAAC_USERNAME" in os.environ or not "LPDAAC_PASSWORD" in os.environ:
                raise RuntimeError("Missing environment variable 'LPDAAC_USERNAME' or 'LPDAAC_PASSWORD'")

            username = os.environ["LPDAAC_USERNAME"]
            password = os.environ["LPDAAC_PASSWORD"]

        self._remote = remote
        self._username = username
        self._password = password
        self.offline_ok = offline_ok

        # if self.offline_ok:
        #     logger.warning("going into offline mode")

        self._listings = {}

        try:
            self._authenticate()
            self._check_remote()
        except Exception as e:
            if self.offline_ok:
                logger.warning("unable to connect to LP-DAAC data pool")
            else:
                raise e

    def _authenticate(self):
        try:
            # https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python

            password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()

            password_manager.add_password(
                realm=None,
                uri="https://urs.earthdata.nasa.gov",
                user=self._username,
                passwd=self._password
            )

            cookie_jar = CookieJar()

            # Install all the handlers.

            opener = urllib.request.build_opener(
                urllib.request.HTTPBasicAuthHandler(password_manager),
                # urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
                # urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
                urllib.request.HTTPCookieProcessor(cookie_jar)
            )

            urllib.request.install_opener(opener)
        except Exception as e:
            message = "unable to authenticate with LP-DAAC data pool"
            if self.offline_ok:
                logger.warning(message)
            else:
                raise ConnectionError(message)

    def _check_remote(self):
        logger.info(f"checking URL: {cl.URL(self.remote)}")

        try:
            response = requests.head(self.remote, headers=CONNECTION_CLOSE)
            status = response.status_code
            duration = response.elapsed.total_seconds()
        except Exception as e:
            logger.exception(e)
            message = f"unable to connect to URL: {self.remote}"

            if self.offline_ok:
                logger.warning(message)
                return
            else:
                raise LPDAACServerUnreachable(message)

        if status == 200:
            logger.info(
                "remote verified with status " + cl.val(200) +
                " in " + cl.time(f"{duration:0.2f}") +
                " seconds: " + cl.URL(self.remote))
        else:
            message = f"status: {status} URL: {self.remote}"

            if self.offline_ok:
                logger.warning(message)
            else:
                raise ConnectionError(message)

    @property
    def remote(self):
        return self._remote

    def get_HTTP_text(self, URL: str) -> str:
        try:
            request = urllib.request.Request(URL)
            response = urllib.request.urlopen(request)
            body = response.read().decode()
        except Exception as e:
            logger.exception(e)
            raise ConnectionError(f"cannot connect to URL: {URL}")

        return body

    def get_HTTP_listing(self, URL: str, pattern: str = None) -> List[str]:
        if URL in self._listings:
            listing = self._listings[URL]
        else:
            text = self.get_HTTP_text(URL)
            soup = BeautifulSoup(text, 'html.parser')
            links = list(soup.find_all('a', href=True))

            # get directory names from links on http site
            listing = sorted([link['href'].replace('/', '') for link in links])
            self._listings[URL] = listing

        if pattern is not None:
            listing = sorted([
                item
                for item
                in listing
                if fnmatch(item, pattern)
            ])

        return listing

    def get_HTTP_date_listing(self, URL: str) -> List[date]:
        return sorted([
            parser.parse(item).date()
            for item
            in self.get_HTTP_listing(URL)
            if self.DATE_REGEX.match(item)
        ])

    def read_HTTP_XML(self, URL: str) -> OrderedDict:
        return xmltodict.parse(self.get_HTTP_text(URL))

    def generate_XML_URL(self, URL: str) -> str:
        return f"{URL}.xml"

    def get_metadata(self, data_URL: str) -> OrderedDict:
        metadata_URL = f"{data_URL}.xml"
        logger.info(f"checking metadata: {cl.URL(metadata_URL)}")
        request = urllib.request.Request(metadata_URL)
        response = urllib.request.urlopen(request)
        duration = response.elapsed.total_seconds()
        body = response.read().decode()
        metadata = xmltodict.parse(body)
        logger.info(f"metadata retrieved in {cl.val(f'{duration:0.2f}')} seconds: {cl.URL(metadata_URL)}")

        return metadata

    def get_remote_checksum(self, URL: str) -> int:
        return int(self.get_metadata(URL)["GranuleMetaDataFile"]["GranuleURMetaData"]["DataFiles"]["DataFileContainer"][
                       "Checksum"])

    def get_remote_filesize(self, URL: str) -> int:
        return int(self.get_metadata(URL)["GranuleMetaDataFile"]["GranuleURMetaData"]["DataFiles"]["DataFileContainer"][
                       "FileSize"])

    def get_local_checksum(self, filename: str, checksum_type: str = "CKSUM") -> str:
        with open(filename, "rb") as file:
            if checksum_type == "CKSUM":
                return str(int(cksum(file)))
            elif checksum_type == "MD5":
                return str(hashlib.md5(file.read()).hexdigest())

    def get_local_filesize(self, filename: str) -> int:
        return getsize(filename)

    def product_directory(self, platform: str, product: str, build: str = None) -> str:
        if build is None:
            build = "001"
        elif isinstance(build, float):
            build = f"{int(build * 10)}:03d"
        elif isinstance(build, int):
            build = f"{build:03d}"

        URL = posixpath.join(self._remote, platform, f"{product}.{build}")

        return URL

    def dates(self, platform: str, product: str, build: str = None) -> List[date]:
        return self.get_HTTP_date_listing(self.product_directory(platform, product, build))

    def date_URL(
            self,
            platform: str,
            product: str,
            acquisition_date: date or str,
            build: str = None) -> str:
        if isinstance(acquisition_date, str):
            acquisition_date = parser.parse(acquisition_date).date()

        URL = posixpath.join(
            self.product_directory(platform, product, build),
            f"{acquisition_date:%Y.%m.%d}"
        )

        return URL

    def files(
            self,
            platform: str,
            product: str,
            acquisition_date: date or str,
            build: str = None,
            pattern: str = None) -> List[str]:
        URL = self.date_URL(platform, product, acquisition_date, build)
        listing = self.get_HTTP_listing(URL, pattern)

        return listing

    def download_URL(
            self,
            URL: str,
            download_location: str = None,
            XML_retries: int = None,
            XML_timeout_seconds: int = None,
            download_retries: int = None,
            download_wait_seconds: int = None) -> str:
        if isdir(download_location):
            filename = join(download_location, posixpath.basename(URL))
        else:
            filename = download_location

        if exists(filename):
            logger.info(f"file already retrieved: {cl.file(filename)}")
            return filename

        # metadata = self.get_metadata(URL)
        metadata_URL = f"{URL}.xml"
        logger.info(f"checking metadata: {cl.URL(metadata_URL)}")

        if isdir(download_location):
            metadata_filename = join(download_location, posixpath.basename(metadata_URL))
        else:
            metadata_filename = f"{download_location}.xml"

        makedirs(dirname(metadata_filename), exist_ok=True)

        if XML_retries is None:
            XML_retries = XML_RETRIES

        if XML_timeout_seconds is None:
            XML_timeout_seconds = XML_TIMEOUT

        if download_retries is None:
            download_retries = DOWNLOAD_RETRIES

        if download_wait_seconds is None:
            download_wait_seconds = DOWNLOAD_WAIT_SECONDS

        metadata = None

        while XML_retries > 0:
            XML_retries -= 1
            command = f"wget -nc -c --user {self._username} --password {self._password} -O {metadata_filename} {metadata_URL}"
            logger.info(command)
            os.system(command)

            if not exists(metadata_filename):
                logger.warning(f"download not found for metadata URL: {metadata_URL}")
                logger.warning(f"waiting {XML_timeout_seconds} for retry")
                sleep(XML_timeout_seconds)
                continue

            XML_metadata_filesize = self.get_local_filesize(metadata_filename)

            if XML_metadata_filesize == 0 and exists(metadata_filename):
                logger.warning(f"removing corrupted zero-size metadata file: {metadata_filename}")

                try:
                    os.remove(metadata_filename)
                except:
                    logger.warning(f"unable to remove zero-size metadata file: {metadata_filename}")

                logger.warning(f"waiting {XML_timeout_seconds} for retry")
                sleep(XML_timeout_seconds)
                continue

            try:
                with open(metadata_filename, "r") as file:
                    metadata = xmltodict.parse(file.read())
            except Exception as e:
                logger.warning(e)
                logger.warning(f"unable to parse metadata file: {metadata_filename}")
                os.remove(metadata_filename)
                logger.warning(f"waiting {XML_timeout_seconds} for retry")
                sleep(XML_timeout_seconds)
                continue

        if metadata is None:
            raise DownloadFailed(f"unable to retrieve metadata URL: {metadata_URL}")  # exit code 16

        remote_checksum = str(
            metadata["GranuleMetaDataFile"]["GranuleURMetaData"]["DataFiles"]["DataFileContainer"]["Checksum"])
        checksum_type = str(
            metadata["GranuleMetaDataFile"]["GranuleURMetaData"]["DataFiles"]["DataFileContainer"]["ChecksumType"])
        remote_filesize = int(
            metadata["GranuleMetaDataFile"]["GranuleURMetaData"]["DataFiles"]["DataFileContainer"]["FileSize"])

        logger.info(
            f"metadata retrieved {checksum_type} checksum: {cl.val(remote_checksum)} size: {cl.val(remote_filesize)} URL: {cl.URL(metadata_URL)}")
        makedirs(dirname(filename), exist_ok=True)
        logger.info(f"downloading {cl.URL(URL)} -> {cl.file(filename)}")

        # Use a temporary file for downloading
        temporary_filename = f"{filename}.download"

        while download_retries > 0:
            download_retries -=1

            try:
                if exists(temporary_filename):
                    temporary_filesize = self.get_local_filesize(temporary_filename)

                    if temporary_filesize > remote_filesize:
                        logger.warning(
                            f"removing corrupted file with size {temporary_filesize} greater than remote size {remote_filesize}: {temporary_filename}")
                        remove(temporary_filename)

                    elif temporary_filesize == remote_filesize:
                        local_checksum = self.get_local_checksum(temporary_filename, checksum_type=checksum_type)

                        if local_checksum == remote_checksum:
                            try:
                                shutil.move(temporary_filename, filename)
                            except Exception as e:
                                if exists(filename):
                                    logger.warning(f"unable to move temporary file: {temporary_filename}")
                                    return filename

                                logger.exception(e)
                                raise DownloadFailed(f"unable to move temporary file: {temporary_filename}")

                            return filename
                        else:
                            logger.warning(
                                f"removing corrupted file with local checksum {local_checksum} and remote checksum {remote_checksum}: {temporary_filename}")
                            remove(temporary_filename)
                    else:
                        logger.info(f"resuming incomplete download: {cl.file(temporary_filename)}")

                command = f"wget -nc -c --user {self._username} --password {self._password} -O {temporary_filename} {URL}"
                logger.info(command)
                os.system(command)

                if not exists(temporary_filename):
                    raise ConnectionError(f"unable to download URL: {URL}")

                local_filesize = self.get_local_filesize(temporary_filename)
                local_checksum = self.get_local_checksum(temporary_filename, checksum_type=checksum_type)

                if local_filesize != remote_filesize or local_checksum != remote_checksum:
                    os.remove(temporary_filename)
                    raise ConnectionError(
                        f"removing corrupted file with local filesize {local_filesize} remote filesize {remote_filesize} local checksum {local_checksum} remote checksum {remote_checksum}: {temporary_filename}")

                # Download successful, rename the temporary file to its proper name
                shutil.move(temporary_filename, filename)

                logger.info(
                    f"successful download with filesize {cl.val(local_filesize)} checksum {cl.val(local_checksum)}: {cl.file(filename)}")

                return filename
            except Exception as e:
                if download_retries == 0:
                    raise e
                else:
                    logger.warning(e)
                    logger.warning(f"waiting {download_wait_seconds} seconds to retry download")
                    sleep(download_wait_seconds)
                    continue
