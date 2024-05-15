"""Interactions with rainfall and river data."""

from os import path
import urllib
import json
import re

import pandas as pd

from ..tool import _data_dir

__all__ = [
    "get_station_data_from_csv",
    "get_stations",
    "get_latest_rainfall_readings",
    "get_live_station_data",
]

_wet_day_file = path.join(_data_dir, "wet_day.csv")
_typical_day_file = path.join(_data_dir, "typical_day.csv")
_station_file = path.join(_data_dir, "stations.csv")


def get_station_data_from_csv(
    filename: str, station_reference: str | None = None
) -> pd.Series:
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference : str, optional
        station_reference to return.

    Returns
    -------
    pandas.Series
        Series of data values

    Examples
    --------
    >>> data = get_station_data_from_csv(_wet_day_file, '0184TH')
    """
    frame = pd.read_csv(filename)
    frame["dateTime"] = pd.to_datetime(frame["dateTime"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    if station_reference is not None:
        frame = frame.loc[frame.stationReference == station_reference]
        frame.drop("stationReference", axis=1, inplace=True)
        frame.set_index("dateTime", inplace=True)

    else:
        frame.set_index(["stationReference", "dateTime"], inplace=True)

    return frame.sort_index()


def get_stations(filename: str = _station_file) -> pd.DataFrame:
    """Return a DataFrame of the measurement stations.

    Parameters
    ----------

    filename: str, optional
        filename to read

    Returns
    -------
    pandas.DataFrame
        DataFrame of the measurement stations.

    Examples
    --------
    >>> stations = get_stations()
    >>> stations.stationReference.head(5) # doctest: +NORMALIZE_WHITESPACE
    0      000008
    1      000028
    2    000075TP
    3    000076TP
    4    000180TP
    Name: stationReference, dtype: object
    """

    return pd.read_csv(filename)


# See https://environment.data.gov.uk/flood-monitoring/doc/reference
# and https://environment.data.gov.uk/flood-monitoring/doc/rainfall
# for full API documentation.

API_URL = "http://environment.data.gov.uk/flood-monitoring/"


rainfall_station = re.compile(".*/(.*)-rainfall-(.*)-t-15_min-(.*)/.*")


def split_rainfall_api_id(input: str) -> tuple[str, str, str]:
    """Split rainfall station API id into component parts
    using a regular expression.
    """
    match = rainfall_station.match(input)

    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return "", "", ""


def get_latest_rainfall_readings() -> pd.DataFrame:
    """Return last readings for all rainfall stations via live API.

    >>> data = get_latest_rainfall_readings()
    """

    url = API_URL + "data/readings?parameter=rainfall&latest"
    data = urllib.request.urlopen(url)
    data = json.load(data)

    dframe = pd.DataFrame(data["items"])

    # split id into parts
    id_data = dframe["@id"].apply(split_rainfall_api_id)
    dframe["stationReference"] = id_data.apply(lambda x: x[0])
    dframe["qualifier"] = id_data.apply(lambda x: x[1].replace("_", " ").title())
    dframe["unitName"] = id_data.apply(lambda x: x[2])
    dframe.drop(["@id", "measure"], axis=1, inplace=True)

    dframe["dateTime"] = dframe["dateTime"].apply(pd.to_datetime)

    dframe.set_index(["stationReference", "dateTime"], inplace=True)

    dframe["parameter"] = "rainfail"

    dframe["value"] = pd.to_numeric(dframe["value"], errors="coerce")

    return dframe.sort_index()


def get_live_station_data(station_reference: str) -> pd.DataFrame:
    """Return recent readings for a specified recording station from live API.

    Parameters
    ----------

    station_reference
        station_reference to return.

    Examples
    --------

    >>> data = get_live_station_data('0184TH')
    """

    url = API_URL + f"id/stations/{station_reference}/readings?_sorted"
    data = urllib.request.urlopen(url)
    data = json.load(data)

    dframe = pd.DataFrame(data["items"])

    return dframe
