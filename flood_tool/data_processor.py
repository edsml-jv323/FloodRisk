import os

import pandas as pd
import numpy as np
import logging
from flood_tool.geo import get_gps_lat_long_from_easting_northing
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    For expanding the data of the postcode (either labelled or unlabelled)
    and combining the data inside all resources files.
    """

    def __init__(
        self, df_postcodes: pd.DataFrame, resource_path: str, verbose: bool = True
    ):
        """
        Read the data of the postcode in a specific path and
        all the data in the resources file, including other features.
        """
        self.verbose = verbose
        self.df_postcodes = df_postcodes
        self.df_stations = pd.read_csv(os.path.join(resource_path, "stations.csv"))
        self.df_district = pd.read_csv(os.path.join(resource_path, "district_data.csv"))
        self.df_typical = pd.read_csv(os.path.join(resource_path, "typical_day.csv"))
        self.df_wet = pd.read_csv(os.path.join(resource_path, "wet_day.csv"))
        self.df_sector = pd.read_csv(os.path.join(resource_path, "sector_data.csv"))

        if self.verbose:
            logger.info("Data loaded successfully loaded.")

    def process_gps(self, longitudes: list[float], latitudes: list[float]) -> pd.DataFrame:
        """Helper method to take lat and lon and extract a
        uniform dataset for training various models."""
        df_gps = pd.DataFrame(np.c_[latitudes, longitudes], columns=["x", "y"])
        df_postcodes = self.get_gps_dataframe(keep_os=True)

        # merge using the closest lat and lon with df_postcodes processed
        distances = cdist(
            df_gps[["x", "y"]],
            df_postcodes[["latitude", "longitude"]],
            metric="euclidean",
        )
        nearest_indices = np.argmin(distances, axis=1)
        nearest_features = df_postcodes.iloc[nearest_indices]
        df_merged = pd.concat(
            [
                df_gps.reset_index(drop=True),
                nearest_features.reset_index(drop=True),
            ],
            axis=1,
        )

        return df_merged

    def process_os(self, eastings: list[float], northings: list[float]) -> pd.DataFrame:
        """
        Helper method to take easting and northing and
        extract a uniform dataset for training various models.
        Closest easting and northing coordinate in the
        data is located to extract required features.
        Tested visually using https://gridreferencefinder.com/os.php
        for difference in mapping, showing decent accuracy.
        """
        df_os = pd.DataFrame(np.c_[eastings, northings], columns=["x", "y"])
        df_postcodes = self.get_gps_dataframe(keep_os=True)

        distances = cdist(
            df_os[["x", "y"]],
            df_postcodes[["easting", "northing"]],
            metric="euclidean",
        )
        nearest_indices = np.argmin(distances, axis=1)
        nearest_features = df_postcodes.iloc[nearest_indices]
        df_merged = pd.concat(
            [
                df_os.reset_index(drop=True),
                nearest_features.reset_index(drop=True),
            ],
            axis=1,
        )

        return df_merged

    def get_gps_dataframe(self, keep_os: bool = False) -> pd.DataFrame:
        """
        The first step is to convert the east-north coordinates in the postcode to GPS
        (only the GPS coordinates are preserved, not the easting and northing).
        """
        # 转换 postcode east-north 到 gps 坐标
        df_postcodes = self.df_postcodes.copy()
        easting = df_postcodes["easting"]
        northing = df_postcodes["northing"]
        latitude, longitude = np.round(
            get_gps_lat_long_from_easting_northing(easting, northing), 6
        )
        if not keep_os:
            df_postcodes = df_postcodes.drop(["easting", "northing"], axis=1)
        df_postcodes["latitude"] = latitude
        df_postcodes["longitude"] = longitude
        return df_postcodes

    def get_stations_dataframe(self) -> pd.DataFrame:
        """
        Find the nearest station with GPS coordinates for each postcode

        Combining features:
        ----------
        stationReference
        stationName
        maxOnRecord
        minOnRecord
        typicalRangeHigh
        typicalRangeLow
        """

        df_stations = self.df_stations.copy()

        df_postcodes = self.get_gps_dataframe(keep_os=True)
        df_stations = df_stations.dropna(subset=["latitude", "longitude"])
        df_gps_reset = df_stations.reset_index(drop=True)
        distances = cdist(
            df_postcodes[["latitude", "longitude"]],
            df_gps_reset[["latitude", "longitude"]],
            metric="euclidean",
        )
        nearest_indices = np.argmin(distances, axis=1)
        df_postcodes["nearest_latitude"] = df_gps_reset.iloc[nearest_indices][
            "latitude"
        ].values
        df_postcodes["nearest_longitude"] = df_gps_reset.iloc[nearest_indices][
            "longitude"
        ].values
        df_gps_reset.drop(columns=["latitude", "longitude"], inplace=True)
        df_nearest_stations = df_gps_reset.iloc[nearest_indices].reset_index(drop=True)
        df_postcode_station = pd.concat([df_postcodes, df_nearest_stations], axis=1)

        return df_postcode_station

    def get_human_dataframe(self) -> pd.DataFrame:
        """
        Using the data combined from the previous step,
        based on the link between postcode and postcodeSector/postcodeDistrict.

        Combining features:
        ----------
        catsPerHousehold
        dogsPerHousehold
        households
        numberOfPostcodeUnits
        headcount
        """
        df_postcode_station = self.get_stations_dataframe()
        df_postcode_station["postcodeDistrict"] = (
            df_postcode_station["postcode"].str.split(" ").str[0]
        )
        pt_district = df_postcode_station.merge(
            self.df_district, on="postcodeDistrict", how="outer"
        )
        pt_district = pt_district.dropna(subset=["postcode"])
        pt_district["postcodeSector"] = pt_district["postcode"].str[:-2]
        df_human = pt_district.merge(self.df_sector, on="postcodeSector", how="outer")
        df_human = df_human.dropna(subset=["postcode"])
        return df_human

    def process_rainfall_data(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
        df.dropna(subset=["value"], inplace=True)

        # transform unitname
        rainfall_mask = (df["parameter"] == "rainfall") & (df["unitName"] == "m")
        df.loc[rainfall_mask, "value"] *= 1000

        df_rainfall = df[df["parameter"] == "rainfall"].copy()
        df_rainfall["value"] = df_rainfall["value"].apply(lambda x: max(0, x))

        rainfall_total = (
            df_rainfall.groupby(["stationReference", df_rainfall["dateTime"].dt.hour])[
                "value"
            ]
            .sum()
            .reset_index()
        )
        rainfall_avg = (
            rainfall_total.groupby("stationReference")["value"].mean().reset_index()
        )
        rainfall_avg.columns = ["stationReference", feature_name]

        return rainfall_avg

    def get_rainfall_dataframe(self) -> pd.DataFrame:
        df_postcode_station = self.get_stations_dataframe()
        df_typical = self.df_typical.copy()
        df_wet = self.df_wet.copy()
        df_stations = self.df_stations.copy()

        # Process typical and wet rainfall data
        rainfall_typical = self.process_rainfall_data(
            df_typical, "typical_average_rainfall_per_hour"
        )
        rainfall_wet = self.process_rainfall_data(df_wet, "wet_average_rainfall_per_hour")

        # Merge with station data
        human_typical = df_stations.merge(
            rainfall_typical, on="stationReference", how="outer"
        ).dropna(subset=["latitude", "typical_average_rainfall_per_hour"])
        human_wet = df_stations.merge(
            rainfall_wet, on="stationReference", how="outer"
        ).dropna(subset=["latitude", "wet_average_rainfall_per_hour"])

        # Combine
        df_postcode_station = self.calculate_nearest(
            df_postcode_station, human_typical, "typical_average_rainfall_per_hour"
        )
        df_postcode_station = self.calculate_nearest(
            df_postcode_station, human_wet, "wet_average_rainfall_per_hour"
        )

        return df_postcode_station

    def calculate_nearest(
        self,
        df_postcode_station: pd.DataFrame,
        df_gps_reset: pd.DataFrame,
        feature_name: str,
    ) -> pd.DataFrame:
        """Helper method for refactoring get_rainfall_data"""
        df_gps_reset = df_gps_reset.reset_index(drop=True)
        distances = cdist(
            df_postcode_station[["latitude", "longitude"]],
            df_gps_reset[["latitude", "longitude"]],
            metric="euclidean",
        )
        nearest_indices = np.argmin(distances, axis=1)
        df_postcode_station[feature_name] = df_gps_reset.iloc[nearest_indices][
            feature_name
        ].values
        return df_postcode_station

    def get_combined_dataframe(self) -> pd.DataFrame:
        """
        Combining features:
        ----------
        catsPerHousehold
        dogsPerHousehold
        households
        numberOfPostcodeUnits
        headcount
        """
        df_rainfall = self.get_rainfall_dataframe()
        df_district = self.df_district.copy()
        df_sector = self.df_sector.copy()

        df_rainfall["postcodeDistrict"] = df_rainfall["postcode"].str.split(" ").str[0]
        df_merged = df_rainfall.merge(df_district, on="postcodeDistrict", how="outer")
        df_merged.dropna(subset=["postcode"], inplace=True)
        df_merged["postcodeSector"] = df_merged["postcode"].str[:-2]

        df_merged = df_merged.merge(df_sector, on="postcodeSector", how="outer")
        df_merged.dropna(subset=["postcode"], inplace=True)

        if self.verbose:
            logger.info("Data successfully merged.")

        return df_merged
