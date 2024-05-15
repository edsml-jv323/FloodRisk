"""Example module in template package."""
import logging
import os

import numpy as np
import pandas as pd

from . import geo
from .data_processor import DataProcessor

from flood_tool.models.flood_predictor import FloodPredictor
from flood_tool.models.local_authority_predictor import LocalAuthorityPredictor
from flood_tool.models.historical_flood_predictor import HistoricalFloodPredictor

from utils.utils import flood_scoring, init_logging
from .models.median_price_predictor import MedianPricePredictor

init_logging()

logger = logging.getLogger(__name__)

__all__ = [
    "Tool",
    "_data_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
]


_data_dir = os.path.join(os.path.dirname(__file__), "resources")


def get_flood_postcode_methods() -> dict[str, str]:
    return {
        "xgb_regressor": "XGBoost regressor",
        "rf_smote": "Random forest classifier with SMOTE",
        "rf_class_downsampling": "Random forest classifier with class downsampling",
        "zero_risk": "All zero risk",
    }


def get_flood_location_methods() -> dict[str, str]:
    return {
        "xgb_regressor": "XGBoost regressor",
        "rf_smote": "Random forest classifier with SMOTE",
        "rf_class_downsampling": "Random forest classifier with class downsampling",
        "zero_risk": "All zero risk",
    }


def get_house_price_methods() -> dict[str, str]:
    return {
        "linear_regression": "Linear regression",
        "all_england_median": "All England median",
        "knn_regressor": "K nearest neighbours regression",
        # "new": "K nearest neighbours regression",
    }


def get_local_authority_methods() -> dict[str, str]:
    return {
        "knn_classifier": "K nearest neighbours classifier",
        "do_nothing": "Do nothing",
    }


def get_historic_flooding_methods() -> dict[str, str]:
    return {
        "baseline_linear": "Baseline logistic regression classifier",
        "rf_classifier": "Random forest classifier",
        "xgb_classifier": "Random forest classifier",
        "all_false": "All False",
    }


flood_class_from_postcode_methods = get_flood_postcode_methods()
local_authority_methods = get_local_authority_methods()
historic_flooding_methods = get_historic_flooding_methods()
house_price_methods = get_house_price_methods()
flood_class_from_location_methods = get_flood_location_methods()


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(
        self,
        unlabelled_unit_data: str = "",
        labelled_unit_data: str = "",
        sector_data: str = "",
        district_data: str = "",
        resource_path: str = "",
        additional_data: dict = {},
    ):
        """
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additiona .csv files containing addtional
            information on households.
        """

        if unlabelled_unit_data == "":
            self.unlabelled_unit_data = os.path.join(_data_dir, "postcodes_unlabelled.csv")
        else:
            self.unlabelled_unit_data = unlabelled_unit_data

        if labelled_unit_data == "":
            self.labelled_unit_data = os.path.join(_data_dir, "postcodes_labelled.csv")
        else:
            self.labelled_unit_data = labelled_unit_data  # from command line

        if sector_data == "":
            self.sector_data = os.path.join(_data_dir, "sector_data.csv")
        else:
            self.sector_data = sector_data

        if district_data == "":
            self.district_data = os.path.join(_data_dir, "district_data.csv")
        else:
            self.district_data = district_data

        if resource_path == "":
            self.resource_path = _data_dir
        else:
            self.resource_path = resource_path

        self._postcodedb = pd.read_csv(
            self.unlabelled_unit_data
        )  # reading in unlabelled data

        # # full database of postcodes
        df_labelled = pd.read_csv(self.labelled_unit_data)
        self.df_stacked = pd.concat(
            [self._postcodedb, df_labelled[self._postcodedb.columns]], ignore_index=True
        )

        # get labelled data for training
        dp_labelled = DataProcessor(
            df_postcodes=df_labelled,
            resource_path=self.resource_path,
            verbose=True,
        )

        self.training_data = dp_labelled.get_combined_dataframe().sort_values(by="postcode")
        # self.training_data = dp_labelled.df_postcodes
        # self.training_data = df_labelled

        # combine all postcode samples
        dp_all = DataProcessor(
            df_postcodes=self.df_stacked,
            resource_path=self.resource_path,
            verbose=True,
        )

        self.combined_data = dp_all.get_combined_dataframe().sort_values(by="postcode")
        # self.combined_data = self.df_stacked

    def train(
        self,
        models: list[str] = [],
        update_labels: str = "",
        tune_hyperparameters: bool = False,
    ) -> None:
        """Train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to train
        update_labels : str, optional
            Filename of a .csv file containing a labelled set of samples.
        tune_hyperparameters : bool, optional
            If true, models can tune their hyperparameters, where
            possible. If false, models use your chosen default hyperparameters.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.train(fcp_methods[0])  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        """

        if update_labels:
            print("updating labelled sample file")
            # update your labelled samples

        for model in models:
            if tune_hyperparameters:
                print(f"tuning {model} hyperparameters")
            else:
                print(f"training {model}")
            # Do your training for the specified models

    def lookup_easting_northing(
        self, postcodes: list[str], dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """Get a dataframe of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        dtype
        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of OSGB36 easthing and northing,
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the input unlabelled postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['M34 7QL'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                  easting  northing
        postcode
        M34 7QL    393470	 394371
        >>> results = tool.lookup_easting_northing(['M34 7QL', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                  easting  northing
        postcode
        M34 7QL  393470.0  394371.0
        AB1 2PQ       NaN       NaN
        """

        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing"]]

    def lookup_lat_long(self, postcodes: list[str]) -> pd.DataFrame:
        """Get a Pandas dataframe containing GPS latitude and longitude
        information for a collection of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NAN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        """

        # first look up easting and northing from postcode db
        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)
        eastings, northings = frame.easting.tolist(), frame.northing.tolist()
        latitude, longitude = geo.get_gps_lat_long_from_easting_northing(
            eastings, northings, rads=False
        )
        return pd.DataFrame(
            np.c_[longitude, latitude],
            columns=["longitude", "latitude"],
            index=postcodes,
        )

    def predict_flood_class_from_postcode(
        self, postcodes: list[str], method: str = "zero_risk"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """
        # retrieve postcodes from postcode database (all postcodes in labelled and unlabelled)
        X = self.combined_data[self.combined_data["postcode"].isin(postcodes)].sort_values(
            by="postcode"
        )
        X.drop_duplicates(inplace=True)

        match method:
            case "zero_risk":
                return pd.Series(
                    data=np.ones(len(postcodes), int),
                    index=np.asarray(postcodes),
                    name="riskLabel",
                )
            case "xgb_regressor" | "rf_smote" | "rf_class_downsampling":
                fp = FloodPredictor(
                    training_data=self.training_data.copy(),
                    model_alias=method,
                    scoring=flood_scoring,
                )
                logger.info(f"{method} test: {fp.test_score}")
                return fp.predict_flood_risk(X=X, index="postcode")
            case _:
                raise NotImplementedError(f"method {method} not implemented")

    def predict_flood_class_from_OSGB36_location(
        self, eastings: list[float], northings: list[float], method: str = "zero_risk"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        """

        data_processor = DataProcessor(
            df_postcodes=self.combined_data,
            resource_path=self.resource_path,
            verbose=False,  # only for auxiliary data
        )
        X = data_processor.process_os(eastings=eastings, northings=northings)
        logger.info(X.shape)

        match method:
            case "zero_risk":
                return pd.Series(
                    data=np.ones(len(eastings), int),
                    index=((est, nth) for est, nth in zip(eastings, northings)),
                    name="riskLabel",
                )
            case "xgb_regressor" | "rf_smote" | "rf_class_downsampling":
                fp = FloodPredictor(
                    training_data=self.training_data.copy(),
                    model_alias=method,
                    scoring=flood_scoring,
                )
                logger.info(f"{method} test: {fp.test_score}")
                return fp.predict_flood_risk(X, index=("easting", "northing"))
            case _:
                raise NotImplementedError(f"method {method} not implemented")

    def predict_flood_class_from_WGS84_locations(
        self, longitudes: list[float], latitudes: list[float], method: str = "zero_risk"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.
        Parameters
        ----------
        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : str (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the method to be used.
        Returns
        -------
        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        data_processor = DataProcessor(
            df_postcodes=self.combined_data,
            resource_path=self.resource_path,
            verbose=False,
        )

        X = data_processor.process_gps(latitudes=latitudes, longitudes=longitudes)

        match method:
            case "zero_risk":
                return pd.Series(
                    data=np.ones(len(longitudes), int),
                    index=[(lng, lat) for lng, lat in zip(longitudes, latitudes)],
                    name="riskLabel",
                )
            case "xgb_regressor" | "rf_smote" | "rf_class_downsampling":
                fp = FloodPredictor(
                    training_data=self.training_data,
                    model_alias=method,
                    scoring=flood_scoring,
                )
                logger.info(f"{method} test: {fp.test_score}")
                return fp.predict_flood_risk(X, index=("latitude", "longitude"))
            case _:
                raise NotImplementedError(f"method {method} not implemented")

    def predict_median_house_price(
        self, postcodes: list[str], method: str = "all_england_median"
    ) -> pd.Series:
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """

        # retrieve postcodes from postcode data source
        X = self.combined_data[self.combined_data["postcode"].isin(postcodes)].sort_values(
            by="postcode"
        )
        X.drop_duplicates(inplace=True)

        match method:
            case "all_england_median":
                return pd.Series(
                    data=np.full(len(postcodes), 245000.0),
                    index=np.asarray(postcodes),
                    name="medianPrice",
                )
            case "linear_regression" | "knn_regressor" | "new":
                mp = MedianPricePredictor(
                    training_data=self.training_data,
                    model_alias=method,
                    scoring="neg_mean_squared_error",
                )
                return mp.predict_from_postcode(X=X, index="postcode")
            case _:
                raise NotImplementedError(f"Method {method} not implemented")

    def predict_local_authority(
        self, eastings: list[float], northings: list[float], method: str = "do_nothing"
    ) -> pd.Series:
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            location, and indexed by eastings and northings.
        """

        match method:
            case "do_nothing":
                return pd.Series(
                    data=np.full(len(eastings), np.nan),
                    index=[(est, nth) for est, nth in zip(eastings, northings)],
                    name="localAuthority",
                )
            case "knn_classifier":
                lap = LocalAuthorityPredictor(
                    training_data=self.training_data,
                    model_alias=method,
                    scoring="f1_macro",
                )
                logger.info(f"{method} test: {lap.test_score}")
                return lap.predict(eastings, northings)
            case _:
                raise NotImplementedError(f"method {method} not implemented")

    def predict_historic_flooding(
        self, postcodes: list[str], method: str = "all_false"
    ) -> pd.Series:
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        """

        # retrieve postcodes from postcode data source to add required features
        X = self.combined_data[self.combined_data["postcode"].isin(postcodes)].sort_values(
            by="postcode"
        )
        X.drop_duplicates(inplace=True)

        match method:
            case "all_false":
                return pd.Series(
                    data=np.full(len(postcodes), False),
                    index=np.asarray(postcodes),
                    name="historicallyFlooded",
                )
            case "baseline_linear" | "rf_classifier" | "xgb_classifier":
                hp = HistoricalFloodPredictor(
                    training_data=self.training_data,
                    model_alias=method,
                    scoring="f1",
                )
                logger.info(f"{method}: {hp.test_score}")
                return hp.predict_historical_flood(X, index="postcode")
            case _:
                raise NotImplementedError(f"method {method} not implemented")

    def predict_total_value(self, postal_data: list[str]) -> pd.Series:
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcodesectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """

        raise NotImplementedError

    def predict_annual_flood_risk(
        self, postcodes: list[str], risk_labels: pd.Series | None = None
    ) -> pd.Series:
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        risk_labels = risk_labels or self.get_flood_class(postcodes)

        raise NotImplementedError
