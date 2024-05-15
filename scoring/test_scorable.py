"""Module to check that your flood tool will be scorable."""

import flood_tool
import pandas as pd
import numbers

tool = flood_tool.Tool()
tool.train()
POSTCODES = ["M34 7QL", "OL4 3NQ"]
EASTINGS = [393470, 394371]
NORTHINGS = [395420, 405669]

PRICE_METHODS = flood_tool.house_price_methods
CLASS_METHODS = flood_tool.flood_class_from_postcode_methods
LCLASS_METHODS = flood_tool.flood_class_from_location_methods
LAUTH_METHODS = flood_tool.local_authority_methods
HIST_METHODS = flood_tool.historic_flooding_methods


def test_lookup_lat_long():
    """Check return type."""
    data = tool.lookup_lat_long(POSTCODES)
    assert issubclass(type(data), pd.DataFrame)
    for postcode in POSTCODES:
        assert postcode in data.index
    assert "latitude" in data
    assert "longitude" in data


def test_predict_flood_class_from_postcode():
    """Check return types."""
    for method in CLASS_METHODS.keys():
        data = tool.predict_flood_class_from_postcode(POSTCODES, method)
        assert issubclass(type(data), pd.Series)
        for postcode in POSTCODES:
            assert postcode in data.index
        assert all([isinstance(val, numbers.Number) for val in data.values])


def test_predict_flood_class_from_OSGB36_location():
    """Check return types."""
    for method in LCLASS_METHODS.keys():
        data = tool.predict_flood_class_from_OSGB36_location(EASTINGS, NORTHINGS, method)
        print(method)
        print(data)
        assert issubclass(type(data), pd.Series)
        for east, north in zip(EASTINGS, NORTHINGS):
            assert (east, north) in data.index
        assert all([isinstance(val, numbers.Number) for val in data.values])


def test_predict_median_house_price():
    """Check return type."""
    for method in PRICE_METHODS.keys():
        data = tool.predict_median_house_price(POSTCODES, method)
        assert issubclass(type(data), pd.Series)
        for postcode in POSTCODES:
            assert postcode in data.index
        assert all([isinstance(val, numbers.Number) for val in data.values])


def test_predict_local_authority():
    """Check return type."""
    for method in LAUTH_METHODS.keys():
        data = tool.predict_local_authority(EASTINGS, NORTHINGS, method)
        assert issubclass(type(data), pd.Series)
        for east, north in zip(EASTINGS, NORTHINGS):
            assert (east, north) in data.index


def test_predict_historic_flooding():
    """Check return type."""
    for method in HIST_METHODS.keys():
        data = tool.predict_historic_flooding(POSTCODES, method)
        print(method, data)
        assert issubclass(type(data), pd.Series)
        for postcode in POSTCODES:
            assert postcode in data.index


if __name__ == "__main__":
    test_predict_historic_flooding()
