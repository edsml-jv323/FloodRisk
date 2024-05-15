"""Test flood tool."""

import numpy as np


import flood_tool.tool as tool


testtool = tool.Tool()


def test_lookup_easting_northing():
    """Check"""
    data = testtool.lookup_easting_northing(["M34 7QL"])
    assert np.isclose(data.iloc[0].easting, 393470).all()
    assert np.isclose(data.iloc[0].northing, 394371).all()


def test_lookup_lat_long():
    """Check"""
    data = testtool.lookup_lat_long(["M34 7QL"])
    assert np.isclose(data.iloc[0].latitude, 53.4461, rtol=1.0e-3).all()
    assert np.isclose(data.iloc[0].longitude, -2.0997, rtol=1.0e-3).all()


# Convenience function to run tests directly.
if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_lat_long()
