"""Test coordinate transformations."""
import flood_tool.geo as geo
import numpy as np


def test_get_easting_northing_from_gps_lat_long():
    """Test lat_long_to_osgb36."""
    lat = np.array([55.5])
    long = np.array([-1.54])
    easting, northing = geo.get_easting_northing_from_gps_lat_long(lat, long)
    assert np.isclose(easting, 429157).all()
    assert np.isclose(northing, 623009).all()


def test_get_gps_lat_long_from_easting_northing():
    """Test lat_long_to_osgb36."""
    easting = np.array([422297.8])
    northing = np.array([412878.7])
    lat, long = geo.get_gps_lat_long_from_easting_northing(easting, northing)
    assert np.isclose(lat, 53.612, rtol=1.0e-3).all()
    assert np.isclose(long, -1.664, rtol=1.0e-3).all()
