"""Score the flood tool based on provided sample data."""

import os
import sys
import time
import logging

import pandas as pd
import numpy as np

import flood_tool
from .scores import SCORES

logger = logging.getLogger(__name__)

DEFAULT_TEST_DATA = os.path.join(flood_tool._data_dir, "postcodes_labelled.csv")

if len(sys.argv) > 1:
    test_data = pd.read_csv(sys.argv[1]).sort_values(by="postcode")
    unlabelled_file = sys.argv[2]
    labelled_file = sys.argv[3]
    tool = flood_tool.Tool(unlabelled_file, labelled_file)
else:
    test_data = pd.read_csv(DEFAULT_TEST_DATA).sort_values(by="postcode")
    unlabelled_file = ""
    labelled_file = ""
    district_file = ""
    sector_file = ""
    tool = flood_tool.Tool(unlabelled_file, labelled_file, district_file, sector_file)


print("\nScoring flood class from postcode methods")
print("=========================================\n")
methods = flood_tool.flood_class_from_postcode_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.train([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_flood_class_from_postcode(test_data.postcode, method)
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    score = sum([SCORES[_p - 1, _t - 1] for _p, _t in zip(prediction, test_data.riskLabel)])
    print(f"{name}: score {score}")


print("\nScoring flood class from easting and northing")
print("=========================================\n")
methods = flood_tool.flood_class_from_location_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.train([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_flood_class_from_OSGB36_location(
        test_data.easting, test_data.northing, method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    score = sum([SCORES[_p - 1, _t - 1] for _p, _t in zip(prediction, test_data.riskLabel)])
    print(f"{name}: score {score}")


print("\nScoring flood class from latitude and longitude")
print("=========================================\n")
methods = flood_tool.flood_class_from_location_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.train([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()

    latitude, longitude = flood_tool.geo.get_gps_lat_long_from_easting_northing(
        test_data.easting.tolist(), test_data.northing.tolist()
    )

    prediction = tool.predict_flood_class_from_WGS84_locations(
        longitude.tolist(), latitude.tolist(), method
    )

    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    score = sum([SCORES[_p - 1, _t - 1] for _p, _t in zip(prediction, test_data.riskLabel)])
    print(f"{name}: score {score}")


print("\nScoring historic flooding methods")
print("=================================\n")
methods = flood_tool.historic_flooding_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.train([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_historic_flooding(test_data.postcode.to_list(), method)
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    prediction = prediction.reindex(test_data.postcode)

    truth = test_data.set_index("postcode")
    truth = truth.historicallyFlooded

    tps = sum(prediction & truth)
    fps = sum(prediction & ~truth)
    tns = sum(~prediction & ~truth)
    fns = sum(~prediction & truth)

    if tps + fps == 0:
        print(f"{name}: precision nan")
    else:
        print(f"{name}: precision {tps/(tps+fps):0.3f}")
    if tps + fns == 0:
        print(f"{name}: recall nan")
    else:
        print(f"{name}: recall {tps/(tps+fns):0.3f}")
    print(f"{name}: accuracy {(tps+tns)/(tps+tns+fps+fns):0.3f}")
    if 2 * tps + fps + fns == 0:
        print(f"{name}: f1 score nan")
    else:
        print(f"{name}: f1 score {2*tps/(2*tps+fps+fns):0.3f}")


print("\nScoring house price methods")
print("===========================\n")
methods = flood_tool.house_price_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.train([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_median_house_price(test_data.postcode.to_list(), method)
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    prediction = prediction.reindex(test_data.postcode)
    truth = test_data.set_index("postcode")
    truth = truth.medianPrice

    valid = prediction.notna() & truth.notna()

    score = np.sqrt(np.mean((prediction.loc[valid] - truth.loc[valid]) ** 2))

    fps = sum(prediction.isna() & ~truth.notna())
    fns = sum(~prediction.isna() & truth.notna())

    print(f"{name}: score {score:0.2f}")
    print(f"{name}: false positives {fps}")
    print(f"{name}: false negatives {fns}")


print("\nScoring local authority methods")
print("===============================\n")
methods = flood_tool.local_authority_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.train([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_local_authority(
        test_data.easting.to_list(), test_data.northing.to_list(), method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    locations = [
        (east, north) for east, north in zip(test_data.easting, test_data.northing)
    ]

    truth = test_data.set_index(pd.Series(locations, index=test_data.index))
    truth = truth.localAuthority

    tps = sum(prediction == truth)

    print(f"{name}: accuracy {tps/len(truth):0.3f}")
