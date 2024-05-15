#!/usr/bin/env python

from setuptools import setup

# Modify, update, and improve this file as necessary.

setup(
    name="Flood Tool",
    version="1.0",
    description="Flood Risk Analysis Tool",
    author="ACDS project Team X",  # update this
    packages=["flood_tool"],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "folium",
        "scikit-learn",
    ],
)
