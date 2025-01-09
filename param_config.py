#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

# API configuration
API_KEY = 'sk-XX'  # Replace with your actual API key
BASE_URL = "https://XXX"

# File path configuration

ROOT_FOLDER = r'/Users/'
STANDARD_IMAGE_PATH = os.path.join(ROOT_FOLDER, "Standard_Median_EVI_RDVI.jpg")
SAMPLE_DIR = os.path.join(ROOT_FOLDER, "timeseries_plots_target")

# Other configuration
CROP_TYPES = ["Maize", "Soybean", "Rice", "Others"]
MAX_ATTEMPTS = 3

