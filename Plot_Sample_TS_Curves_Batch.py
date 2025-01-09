#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import argparse
from image_processing import sentinel2toa, cloudMask, addVariables, normalizeRadar, addS1Variables

# Set HTTP proxy (adjust according to your network environment)
os.environ['HTTP_PROXY'] = 'http://xxxx'
os.environ['HTTPS_PROXY'] = 'http://xxxx'

# Initialize Earth Engine
ee.Initialize()

# Define study area and time range
target_year = 2020
startMonth = 6
startDay = 1
endMonth = 10
endDay = 30
province_name = 'Heilongjiang Sheng'

start_date_str = '2020-06-01'

ymax_optical = 0.908443059574166
ymin_optical = 0.014579322388449917
ymax_radar = 3.015914464333234
ymin_radar = 2.3148448006954974

# Define study area and cropland mask
aoi = ee.FeatureCollection("FAO/GAUL/2015/level2") \
    .filter(ee.Filter.eq('ADM1_NAME', province_name)) \
    .geometry()

cropland = ee.ImageCollection("ESA/WorldCover/v100") \
    .first() \
    .select('Map') \
    .eq(40) \
    .rename('cropland') \
    .clip(aoi)


def getTimeSeries(point, year, optical_index, radar_index, use_cluster=True):
    """
    Get time series data for a given point, including SNIC segmentation and cluster averaging

    :param point: ee.Geometry.Point Sampling point
    :param year: int Target year
    :param optical_index: str Selected optical index
    :param radar_index: str Selected radar index
    :return: tuple Tuple containing S2 and S1 time series data
    """
    start = ee.Date.fromYMD(year, startMonth, startDay)
    end = ee.Date.fromYMD(year, endMonth, endDay)

    region = point.buffer(1500).bounds()

    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(start, end) \
        .map(sentinel2toa) \
        .map(addVariables) \
        .map(cloudMask)

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(region) \
        .filterDate(start, end) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .map(normalizeRadar) \
        .map(addS1Variables)

    if use_cluster is False:
        # For classes 1 and 3, use pixel-level data directly
        cluster_mask = ee.Image(1)  # Create a mask of all 1s, no filtering
    else:
        snic_image = s2.median().select(['blue', 'green', 'red', 'nir'])
        snic = ee.Algorithms.Image.Segmentation.SNIC(
            image=snic_image,
            size=10,  # Superpixel size
            compactness=0.7,
            connectivity=8,
            neighborhoodSize=64
        )

        # Get cluster ID at sampling point
        cluster_id = snic.select('clusters').reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=10
        ).get('clusters')

        # Create cluster mask
        cluster_mask = snic.select('clusters').eq(ee.Number(cluster_id))

        # Calculate number of pixels in cluster
        cluster_size = cluster_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=10,
            maxPixels=1e9
        ).get('clusters')

        # Get cluster size (pixel count)
        cluster_size_value = cluster_size.getInfo()
        print(f"Cluster sample size (pixels): {cluster_size_value}")

    def addDateBandAndMask(image):
        return image.addBands(image.metadata('system:time_start').rename('date')) \
            .updateMask(cluster_mask)

    s2_masked = s2.map(addDateBandAndMask)
    s1_masked = s1.map(addDateBandAndMask)

    def extractMean(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            maxPixels=1e9
        )
        return ee.Feature(None, mean)

    s2_timeseries = s2_masked.select(['date', optical_index]).map(extractMean).getInfo()
    s1_timeseries = s1_masked.select(['date', radar_index]).map(extractMean).getInfo()

    return s2_timeseries, s1_timeseries

def clean_and_process_df(s2_data, s1_data, optical_index, radar_index):
    """
    Clean and process time series data from GEE

    :param s2_data: dict S2 time series data
    :param s1_data: dict S1 time series data
    :param optical_index: str Selected optical index
    :param radar_index: str Selected radar index
    :return: tuple Containing DataFrame for plotting and complete DataFrame for saving
    """
    s2_df = pd.DataFrame([
        {'date': pd.to_datetime(feature['properties']['date'], unit='ms'),
         optical_index: feature['properties'].get(optical_index, np.nan)}
        for feature in s2_data['features']
    ])
    s2_df = s2_df.set_index('date')

    s1_df = pd.DataFrame([
        {'date': pd.to_datetime(feature['properties']['date'], unit='ms'),
         radar_index: feature['properties'].get(radar_index, np.nan)}
        for feature in s1_data['features']
    ])
    s1_df = s1_df.set_index('date')

    df_for_plot = pd.merge(s2_df, s1_df, left_index=True, right_index=True, how='outer')
    df_for_plot = df_for_plot.interpolate(method='time')

    start_date = pd.to_datetime(f'{target_year}/{startMonth}/{startDay}')
    end_date = pd.to_datetime(f'{target_year}/{endMonth}/{endDay}')
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    df_for_save = pd.DataFrame(index=full_date_range, columns=[optical_index, radar_index])

    for date in full_date_range:
        date_str = date.strftime('%Y-%m-%d')
        if date.strftime('%Y-%m-%d') in s2_df.index.strftime('%Y-%m-%d'):
            df_for_save.loc[date, optical_index] = s2_df.loc[s2_df.index.strftime('%Y-%m-%d') == date_str, optical_index].values[0]
        if date.strftime('%Y-%m-%d') in s1_df.index.strftime('%Y-%m-%d'):
            df_for_save.loc[date, radar_index] = s1_df.loc[s1_df.index.strftime('%Y-%m-%d') == date_str, radar_index].values[0]

    df_for_save = df_for_save.fillna(-9999)

    return df_for_plot, df_for_save

def save_to_csv(df, sample_number, output_dir, optical_index, radar_index):
    """
    Save index data to separate CSV files

    :param df: pandas.DataFrame DataFrame containing index data
    :param sample_number: int Sample number
    :param output_dir: str Output directory path
    :param optical_index: str Selected optical index
    :param radar_index: str Selected radar index
    """
    optical_file = os.path.join(output_dir, f'{optical_index.lower()}_timeseries_sample_{sample_number}.csv')
    df[optical_index].to_csv(optical_file, date_format='%Y/%m/%d')
    print(f"{optical_index} data saved to: {optical_file}")

    radar_file = os.path.join(output_dir, f'{radar_index.lower()}_timeseries_sample_{sample_number}.csv')
    df[radar_index].to_csv(radar_file, date_format='%Y/%m/%d')
    print(f"{radar_index} data saved to: {radar_file}")


def smooth_timeseries(df, window_size=7):
    """
    Smooth time series data

    :param df: pandas.DataFrame Original DataFrame
    :param window_size: int Smoothing window size
    :return: pandas.DataFrame Smoothed DataFrame
    """
    smoothed_df = df.copy()
    for column in df.columns:
        # Replace -9999 with NaN for smoothing
        series = df[column].replace(-9999, np.nan)
        # Apply rolling mean for smoothing
        smoothed_df[column] = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed_df


def plot_smooth_timeseries(df, sample_number, output_dir, output_file_path, optical_index, radar_index):
    """
    Plot smoothed time series
    """
    # Create smoothed data
    smoothed_df = smooth_timeseries(df)

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    smoothed_df.index = pd.to_datetime(smoothed_df.index)

    optical_data = smoothed_df[optical_index].replace(-9999, np.nan)
    ln1 = ax1.plot(smoothed_df.index, optical_data, label=f"{optical_index}",
                   color='green', linestyle='-', linewidth=3)
    ax1.set_ylabel(optical_index, color='green', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='green', labelsize=16)

    radar_data = smoothed_df[radar_index].replace(-9999, np.nan)
    ln2 = ax2.plot(smoothed_df.index, radar_data, label=f"{radar_index}",
                   color='blue', linestyle='-', linewidth=3)
    ax2.set_ylabel(radar_index, color='blue', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=16)

    ax1.set_xlabel('Date', fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.7)

    start_date = pd.Timestamp(start_date_str)
    end_date = smoothed_df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='10D')
    ax1.set_xticks(date_range)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=14)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=16)

    # Use same y-axis range
    ax1.set_ylim(ymin_optical, ymax_optical)  # Range from standard sample
    ax2.set_ylim(ymin_radar, ymax_radar)

    plt.tight_layout()
    # Modify output filename to add "smoothed" identifier
    smooth_output_path = output_file_path.replace('.jpg', '_smoothed.jpg')
    plt.savefig(smooth_output_path, dpi=500, bbox_inches='tight')
    print(f"Smoothed time series plot for sample {sample_number} saved to: {smooth_output_path}")
    plt.close(fig)


def plot_timeseries(df, sample_number, output_dir, output_file_path, optical_index, radar_index):
    """
    Plot time series for given DataFrame and save

    :param df: pandas.DataFrame DataFrame containing index data
    :param sample_number: int Sample number
    :param output_dir: str Output directory path
    :param output_file_path: str Output file path
    :param optical_index: str Selected optical index
    :param radar_index: str Selected radar index
    """
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    df.index = pd.to_datetime(df.index)

    optical_data = df[optical_index].replace(-9999, np.nan)
    ln1 = ax1.plot(df.index, optical_data, label=optical_index, color='green', marker='o', linestyle='-', markersize=4.5, linewidth=3)
    ax1.set_ylabel(optical_index, color='green', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='green', labelsize=16)

    radar_data = df[radar_index].replace(-9999, np.nan)
    ln2 = ax2.plot(df.index, radar_data, label=radar_index, color='blue', marker='s', linestyle='-', markersize=4.5, linewidth=3)
    ax2.set_ylabel(radar_index, color='blue', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=16)

    ax1.set_xlabel('Date', fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.7)

    start_date = pd.Timestamp(start_date_str)
    end_date = df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='10D')
    ax1.set_xticks(date_range)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=14)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=16)

    ax1.set_ylim(ymin_optical, ymax_optical)  # Range from standard sample
    ax2.set_ylim(ymin_radar, ymax_radar)

    plt.tight_layout()
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    print(f"SNIC cluster time series plot for sample {sample_number} saved to: {output_file_path}")
    plt.close(fig)

    save_to_csv(df, sample_number, output_dir, optical_index, radar_index)



def load_local_data(sample_number, output_dir, optical_index, radar_index):
    """
    Load data from local CSV files and process -9999 values

    :return: DataFrame or None (if files don't exist)
    """
    optical_file = os.path.join(output_dir, f'{optical_index.lower()}_timeseries_sample_{sample_number}.csv')
    radar_file = os.path.join(output_dir, f'{radar_index.lower()}_timeseries_sample_{sample_number}.csv')

    if os.path.exists(optical_file) and os.path.exists(radar_file):
        try:
            # Read optical index data
            optical_df = pd.read_csv(optical_file, index_col=0, parse_dates=True)
            optical_df.index = pd.to_datetime(optical_df.index)
            # Replace -9999 with NaN
            optical_df = optical_df.replace(-9999, np.nan)

            # Read radar index data
            radar_df = pd.read_csv(radar_file, index_col=0, parse_dates=True)
            radar_df.index = pd.to_datetime(radar_df.index)
            # Replace -9999 with NaN
            radar_df = radar_df.replace(-9999, np.nan)

            # Merge data
            df = pd.merge(optical_df, radar_df, left_index=True, right_index=True, how='outer')
            df.columns = [optical_index, radar_index]

            # Interpolate data (optional)
            df = df.interpolate(method='time')

            return df
        except Exception as e:
            print(f"Error reading local data: {e}")
            return None

    return None


def main(optical_index='NDVI', radar_index='RVI'):
    root_folder = '/users/qinxl/Documents'
    input_file = os.path.join(root_folder, 'points_Harbin_2020.csv')
    samples_df = pd.read_csv(input_file)

    output_dir = os.path.join(root_folder, 'timeseries_plots_target')
    os.makedirs(output_dir, exist_ok=True)

    for index, row in samples_df.iterrows():
        sample_number = row['sample_number']
        longitude = row['longitude']
        latitude = row['latitude']


        print(f"Processing sample {sample_number}: longitude {longitude}, latitude {latitude}")

        output_file_path = os.path.join(output_dir,
                                       f'snic_cluster_{optical_index.lower()}_{radar_index.lower()}_timeseries_sample_{sample_number}.jpg')

        if os.path.exists(output_file_path):
            print(f"Output file already exists for sample {sample_number}, skipping")
            continue

        # Try to load data from local storage
        df_for_plot = load_local_data(sample_number, output_dir, optical_index, radar_index)

        if df_for_plot is None:
            # If local data doesn't exist, fetch from GEE
            print(f"Local data not found, fetching data from GEE for sample {sample_number}")
            point = ee.Geometry.Point([longitude, latitude])
            s2_timeseries, s1_timeseries = getTimeSeries(point, target_year, optical_index, radar_index,
                                                        use_cluster=True)
            df_for_plot, df_for_save = clean_and_process_df(s2_timeseries, s1_timeseries, optical_index, radar_index)
            # Save data locally
            save_to_csv(df_for_save, sample_number, output_dir, optical_index, radar_index)
        else:
            print(f"Using local data to plot sample {sample_number}")

        # Plot original time series
        plot_timeseries(df_for_plot, sample_number, output_dir, output_file_path,
                       optical_index, radar_index)

        # Plot smoothed time series
        plot_smooth_timeseries(df_for_plot, sample_number, output_dir, output_file_path,
                              optical_index, radar_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SNIC segmentation and plot time series for selected indices.')
    parser.add_argument('--optical_index', type=str, default='EVI',
                       help='Optical index to plot (e.g., NDVI, RVI, EVI)')
    parser.add_argument('--radar_index', type=str, default='RDVI',
                       help='Radar index to plot (e.g., RDVI, DPSVI, RFDI, VSI)')
    args = parser.parse_args()

    main(optical_index=args.optical_index, radar_index=args.radar_index)
