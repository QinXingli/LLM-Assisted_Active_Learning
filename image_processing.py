#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import ee
import os

# Remember to change to your own proxy address
os.environ['HTTP_PROXY'] = 'http://XXXX'
os.environ['HTTPS_PROXY'] = 'http://XXXX'

# Initialize Earth Engine
ee.Initialize()

# Define Sentinel-2 and Sentinel-1 data processing functions
def sentinel2toa(img):
    return img.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
        ['aerosol', 'blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'h2o', 'cirrus', 'swir1', 'swir2']
    ) \
        .divide(10000).toDouble() \
        .set('solar_azimuth', img.get('MEAN_SOLAR_AZIMUTH_ANGLE')) \
        .set('solar_zenith', img.get('MEAN_SOLAR_ZENITH_ANGLE')) \
        .set('system:time_start', img.get('system:time_start'))


def cloudMask(toa):
    def rescale(img, thresholds):
        return img.subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

    score = ee.Image(1)
    score = score.min(rescale(toa.select(['blue']), [0.1, 0.5]))
    score = score.min(rescale(toa.select(['aerosol']), [0.1, 0.3]))
    score = score.min(rescale(toa.select(['aerosol']).add(toa.select(['cirrus'])), [0.15, 0.2]))
    score = score.min(rescale(toa.select(['red']).add(toa.select(['green'])).add(toa.select('blue')), [0.2, 0.8]))

    ndmi = toa.normalizedDifference(['red4', 'swir1'])
    score = score.min(rescale(ndmi, [-0.1, 0.1]))

    ndsi = toa.normalizedDifference(['green', 'swir1'])
    score = score.min(rescale(ndsi, [0.8, 0.6]))

    cloudScoreThreshold = 0.3
    cloud = score.gt(cloudScoreThreshold)

    mask = cloud.eq(0)
    return toa.updateMask(mask)


def addVariables(image):
    DOY = image.date().getRelative('day', 'year')
    year = image.date().get('year')

    GCVI = image.expression('(NIR / GREEN) - 1', {
        'NIR': image.select('nir'),
        'GREEN': image.select('green')
    }).float().rename('GCVI')

    GLI = image.expression('(2 * GREEN - RED - BLUE) / (2 * GREEN + RED + BLUE)', {
        'GREEN': image.select('green'),
        'RED': image.select('red'),
        'BLUE': image.select('blue')
    }).float().rename('GLI')

    DVI = image.expression('NIR - RED', {
        'NIR': image.select('nir'),
        'RED': image.select('red')
    }).float().rename('DVI')

    # 添加比值植被指数 (RVI)
    RVI = image.expression('NIR / RED', {
        'NIR': image.select('nir'),
        'RED': image.select('red')
    }).float().rename('RVI')

    return image \
        .addBands(image.normalizedDifference(['nir', 'red']).toDouble().rename('NDVI')) \
        .addBands(image.expression('2.5*((nir-red)/(nir+6*red-7.5*blue+1))', {
        'nir': image.select('nir'),
        'red': image.select('red'),
        'blue': image.select('blue')
    }).toDouble().rename('EVI')) \
        .addBands(image.normalizedDifference(['nir', 'swir1']).toDouble().rename('LSWI')) \
        .addBands(GCVI) \
        .addBands(GLI) \
        .addBands(DVI) \
        .addBands(RVI)  # 添加RVI到返回的图像中


def normalizeRadar(image):
    vh = image.select('VH')
    vv = image.select('VV')
    angle = image.select('angle')

    vh_norm = vh.subtract(angle.multiply(ee.Number(3.14159).divide(180.0)).cos().log10().multiply(10.0))
    vv_norm = vv.subtract(angle.multiply(ee.Number(3.14159).divide(180.0)).cos().log10().multiply(10.0))

    return image.addBands(vh_norm.rename('VH_norm')).addBands(vv_norm.rename('VV_norm'))


def addS1Variables(image):
    vv = image.select('VV_norm')
    vh = image.select('VH_norm')

    # Original indices
    vv_vh_ratio = vv.divide(vh).rename('VV_VH_ratio')
    vv_vh_norm_diff = vv.subtract(vh).divide(vv.add(vh)).rename('VV_VH_norm_diff')
    rdvi = vh.multiply(4).divide(vv.add(vh)).rename('RDVI')  # Renamed to RDVI to avoid conflict with optical RVI

    # Additional indices
    # 1. Normalized Radar Polarization Backscatter (NRPB)
    nrpb = vh.divide(vv).rename('NRPB')

    # 2. Radar Ratio Vegetation Index (RRVI)
    rrvi = vv.divide(vh).rename('RRVI')

    # 3. Dual-Polarization SAR Vegetation Index (DPSVI)
    dpsvi = vv.multiply(vh).divide(vv.add(vh)).rename('DPSVI')

    # 4. Polarization Difference Index (PDI)
    pdi = vv.subtract(vh).rename('PDI')

    # 5. Radar Forest Degradation Index (RFDI)
    rfdi = vv.subtract(vh).divide(vv.add(vh)).rename('RFDI')

    # 6. Volume Scattering Index (VSI)
    vsi = vh.divide(vv.add(vh)).rename('VSI')

    return image.addBands(vv_vh_ratio) \
        .addBands(vv_vh_norm_diff) \
        .addBands(rdvi) \
        .addBands(nrpb) \
        .addBands(rrvi) \
        .addBands(dpsvi) \
        .addBands(pdi) \
        .addBands(rfdi) \
        .addBands(vsi)
