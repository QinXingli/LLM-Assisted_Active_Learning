#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import glob
import re
import json
import csv
from image_encode import validate_and_encode_image
from api_client import get_classification
from recognize_result_processor_transfer import extract_crop_classification, process_results, write_results_to_csv
import param_config as config


def process_sample(standard_image_path, unknown_image_path, output_dir, sample_number):
    standard_encoded_image = validate_and_encode_image(standard_image_path, resample=False)
    unknown_encoded_image = validate_and_encode_image(unknown_image_path,  resample=False)

    if not standard_encoded_image or not unknown_encoded_image:
        print(f"Failed to encode one or both images for sample {sample_number}")
        return


    prompt = f"""
    This is a crop type identification task for analyzing vegetation index time series data in Harbin, China. As an agricultural remote sensing expert, please analyze the provided data to determine the crop type.
    
    Input Data:
    1. Reference Data (2019) (The first image):
    - Standard EVI (Green line) and RDVI (Blue line) time series for three crop types:
      * Maize
      * Soybean
      * Rice
    - EVI (Enhanced Vegetation Index): Optical reflectance characteristics 
    - RDVI (Radar Vegetation Index): Radar backscatter characteristics
    - Note: Reference curves are averaged patterns from 2019
    
    2. Unknown Crop Data (2020) (The second image):
    - Time period: June 1 - October 30, 2020
    - Location: Harbin, Heilongjiang Province, China
    - Single sample time series
    
    Analysis Requirements:
    1. Temporal Pattern Analysis
    - In your analysis, you must explicitly identify and discuss the following key time points:
        1.1. The exact dates of the top three highest EVI values
        1.2. The exact dates of the top three lowest EVI values
        1.3. The exact dates of the top three highest RDVI values
        1.4. The exact dates of the top three lowest RDVI values
    - Analyze growth stages and phenological transitions
    - Consider potential phenological shifts between 2019-2020
    - Since the backscattering coefficient of rice will decrease significantly after the nodulation stage (late July). 
    - And rice absorbs radar waves better than corn and soybean, so overall, the magnitude of backscattering coefficients are ranked as: rice < maize < soybean
    
    2. Growth Characteristics Assessment
    - Growth rate during different phases
    - Peak timing and magnitude
    - Senescence patterns
    - Relationship between EVI and RDVI patterns
    - Overall growing season characteristics
    
    3. Comparative Analysis
    - Compare unknown sample with reference patterns of Maize, Soybean, and Rice
    - Consider:
      * Temporal alignment of key features
      * Relative magnitude of peaks
      * Growth rate similarities
      * Pattern matching accounting for year-to-year variations
    - If pattern doesn't match any reference crop, classify as "Others"
    
    4. Classification Framework
    - Primary classification options: Maize, Soybean, Rice, or Others
    - Use "Others" if:
      * Pattern significantly deviates from all three reference crops
      * Shows characteristics inconsistent with known crop types
    - Consider multiple factors - avoid single-feature classification
    
    Output Requirements:
    1. Detailed Analysis
    - Professional yet accessible explanation
    - Key time points and features identified
    - Clear reasoning process
    - Limitations or uncertainties noted
    
    2. Final Classification (JSON format):
    ```json
    {{
      "crop_type": "CropName",
      "confidence": "Number",
      "reasoning": "Brief explanation of your judgment rationale"
    }}
    ```
    
    Important Considerations:
    - Focus on pattern analysis over absolute values
    - Account for interannual variability (2019 vs 2020)
    - Consider both EVI and RDVI relationships
    - Document any data quality issues
    - Provide comprehensive justification for classification
    - crop_type must be one of the following: {config.CROP_TYPES}.
    - confidence must be "High", "Medium", "Low"

    """

    results = []
    for i in range(config.MAX_ATTEMPTS):
        print(f"\nAttempt {i + 1} for sample {sample_number}:")
        response = get_classification(standard_encoded_image, unknown_encoded_image, prompt)
        if response:
            result = extract_crop_classification(response)
            results.append(result)
            print(f"Classification result: {result['crop_type']}")
        else:
            print("Failed to get classification")

        if len(results) == 2 and results[0]['crop_type'] == results[1]['crop_type']:
            break

    final_result = process_results(results)

    print(f"\nFinal Classification Result for sample {sample_number}:")
    print(json.dumps(final_result, indent=2))

    write_results_to_csv(output_dir, sample_number, final_result, results)

def main():
    sample_images = glob.glob(os.path.join(config.SAMPLE_DIR, "snic_cluster_evi_rdvi_timeseries_sample_*_smoothed.jpg"))

    for sample_image in sample_images:
        sample_number = re.search(r'sample_(\d+)', sample_image).group(1)
        print(f"\nProcessing sample {sample_number}")

        output_dir = os.path.dirname(sample_image)
        output_dir = os.path.join(output_dir, "recognize_result_claude")
        os.makedirs(output_dir, exist_ok=True)

        csv_file_path = os.path.join(output_dir, f"classification_results_sample_{sample_number}.csv")
        if os.path.exists(csv_file_path):
            print(f"Classification results already exist for sample {sample_number}")
            continue

        process_sample(config.STANDARD_IMAGE_PATH, sample_image, output_dir, sample_number)

if __name__ == "__main__":
    main()