#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import json
from collections import Counter
import csv
import os

def extract_crop_classification(response_content):
    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            return json_data
        except json.JSONDecodeError:
            print("Failed to parse JSON from the response")
    else:
        print("No JSON found in the response")

    return {
        "crop_type": "Unknown",
        "confidence": 0,
        "reasoning": "Failed to extract information from the response"
    }

def process_results(results):
    if len(results) < 2:
        return {"crop_type": "Unknown", "confidence": 0, "alternative_types": [],
                "reasoning": "Insufficient successful classifications"}

    crop_types = [result['crop_type'] for result in results]
    crop_type_counts = Counter(crop_types)
    most_common_crop_type, count = crop_type_counts.most_common(1)[0]

    if count >= 2:
        matching_results = [result for result in results if result['crop_type'] == most_common_crop_type]
        return matching_results[0]  # We can choose any of the matching results
    else:
        return {"crop_type": "Unknown", "confidence": 0, "alternative_types": [],
                "reasoning": "No consistent classification across attempts"}

def write_results_to_csv(output_dir, sample_number, final_result, results=[]):
    csv_file_path = os.path.join(output_dir, f"classification_results_sample_{sample_number}.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        if not results:
            fieldnames = ['Sample', 'Final_Crop_Type', 'Final_Confidence']
        else:
            fieldnames = ['Sample', 'Final_Crop_Type', 'Final_Confidence', 'Attempt1_Crop_Type', 'Attempt2_Crop_Type',
                          'Attempt3_Crop_Type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        if not results:
            writer.writerow({
                'Sample': sample_number,
                'Final_Crop_Type': final_result['crop_type'],
                'Final_Confidence': final_result['confidence']
            })
        else:
            writer.writerow({
                'Sample': sample_number,
                'Final_Crop_Type': final_result['crop_type'],
                'Final_Confidence': final_result['confidence'],
                'Attempt1_Crop_Type': results[0]['crop_type'] if len(results) > 0 else "Unknown",
                'Attempt2_Crop_Type': results[1]['crop_type'] if len(results) > 1 else "Unknown",
                'Attempt3_Crop_Type': results[2]['crop_type'] if len(results) > 2 else results[1]['crop_type'] if len(results) > 1 else "Unknown"
            })

    print(f"Results saved to {csv_file_path}")
