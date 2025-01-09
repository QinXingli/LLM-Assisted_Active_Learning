#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from openai import OpenAI
# from . import param_config as config
from API_Process import param_config as config
import httpx


client = OpenAI(
  base_url=config.BASE_URL,
  api_key=config.API_KEY,
  http_client=httpx.Client(
    base_url=config.BASE_URL,
    follow_redirects=True,
  ),
)


def read_super_prompt():
    file_path = config.SUPER_PROMPT_FILE
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading the super prompt file: {e}")
        return None


def get_classification(standard_encoded_image, unknown_encoded_image, prompt):
    super_prompt = read_super_prompt()
    system_prompt = """
    You are an expert agricultural remote sensing analyst with the following specialized knowledge and capabilities:

    1. Professional Background:
    - Ph.D. in Agricultural Remote Sensing
    - 15+ years experience in crop monitoring and classification
    - Deep expertise in vegetation index analysis
    - Specialized knowledge of crop phenology in Northeast China
    
    2. Core Competencies:
    - Time series analysis of vegetation indices (EVI, RDVI)
    - Crop growth pattern recognition
    - Phenological stage identification
    - Multi-temporal data interpretation
    - Agricultural systems in Heilongjiang Province
    
    3. Analysis Approach:
    - Systematic and thorough
    - Data-driven decision making
    - Clear documentation of reasoning
    - Conservative in uncertainty assessment
    - Comprehensive pattern analysis
    
    4. Domain Knowledge:
    - Growing patterns of major crops (Maize, Soybean, Rice)
    - Regional cropping calendars
    - Year-to-year variability factors
    - Environmental impacts on crop growth
    - Limitations of remote sensing data
    
    5. Communication Style:
    - Clear and structured analysis
    - Professional yet accessible language
    - Explicit reasoning chains
    - Balanced consideration of evidence
    - Transparent about uncertainty
    
    Your task is to analyze vegetation index time series data to identify crop types, following the detailed protocol provided below.

    """
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{standard_encoded_image}"
            }
        },

        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{unknown_encoded_image}"
            }
        },
        {
            "type": "text",
            "text": prompt
        },

    ]
    try:
        completion = client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=4096,
            temperature=1,
            stream=True
        )
        # return completion.choices[0].message.content
        # Process streaming response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content

        return full_response

    except Exception as e:
        print(f"An error occurred during API call: {str(e)}")
        return None