#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import base64
import logging

def encode_image(image_path: str, resample: bool = True, scale_factor: float = 0.75, jpeg_quality: int = 85) -> str:
    """
    Read and encode image, optionally perform resampling, convert PNG to JPEG with compression
    Args:
        image_path: Path to the image
        resample: Whether to perform resampling, defaults to True
        scale_factor: Resampling scale factor, defaults to 0.75
        jpeg_quality: JPEG compression quality (1-100), defaults to 85
                     - Lower value = Smaller file size but lower quality
                     - Higher value = Better quality but larger file size

    Returns:
        str: Base64 encoded image string
    """
    try:
        from PIL import Image
        import io
        import base64
        import os

        # 获取输入图像的格式
        img_format = os.path.splitext(image_path)[1].lower()

        # Get input image format
        img_format = os.path.splitext(image_path)[1].lower()

        # Open original image
        with Image.open(image_path) as img:
            # Ensure image is in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if resample:
                # Calculate new dimensions
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)

                # Perform resampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save image to byte stream
            img_byte_arr = io.BytesIO()

            # If PNG format, convert to JPEG and compress
            if img_format == '.png':
                # Enable optimization
                img.save(img_byte_arr,
                         format='JPEG',
                         quality=jpeg_quality,
                         optimize=True,
                         dpi=(500, 500))
            else:
                # Keep original format
                img.save(img_byte_arr, format='JPEG', dpi=(500, 500))

            img_byte_arr = img_byte_arr.getvalue()

            # Print size comparison before and after compression
            original_size = os.path.getsize(image_path) / (1024 * 1024)  # Convert to MB
            compressed_size = len(img_byte_arr) / (1024 * 1024)  # Convert to MB
            print(f"Original size: {original_size:.2f}MB")
            print(f"Compressed size: {compressed_size:.2f}MB")
            print(f"Compression ratio: {compressed_size / original_size:.2%}")

        # Return base64 encoding
        return base64.b64encode(img_byte_arr).decode('utf-8')

    except Exception as e:
        print(f"Error processing image file: {str(e)}")
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return ""

def validate_and_encode_image(image_path: str, resample: bool = True, scale_factor: float = 0.75) -> str:
    """
    Validate and encode image file
    Args:
        image_path: Path to the image
        resample: Whether to perform resampling, defaults to True
        scale_factor: Resampling scale factor, defaults to 0.75

    Returns:
        str: Base64 encoded image string
    """
    try:
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return ""

        if not os.path.isfile(image_path):
            logging.error(f"Path is not a file: {image_path}")
            return ""

        encoded = encode_image(image_path, resample, scale_factor)
        if encoded:
            return encoded
        else:
            logging.error(f"Failed to encode image: {image_path}")
            return ""

    except Exception as e:
        logging.error(f"Error in validate_and_encode_image: {str(e)}")
        return ""