"""
Computer Vision Image Analyzer
Emergency Triage AI
"""

from PIL import Image
import numpy as np


def analyze_medical_image(image_file):
    """
    Basic image analysis for educational emergency triage project.
    Detects simple visual indicators such as redness, darkness, and brightness.
    """

    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)

    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]

    avg_red = float(np.mean(red_channel))
    avg_green = float(np.mean(green_channel))
    avg_blue = float(np.mean(blue_channel))
    avg_brightness = float(np.mean(img_array))

    redness_score = avg_red - ((avg_green + avg_blue) / 2)

    flags = []

    if redness_score > 35:
        flags.append("High redness detected - possible inflammation, bleeding, burn, or rash.")

    if avg_brightness < 70:
        flags.append("Dark image detected - image quality may be poor.")

    if avg_brightness > 210:
        flags.append("Very bright image detected - possible overexposure.")

    if not flags:
        flags.append("No major visual risk pattern detected by basic image analysis.")

    return {
        "image_size": image.size,
        "avg_red": round(avg_red, 2),
        "avg_green": round(avg_green, 2),
        "avg_blue": round(avg_blue, 2),
        "avg_brightness": round(avg_brightness, 2),
        "redness_score": round(redness_score, 2),
        "visual_flags": flags,
        "clinical_note": "Computer vision output is only a support feature. Final decision must be made by clinician."
    }