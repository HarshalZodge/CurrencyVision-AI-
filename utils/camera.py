"""
Camera Utility Module for CurrencyVision AI.
Provides live webcam image capture processing using Streamlit camera input.
"""

from typing import Optional
from PIL import Image
import streamlit as st


def capture_webcam_image(key: str = "webcam_input") -> Optional[Image.Image]:
    """
    Renders Streamlit camera input widget and converts uploaded camera photo to PIL Image.

    Args:
        key (str): Streamlit unique widget key.

    Returns:
        Optional[PIL.Image.Image]: Captured image object or None if no image taken.
    """
    st.markdown("### 📷 Real-Time Webcam Currency Capture")
    st.caption("Align your Indian currency note clearly inside the camera preview frame and click 'Take Photo'.")

    camera_photo = st.camera_input(
        label="Capture Currency Note",
        key=key,
        help="Point camera directly at the note with clear lighting.",
    )

    if camera_photo is not None:
        try:
            image = Image.open(camera_photo)
            return image
        except Exception as e:
            st.error(f"Error reading image from webcam: {str(e)}")
            return None

    return None
