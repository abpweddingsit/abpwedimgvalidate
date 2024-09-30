import requests
from PIL import Image
from io import BytesIO
import numpy as np

class ImageURL:

    def __init__(self):
        pass

    def image_url_to_array(self, image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an error for bad responses
            image = Image.open(BytesIO(response.content))
            return np.array(image), None

        except Exception as e:
            print(f"Error downloading or processing the image: {e}")
            return None, str(e)