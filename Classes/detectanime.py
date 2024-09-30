import sys
import os
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Constant.constant import model,processor


class Animatedimage:

    def __init__(self):
        pass

    def check_if_cartoon(self,image):
        try:
            image = Image.fromarray(image)
            inputs = processor(text=["image of a real person", "animated image or image of cartoon image"], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)
            result_index = torch.argmax(probs)
            animated_confidence = probs[0][result_index]
            print(f"Anime Confidence: {animated_confidence}")
            if result_index >= 1 and animated_confidence > 0.78:
                print("Animated Image Detected")
                return "Cartoon", None
            else:
                del(image)
                return "Real", None
        except Exception as e:
            del(image)
            print("Animated Image Exception")
            return None, str(e)