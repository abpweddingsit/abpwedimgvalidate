import cv2
import torch
from Constant.constant import NSFWModel, NSFWProcessor

class DetectNSFW:

    def __init__(self):
        pass
    
    def detect_nsfw(self, image):        
        try:
            print("(114) NSFW Try")
            # Ensure the image is in the right format for OpenCV
            img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR) if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            with torch.no_grad():
                inputs = NSFWProcessor(images=img, return_tensors="pt")
                outputs = NSFWModel(**inputs)
                logits = outputs.logits
            
            predicted_label = logits.argmax(-1).item()
            label = NSFWModel.config.id2label[predicted_label]
            confidence = torch.softmax(logits, dim=-1)[0][predicted_label].item()
            print(f"      NSFW: {label} Confidence: {confidence}")

            if label == 'nsfw':
                del(image,img)
                return 'Image contains NSFW content', confidence
            else:
                del(image,img)
                return None, confidence
        except Exception as e:
            del(image,img)
            print("(133) NSFW Except")
            return str(e), None