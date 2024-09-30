from PIL import Image
import torch
from Constant.constant import yolo_model, mapping

class YOLO:

    def __init__(self):
        pass

    def process_yolo(self, image):
        """Processes the provided NumPy array through the YOLO model."""
        try:
            print("(329) YOLO Processing Try")

            # Convert NumPy array back to PIL Image
            image = Image.fromarray(image)

            results = yolo_model(image)

            if len(results[0].boxes) == 0:
                del(image)
                return "Accepted", None, None, None

            conf = torch.max(results[0].boxes.conf).item()

            if conf < 0.8:
                del(image)
                return "Accepted", None, conf, None

            z = torch.argmax(results[0].boxes.conf).item()
            a = int(results[0].boxes.cls[z].item())
            detected_class = mapping[a]
            print(f"        YOLO Class: {detected_class} and Confidence: {conf}")

            if a == 2:  # Eyeglasses
                print("(347) YOLO Eyeglass Acceptance")
                del(image)
                return "Accepted", None, conf, detected_class

            del(image)
            return "Rejected", detected_class, conf, detected_class

        except Exception as e:
            del(image)
            print("(352) YOLO Exception")
            return "Rejected", f"{e}", None, None