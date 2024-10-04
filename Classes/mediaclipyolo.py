import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Constant.constant import mp_face_mesh,preprocess,device,clip_model,text_tokens,RNmodel,rn101text,rn101textlist,text,yolo_model,mapping

class MediaPipeClipYolo:

    def __init__(self):
        pass
    
    # Mediapipe
    def detect_landmarks(self, image):
        print("(249) Mediapipe Processing")
        results = mp_face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def process_single_image(self,image):
        
        try:
            print("(258) Mediapipe Single Image Processing")
            image_top = image[:image.shape[0] // 2, :]
            image_bottom = image[image.shape[0] // 2:, :]
            landmarks_top = self.detect_landmarks(image_top)
            landmarks_bottom = self.detect_landmarks(image_bottom)
            top_face_detected = landmarks_top is not None
            bottom_face_detected = landmarks_bottom is not None
            Result2 = 'Accepted' if top_face_detected and bottom_face_detected else 'Rejected'
            error_message = ""
            if Result2 == 'Rejected':
                if not top_face_detected:
                    error_message += "Top Face Error; "
                if not bottom_face_detected:
                    error_message += "Bottom Face Error; "
                error_message = error_message.rstrip("; ")
            return Result2, error_message
        
        except Exception as e:
            print("(276) Mediapipe Single Image Exception")
            return 'Rejected', str(e)

    # CLIP
    def process_image_clip(self,image):
        try:
            print("(243) CLIP B32 Processing Try")
            image = Image.fromarray(image)  # Converts NumPy array to PIL Image
            B32image = preprocess(image).unsqueeze(0).to(device)  # Converts PIL Image to Tensor
            
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(B32image, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            predicted_index = probs.argmax()
            confidence = probs[0][predicted_index]
            detected_class = text[predicted_index]
            print(f"\nB32 Detected Class: {detected_class} and Confidence: {confidence}\n")
            
            if confidence > 0.5 and (detected_class == "a sunglass" or detected_class == "a reading glass"):            
                if detected_class in ["a sunglass", "a reading glass"]:
                    print("(298) CLIP RN101 Processing Try")

                    with torch.no_grad():
                        rn101_logits_per_image, rn101_logits_per_text = RNmodel(B32image, rn101text)
                        rn101_probs = rn101_logits_per_image.softmax(dim=-1).cpu().numpy()
                    rn101_predicted_index = rn101_probs.argmax()
                    rn101_confidence = rn101_probs[0][rn101_predicted_index]
                    RNdetected_class = rn101textlist[rn101_predicted_index]
                    print(f"\nRN101 Confidence: {rn101_confidence} Predicted Class RN101: {RNdetected_class}\n")

                    if rn101_confidence > 0.5 and rn101textlist[rn101_predicted_index] == "a reading glass":
                        print("Accepted by RN101 for Eyeglasses")
                        return "Accepted", None, confidence, detected_class
                    else:
                        return "Rejected", f"Error: {detected_class}", confidence, detected_class
                else:
                    return "Rejected", f"Error: {detected_class}", confidence, detected_class
            
            elif confidence > 0.8:                                                              # Rejection for Headware
                return "Rejected", f"Error: {detected_class}", confidence, detected_class
            
            else:
                return "Accepted", None, confidence, detected_class
            
        except Exception as e:
            print("(323) CLIP Processing Exception")
            return 'Rejected', str(e), 0, None
        
    # YOLO
    def process_yolo(self,image):
        try:
            print("(329) YOLO Processing Try")
            image = Image.fromarray(image)
            results = yolo_model(image)
            
            if len(results[0].boxes) == 0:
                return "Accepted", None, None, None
            
            conf = torch.max(results[0].boxes.conf).item()
            
            if conf < 0.8:
                return "Accepted", None, conf, None
            
            z = torch.argmax(results[0].boxes.conf).item()
            a = int(results[0].boxes.cls[z].item())
            detected_class = mapping[a]
            print(f"YOLO Class: {detected_class} and Confidence: {conf}")
            
            if a == 2:                                          # Eyeglasses
                print("(347) YOLO Eyeglass Acceptance")
                return "Accepted", None, conf, detected_class
            return "Rejected", detected_class, conf, detected_class
        
        except Exception as e:
            print("(352) YOLO Exception")
            return "Rejected", f"{e}", None, None
