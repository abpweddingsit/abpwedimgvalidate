import os
import shutil
import base64
import cv2
import torch
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoModelForImageClassification, ViTImageProcessor
import clip
from ultralytics import YOLO
import mediapipe as mp
import face_recognition
from insightface.app import FaceAnalysis
import numpy as np


BASE_FOLDER = "/home/abp/Documents/ABPProduction/ABP/ProfileModeration/Version11/API/Demo" 
YOLO_FOLDER = "/home/abp/Documents/ABPProduction/ABP/ProfileModeration/Version11/best.pt"

# Creating the BASE FOLDER
if not os.path.exists(BASE_FOLDER):
    os.makedirs(BASE_FOLDER)

# INSIGHT-FACE
try:
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
except Exception as e:
    print(f"(30) Error Loading InsightFace Model: {e}")
    app = None

# MEDIAPIPE
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.0,
    min_tracking_confidence=0.90
)

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
text = ["a cap", "a hat", "a sunglass", "a helmet", "a reading glass", "a mask"]
text_tokens = clip.tokenize(text).to(device)

# CLIP RN101 Model
RNmodel, RNpreprocess = clip.load("RN101", device=device)
rn101textlist = ["a sunglass", "a reading glass"]
rn101text = clip.tokenize(rn101textlist).to(device)

# YOLO
yolo_model = YOLO(YOLO_FOLDER)
mapping = {0 : "sunglasses", 1 : "sunglasses", 2 : "eyeglasses", 3 : "headware", 4 : "headware", 5 : "headware"}

# Coversion to Image
def base64_to_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        print("(64) Base64 Try")
        return image, None
    except Exception as e:
        print("(67) Base64 Except")
        return None, str(e)
    

# Saving the Converted Image
def save_image(image, image_name):
    try:
        image_path = os.path.join(BASE_FOLDER, image_name)
        Image.fromarray(image).save(image_path)
        print("(76) Image Saving Successfull")
        return image_path, None
    except Exception as e:
        print("(79) Image Saving Failed")
        return None, str(e)

# NSFW
def detect_nsfw(image):
    try:
        print("(85) NSFW Try")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_label].item()
        if label == 'nsfw':
            return 'Image contains NSFW content', confidence
        else:
            return None, confidence
    except Exception as e:
        print("(101) NSFW Except")
        return str(e), None

# Cropping the Face from the Image (If Face Exists {Face Recognition})
def crop_faces(image, output_dir, expansion_factor=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        print("(109) Face Recognition Try")
        image = np.array(image)
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) != 1:
            return False, 'Multiple faces detected' if len(face_locations) > 1 else 'No Face Detected'
        
        pil_image = Image.fromarray(image)
        base_name = 'face.png'
        name, ext = os.path.splitext(base_name)
        top, right, bottom, left = face_locations[0]
        height, width, _ = image.shape
        expansion_width = int((right - left) * expansion_factor)
        expansion_height = int((bottom - top) * expansion_factor)
        new_top = max(0, top - expansion_height)
        new_bottom = min(height, bottom + expansion_height)
        new_left = max(0, left - expansion_width)
        new_right = min(width, right + expansion_width)
        face_image = pil_image.crop((new_left, new_top, new_right, new_bottom))
        face_path = os.path.join(output_dir, f"{name}{ext}")
        face_image.save(face_path)
        return True, None
    except Exception as e:
        print("(132) Face Recognition Exception")
        return False, str(e)

# Saving the Largest Face (Multiple Faces)
def save_face(largestface, image, output_dir, expansion_factor=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        print("(140) Largest Face Try")
        image = np.array(image)
        pil_image = Image.fromarray(image)
        base_name = 'face.png'
        name, ext = os.path.splitext(base_name)
        left, top, right, bottom = largestface.bbox
        height, width, _ = image.shape
        expansion_width = int((right - left) * expansion_factor)
        expansion_height = int((bottom - top) * expansion_factor)
        new_top = max(0, top - expansion_height)
        new_bottom = min(height, bottom + expansion_height)
        new_left = max(0, left - expansion_width)
        new_right = min(width, right + expansion_width)
        face_image = pil_image.crop((new_left, new_top, new_right, new_bottom))
        face_path = os.path.join(output_dir, f"{name}{ext}")
        face_image.save(face_path)
        return True, None
    except Exception as e:
        print("(158) Largest Face Except")
        return False, str(e)

# Insight Face Processing
def check_image(image_path):
    try:
        print("(164) Insight Face Processing Try")
        img = cv2.imread(image_path)
        faces = app.get(img)
        
        if len(faces) == 1:
            success, error = crop_faces(Image.open(image_path), 'TempFaces')
            if success:
                return 'Accepted', None
            else:
                return 'Rejected', error if error else 'Face cropping failed'
        elif len(faces) > 1:
            areas = []
            largestfacearea = 0
            largestfaceindex = None
            for index, face in enumerate(faces):
                bbox = face.bbox
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                areas.append(area)
                if area > largestfacearea:
                    largestfaceindex = index
                    largestfacearea = area
            areas.sort(reverse=True)
            area_difference = (areas[0] - areas[1]) / areas[0]
            if area_difference > 0.80:
                largestface = faces[largestfaceindex]
                success, error = save_face(largestface, Image.open(image_path), "TempFaces")
                if success:
                    return 'Accepted', None
                else:
                    return 'Rejected', error if error else 'Face cropping failed'
            else:
                return 'Rejected', 'Multiple faces detected'
        else:
            confidence_scores = [face.det_score for face in faces] if faces else []
            error_msg = f"No Face Detected. Face Confidence Score: {confidence_scores[0]}" if confidence_scores else "No Face Detected"
            return 'Rejected', error_msg
    except Exception as e:
        print("(201) Insight Face Processing Exception")
        return 'Rejected', str(e)

# MediaPipe Processing
def detect_landmarks(image):
    print("(206) Mediapipe Processing")
    results = mp_face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]

# Mediapipe Face Processing
def process_single_image(image):
    try:
        print("(215) Mediapipe Single Image Processing")
        image_top = image[:image.shape[0] // 2, :]
        image_bottom = image[image.shape[0] // 2:, :]
        landmarks_top = detect_landmarks(image_top)
        landmarks_bottom = detect_landmarks(image_bottom)
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
        print("(233) Mediapipe Single Image Exception")
        return 'Rejected', str(e)


# CLIP Processing
def process_image_clip(image):
    try:
        print("(240) CLIP B32 Processing Try")
        image = Image.fromarray(image)  # Converts NumPy array to PIL Image
        image = preprocess(image).unsqueeze(0).to(device)  # Converts PIL Image to Tensor
        
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        predicted_index = probs.argmax()
        confidence = probs[0][predicted_index]
        detected_class = text[predicted_index]
        print(f"B32 Detected Class: {detected_class} and Confidence: {confidence}")
        
        if confidence > 0.5 and (detected_class == "a sunglass" or detected_class == "a reading glass"):            
            if detected_class in ["a sunglass", "a reading glass"]:
                rn101_image = RNpreprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    rn101_logits_per_image, rn101_logits_per_text = RNmodel(rn101_image, rn101text)
                    rn101_probs = rn101_logits_per_image.softmax(dim=-1).cpu().numpy()
                rn101_predicted_index = rn101_probs.argmax()
                rn101_confidence = rn101_probs[0][rn101_predicted_index]
                print(f"RN101 Confidence: {rn101_confidence}")
                print(f"Predicted Index RN101: {rn101_predicted_index}")
                if rn101_confidence > 0.5 and rn101textlist[rn101_predicted_index] == "a reading glass":
                    print("Accepted by RN101")
                    return "Accepted", None, confidence, detected_class
                else:
                    return "Rejected", "RN101 model did not confirm 'reading glass' with sufficient confidence", confidence, detected_class
            else:
                return "Rejected", f"Error: {detected_class}", confidence, detected_class
        
        elif confidence > 0.8:
            return "Rejected", f"Error: {detected_class}", confidence, detected_class
        
        else:
            return "Accepted", None, confidence, detected_class
        
    except Exception as e:
        print("(276) CLIP Processing Exception")
        return 'Rejected', str(e), 0, None

# YOLO Processing
def process_yolo(image):
    try:
        print("(282) YOLO Processing Try")
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
        
        if a == 2:                                          # Class 2 is Eyeglasses
            print("(299) YOLO Eyeglass Acceptance")
            return "Accepted", None, conf, detected_class
        return "Rejected", detected_class, conf, detected_class
    
    except Exception as e:
        print("(304) YOLO Exception")
        return "Rejected", f"{e}", None, None

# Final Result
def get_result(base64_image):
    final_result = ""
    errstring = ""
    confidence_scores = {}
    status = 0  # Default status is 0 (Rejected)

    # Convert base64 to image and save it as image.png
    image, error = base64_to_image(base64_image)
    
    if error:
        return 'Rejected', error, confidence_scores, status
    
    image_path, error = save_image(image, 'image.png')
    
    if error:
        return 'Rejected', error, confidence_scores, status
    
    # Process the saved image
    Result1, error1 = check_image(image_path)
    
    if Result1 == 'Rejected':
        final_result = "Rejected"
        errstring += error1
    else:
        Result2, errormedia = process_single_image(image)
        Result3, errorclip, clip_confidence, detected_class = process_image_clip(image)
        Result4, erroryolo, yolo_confidence, yolo_class = process_yolo(image)
        errornsfw, nsfw_confidence = detect_nsfw(image)
        
        clip_confidence = float(clip_confidence) if clip_confidence is not None else 0.0
        yolo_confidence = float(yolo_confidence) if yolo_confidence is not None else 0.0
        nsfw_confidence = float(nsfw_confidence) if nsfw_confidence is not None else 0.0
        
        # Combined Result
        print("\n\nCOMBINED RESULT:")
        print(f"- \n Insight Face Result: {Result1}, \n Media pipe Result: {Result2}, \n Clip B/32 Result: {Result3}, \n yolo Result: {Result4}, \n NSFW Result: {'Rejected' if errornsfw else 'Accepted'}.")
        print(f"\n Insight Face Error: {error1}, \n Media pipe Error: {errormedia}, \n Clip B/32 Error: {errorclip}, \n yolo error: {erroryolo}, \n NSFW error: {errornsfw}.\n")

        confidence_scores['CLIP B32'] = {
            "Confidence": clip_confidence,
            "Detected Class": detected_class
        }
        
        confidence_scores['YOLO'] = {
            "Confidence": yolo_confidence,
            "Detected Class": yolo_class
        }
        
        if errornsfw:
            confidence_scores['NSFW'] = nsfw_confidence
            errstring += f"NSFW content detected: {errornsfw}. "
            final_result = "Rejected"
            status = 0
            return f"Final Result: {final_result}", errstring, confidence_scores, status
        
        accepted_count = sum([Result2 == 'Accepted', Result3 == 'Accepted', Result4 == 'Accepted'])
        
        if accepted_count >= 2:
            final_result = "Accepted"
            status = 1
        elif errorclip is None and erroryolo == "sunglasses":
            final_result = "Accepted"
            status = 1
        else:
            final_result = "Rejected"
            status = 0
            
            if error1 is not None:
                errstring += "No Face or multiple faces present or Face clearly not visible or The URL is unreachable. "
            if errormedia is not None:
                errstring += "Facial Features clearly not visible. "
            if errorclip is not None:
                errstring += "Eyewear or headwear detected. "
            if erroryolo is not None:
                errstring += "Eyewear or headwear detected. "
    
    # Clean up temporary folders
    for folder in ['TempFaces']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    # Returning Final Result
    return f"Final Result: {final_result}", errstring, confidence_scores, status