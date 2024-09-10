import os
import shutil
import base64
import cv2
import torch
from PIL import Image, ExifTags
from io import BytesIO
import requests
from transformers import AutoModelForImageClassification, ViTImageProcessor
import clip
from ultralytics import YOLO
import mediapipe as mp
import face_recognition
from insightface.app import FaceAnalysis
import numpy as np

# Warnings Ignore
import warnings
warnings.filterwarnings("ignore")

BASE_FOLDER = "Demo" 
YOLO_FOLDER = "best.pt"

# Creating the BASE FOLDER
if not os.path.exists(BASE_FOLDER):
    os.makedirs(BASE_FOLDER)

# INSIGHT-FACE
try:
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
except Exception as e:
    print(f"(33) Error Loading InsightFace Model: {e}")
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
    # print(f"BASE64: \n{base64_str}")

    try:
        # Decode Base64 string
        image_data = base64.b64decode(base64_str)
        
        # Open image using PIL
        image = Image.open(BytesIO(image_data))
        
        # Check and correct image orientation based on EXIF metadata
        try:
            # If the image has EXIF data, correct orientation
            exif = image._getexif()
            if exif:
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        if value == 3:
                            image = image.rotate(180, expand=True)
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                        elif value == 8:
                            image = image.rotate(90, expand=True)
                        break
        except Exception as exif_error:
            print(f"(87) EXIF correction failed: {exif_error}")
        
        # Convert image to RGB and then to NumPy array
        image = image.convert("RGB")
        image = np.array(image)
        
        print("(93) Base64 Try")
        return image, None
    except Exception as e:
        print("(96) Base64 Except")
        return None, str(e)
    

# Saving the Converted Image
def save_image(image, image_name):
    try:
        image_path = os.path.join(BASE_FOLDER, image_name)
        Image.fromarray(image).save(image_path)
        print("(105) Image Saving Successfull")
        return image_path, None
    except Exception as e:
        print("(108) Image Saving Failed")
        return None, str(e)

# NSFW
def detect_nsfw(image):
    try:
        print("(114) NSFW Try")
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

        print(f"\nNSFW: {label} Confidence: {confidence}\n")

        if label == 'nsfw':
            return 'Image contains NSFW content', confidence
        else:
            return None, confidence
    except Exception as e:
        print("(133) NSFW Except")
        return str(e), None

# Cropping the Face from the Image (If Face Exists {Face Recognition})
def crop_faces(image, output_dir, expansion_factor=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        print("(141) Face Recognition Try")
        image = np.array(image)
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) != 1:
            return False, 'Multiple faces detected' if len(face_locations) > 1 else 'No Face Detected'
        
        pil_image = Image.fromarray(image)
        base_name = 'face.jpg'
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
        print("(164) Face Recognition Exception")
        return False, str(e)

# Saving the Largest Face (Multiple Faces)
def save_face(largestface, image, output_dir, expansion_factor=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        print("(172) Largest Face Try")
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
        print("(190) Largest Face Except")
        return False, str(e)

# Insight Face Processing
def check_image(image_path):
    try:
        print("(196) Insight Face Processing Try")
        img = cv2.imread(image_path)
        # print(image_path)
        faces = app.get(img)
        
        if len(faces) == 1:
            success, error = crop_faces(Image.open(image_path), 'TempFaces')
            if success:
                return 'Accepted', None
            else:
                if error == "No Face Detected":
                    return "Rejected", 0
                elif error == "Multiple faces detected":
                    return 'Rejected', 1
                else:
                    return "Rejected", 2        #Face cropping failed'
                
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
                    return 'Rejected', 2
                    # return 'Rejected', error if error else 'Face cropping failed'
            else:
                return 'Rejected', 1
                # return 'Rejected', 'Multiple faces detected'
        else:
            confidence_scores = [face.det_score for face in faces] if faces else []
            error_msg = f"No Face Detected. Face Confidence Score: {confidence_scores[0]}" if confidence_scores else "No Face Detected"
            return 'Rejected', 0
            # return 'Rejected', error_msg
    except Exception as e:
        print("(243) Insight Face Processing Exception")
        return 'Rejected', 2
        # return 'Rejected', str(e)

# MediaPipe Processing
def detect_landmarks(image):
    print("(249) Mediapipe Processing")
    results = mp_face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]

# Mediapipe Face Processing
def process_single_image(image):
    try:
        print("(258) Mediapipe Single Image Processing")
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
        print("(276) Mediapipe Single Image Exception")
        return 'Rejected', str(e)


# CLIP Processing
def process_image_clip(image):
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

# YOLO Processing
def process_yolo(image):
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

    
def get_result(base64_image):
    final_result = ""
    errstring = ""
    confidence_scores = {}
    detected_classes = {}
    status = 0                                    # Default status being 0 (Rejected)

    # Convert base64 to image and save it as image.jpg
    image, error = base64_to_image(base64_image)
    
    # Error in Base64
    if error:
        return {
            "status": status,
            "DetectedClass": {
                "ID_1": 1.0,                        # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }
    
    image_path, error = save_image(image, 'image.jpg')
    
    # Error in Saving
    if error:
        return {
            "status": status,
            "Detected Class": {
                "ID_1": 1.0,                        # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }
    
    # Processing NSFW
    NSFW_String, NSFW_Confidence = detect_nsfw(image)
    if NSFW_String == "Image contains NSFW content":
        return {
            "status": status,
            "DetectedClass": {
                "ID_1": None,                       # Invalid Image
                "ID_2": NSFW_Confidence,            # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }
    
    # No Face
    Face_Result, Error_Code = check_image(image_path)
    if Face_Result == "Rejected":    
        if Error_Code == 0:
            return {
            "status": status,
            "DetectedClass": {
                "ID_1": None,                       # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": 1.0,                        # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }

        elif Error_Code == 1:
            return {
            "status": status,
            "DetectedClass": {
                "ID_1": None,                       # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": 1.0,                        # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }

        else:
            return {
            "status": status,
            "DetectedClass": {
                "ID_1": 1.0,                        # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }

    
    # CLIP YOLO    
    else:
        Result2, errormedia = process_single_image(image)
        Result3, errorclip, clip_confidence, detected_class = process_image_clip(image)
        Result4, erroryolo, yolo_confidence, yolo_class = process_yolo(image)
        # errornsfw, nsfw_confidence = detect_nsfw(image)
        
        clip_confidence = float(clip_confidence) if clip_confidence is not None else 0.0
        yolo_confidence = float(yolo_confidence) if yolo_confidence is not None else 0.0
        # nsfw_confidence = float(nsfw_confidence) if nsfw_confidence is not None else 0.0

        confidence_scores['CLIP B32'] = {
            "Confidence": clip_confidence,
            "Detected Class": detected_class
        }
        
        confidence_scores['YOLO'] = {
            "Confidence": yolo_confidence,
            "Detected Class": yolo_class
        }

        # # Combined Result
        # print("\n\nCOMBINED RESULT:")
        # print(f"- \n Insight Face Result: {Face_Result}, \n Media pipe Result: {Result2}, \n Clip B/32 Result: {Result3}, \n yolo Result: {Result4}, \n NSFW Result: {'Rejected' if errornsfw else 'Accepted'}.")
        # print(f"\n Insight Face Error: {error1}, \n Media pipe Error: {errormedia}, \n Clip B/32 Error: {errorclip}, \n yolo error: {erroryolo}, \n NSFW error: {errornsfw}.\n")

        accepted_count = sum([Result2 == 'Accepted', Result3 == 'Accepted', Result4 == 'Accepted'])
        
        if accepted_count >= 2:
            final_result= {
            "status": 1,
            "DetectedClass": {
                "ID_1": None,                       # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }

        elif errorclip is None and erroryolo == "sunglasses":
            final_result= {
            "status": 1,
            "DetectedClass": {
                "ID_1": None,                       # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": None,                       # Eye
                "ID_6": None,                       # Cap
            },
            "confidence_scores":{}
        }
            
        else:
            final_result= {
            "status": 0,
            "DetectedClass": {
                "ID_1": None,                       # Invalid Image
                "ID_2": None,                       # NSFW
                "ID_3": None,                       # No Face
                "ID_4": None,                       # Multiple Faces                       
                "ID_5": 1.0,                       # Eye
                "ID_6": 1.0,                       # Cap
            },
            "confidence_scores": confidence_scores
        }
    
    # Combined Result
    print("\n\nCOMBINED RESULT:")
    print(f"- \n Insight Face Result: {Face_Result}, \n Media pipe Result: {Result2}, \n Clip B/32 Result: {Result3}, \n yolo Result: {Result4}, \n")
    # print(f"\n Insight Face Error: {error1}, \n Media pipe Error: {errormedia}, \n Clip B/32 Error: {errorclip}, \n yolo error: {erroryolo}, \n NSFW error: {errornsfw}.\n")

    # Clean up temporary folders
    for folder in ['TempFaces']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    # Returning Final Result
    return final_result