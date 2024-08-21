import os
import shutil
import insightface
from insightface.app import FaceAnalysis
import cv2
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import face_recognition
import mediapipe as mp
import torch
import clip
from ultralytics import YOLO
import requests
from PIL import Image
import gradio as gr
from io import BytesIO
from transformers import AutoModelForImageClassification, ViTImageProcessor

BASE_FOLDER = "/home/abp/Documents/ABPProduction/ABP/ProfileModeration/Version8/Images"
YOLO_FOLDER = "/home/abp/Documents/ABPProduction/ABP/ProfileModeration/Version8/best.pt"



try:
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
except Exception as e:
    print(f"Error loading model: {e}")
    app = None

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.0,
    min_tracking_confidence=0.90
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
text = ["a cap", "a hat", "a sunglass", "a helmet", "a reading glass", "a mask"]
text_tokens = clip.tokenize(text).to(device)

RNmodel, RNpreprocess = clip.load("RN101", device=device)
rn101textlist = ["a sunglass","a reading glass"]
rn101text = clip.tokenize(rn101textlist).to(device)

yolo_model = YOLO(YOLO_FOLDER)                                                                     # YOLO Path

mapping = {0 : "sunglasses", 1 : "sunglasses", 2 : "eyeglasses", 3 : "headware", 4 : "headware", 5 : "headware"}

def download_and_convert_image(image_url, output_folder=BASE_FOLDER):
    try:
        response = requests.get(image_url)
        image_data = response.content
        if not response.ok:
            return None, "Error downloading image"

        image = Image.open(BytesIO(image_data))
        if image.format == 'WebP':
            image = image.convert('RGB')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        filename = os.path.basename(image_url)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        image.save(output_path, 'PNG')
        return output_path, None
    except Exception as e:
        return None, str(e)

def detect_nsfw(image_path):
    try:
        img = cv2.imread(image_path)
        model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        # print(f"NSFW Label: {label}")

        if label == 'nsfw':
            return 'Image contains NSFW content'
        else:
            return None                                                 # No NSFW content detected
    except Exception as e:
        return str(e)

def crop_faces(image_path, output_dir, expansion_factor=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            return False, 'Multiple faces detected' if len(face_locations) > 1 else 'No Face Detected'

        pil_image = Image.open(image_path)
        base_name = os.path.basename(image_path)
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
        return False, str(e)

def save_face(largestface, image_path, output_dir, expansion_factor=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path)
        base_name = os.path.basename(image_path)
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
        return False, str(e)


def check_image(image_url):
    try:
        image_path, error = download_and_convert_image(image_url)
        if error:
            return 'Rejected', error

        image = cv2.imread(image_path)
        faces = app.get(image)
        
        if len(faces) == 1:
            success, error = crop_faces(image_path, 'TempFaces')
            if success:
                return 'Accepted', None
            else:
                return 'Rejected', error if error else 'Face cropping failed'

        elif len(faces) > 1:
            # Area Checking in Group Photoes
            areas = []
            largestfacearea = 0
            largestfaceindex = None
            for index,face in enumerate(faces):
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
                success,error = save_face(largestface,image_path,"TempFaces")
                if success:
                    return 'Accepted', None
                else:
                    return 'Rejected', error if error else 'Face cropping failed'
                # success, error = crop_faces(image_path, 'TempFaces')
                # return 'Accepted', 'Single Face'
            else:
                return 'Rejected', 'Multiple faces detected'

        else:
            return 'Rejected', 'No Face Detected'

    except Exception as e:
        return 'Rejected', str(e)

def detect_landmarks(image):
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]

def get_image_path_from_url(image_url, base_folder=BASE_FOLDER):
    filename = os.path.basename(image_url)
    image_name = os.path.splitext(filename)[0] + '.png'
    return os.path.join(base_folder, image_name)

def process_single_image(image_path):
    try:
        # image_path = get_image_path_from_url(image_url)
        if not os.path.exists(image_path):
            return 'Rejected', f"Image not found at {image_path}"

        image = cv2.imread(image_path)
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
        return 'Rejected', str(e)

def process_image_clip(image_path):
    try:
        img = Image.open(image_path)
        image = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        predicted_index = probs.argmax()
        confidence = probs[0][predicted_index]
        detected_class = text[predicted_index]
        print(f"CLIP B32 Confidence: {confidence}")
        print(f"Detected Class B32: {detected_class}")

        if confidence > 0.5 and (detected_class == "a sunglass" or detected_class == "a reading glass"):            
            if detected_class in ["a sunglass", "a reading glass"]:
                rn101_image = RNpreprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    rn101_logits_per_image, rn101_logits_per_text = RNmodel(rn101_image, rn101text)
                    rn101_probs = rn101_logits_per_image.softmax(dim=-1).cpu().numpy()
                rn101_predicted_index = rn101_probs.argmax()
                rn101_confidence = rn101_probs[0][rn101_predicted_index]
                print(f"RN101 Confidence: {rn101_confidence}")
                print(f"Predicted Index RN101: {rn101_predicted_index}")
                if rn101_confidence > 0.5 and rn101textlist[rn101_predicted_index] == "a reading glass":
                    # print("Accepted by RN101")
                    return "Accepted", None
                else:
                    return "Rejected", "RN101 model did not confirm 'reading glass' with sufficient confidence"
            else:
                return "Rejected", f"Error: {detected_class}"
        
        elif confidence > 0.8:
            # print(f"CLIP B32 (80%): {detected_class}")
            return "Rejected",f"Error: {detected_class}"
        
        else:
            return "Accepted", None
    except Exception as e:
        return "Error processing image", str(e)

def detect_image_class(image_path):
    try:
        source = Image.open(image_path)
        results = yolo_model(source)
        if len(results[0].boxes) == 0:
            return "Accepted", None

        conf = torch.max(results[0].boxes.conf).item()
        if conf < 0.8:
            return "Accepted", None

        z = torch.argmax(results[0].boxes.conf).item()
        a = int(results[0].boxes.cls[z].item())
        if a == 2: #
            return "Accepted", None
        return "Rejected", mapping[a]
    except Exception as e:
        return "Rejected", f"{e}"

def get_result(image_url):
    final_result = ""
    errstring = ""
    Result1, error1 = check_image(image_url)
    if Result1 == 'Rejected':
        final_result = "Rejected"
        errstring += error1
    else:
        image_path = get_image_path_from_url(image_url)
        Result2, errormedia = process_single_image(image_path)
        Result3, errorclip = process_image_clip(image_path)
        Result4, erroryolo = detect_image_class(image_path)
        errornsfw = detect_nsfw(image_path)  # Directly get NSFW error message

        # Combined Result
        print(f"{image_url} - \n Insight Face Result: {Result1}, \n Media pipe Result: {Result2}, \n Clip B/32 Result: {Result3}, \n yolo Result: {Result4}, \n NSFW Result: {'Rejected' if errornsfw else 'Accepted'}.")
        print(f"{image_url} - \n Insight Face Error: {error1}, \n Media pipe Error: {errormedia}, \n Clip B/32 Error: {errorclip}, \n yolo error: {erroryolo}, \n NSFW error: {errornsfw}.")

        # Append NSFW error message only if it's NSFW
        if errornsfw:
            errstring += f"NSFW content detected: {errornsfw}. "
            final_result = "Rejected"
            return f"Final Result: {final_result} : {errstring}"

        # Final Result
        accepted_count = sum([Result2 == 'Accepted', Result3 == 'Accepted', Result4 == 'Accepted'])
        if accepted_count >= 2:
            final_result = "Accepted"
        elif errorclip == None and erroryolo == "sunglasses":
            final_result = "Accepted"
            print("Final Acceptance by RN101")
        else:
            final_result = "Rejected"
            if error1 is not None:
                errstring += "No Face or multiple face present or Face clearly not visible or The URL is unreachable"
            if errormedia is not None:
                errstring += " - Facial Features clearly not visible"
            if errorclip is not None:
                errstring += " - Eyewear or headwear present"
            if erroryolo is not None:
                errstring += " - Eyewear or headwear present"

    # Delete the folders Images and TempFaces
    for folder in ['Images', 'TempFaces']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    return f"Final Result: {final_result} : {errstring}"

# Function to download and display the image
def display_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

if __name__=="__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                faceimg = gr.Textbox(value="", label="Link")
                facebtn = gr.Button(value="Get Face")
                face_out = gr.Image(type="pil")
            with gr.Column(scale=1):
                aadharbtn = gr.Button(value="Validate Face")
                aadhar_out = gr.Textbox(value="", label="Output")
        facebtn.click(display_image_from_url, inputs=faceimg, outputs=face_out)
        aadharbtn.click(get_result, inputs=faceimg, outputs=aadhar_out)
    demo.launch(share=True, server_port=7861)