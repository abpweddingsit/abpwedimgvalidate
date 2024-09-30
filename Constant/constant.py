from transformers import AutoModelForImageClassification, ViTImageProcessor
# from paddleocr import PaddleOCR
import re
from transformers import CLIPProcessor, CLIPModel
import torch
import mediapipe as mp
import clip
from ultralytics import YOLO


# PATHS
YOLO_FOLDER = "/home/abp/Documents/ABPProduction/ABP/ProfileModeration/Version18/API27/API27/best.pt"

# NSFW
NSFWModel = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
NSFWProcessor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')

# PaddleOCR
PATTERN = re.compile(r'(?<!\d)(?:\+91[\-\s]?)?(?:[789](?:[^\s]?\d){3,9})(?!\d)') #r'\d{10}'

# CLIP Anime
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
# CLIP B32
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