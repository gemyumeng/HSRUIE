import torch
import clip
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

image = preprocess(Image.open("/home/Data_yuanbao/ym2/CLIP-main/000136.png")).unsqueeze(0).to(device)
text = clip.tokenize(["The objects in the image are clearly identifiable","The objects in the image are not clearly identifiable"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]