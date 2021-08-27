import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import jsonify, request, make_response
import cv2
from PIL import Image
import numpy as np
import time
import os

from setsolverapi import app
from setsolverapi.classifier import classifier as clf

@app.route("/")
def root():
    return jsonify({'msg' : 'Try POSTing to the /solve endpoint with an RGB image attachment.'})

@app.route('/solve', methods=['POST'])
def solve():
    image_file = request.files['file']
    image = Image.open(image_file)
    cards = get_cards(image)
    
    predictions = list()
    for card in cards:
        probability, card_id = get_card_prediction(card)
        prediction = {}
        prediction["card"] = card_id
        prediction["probability"] = probability
        predictions.append(prediction)

    return jsonify(predictions)

def get_cards(image):
    im = transform_pil_to_opencv(image)
    cards = list()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)   
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    size_threshold = 0.5 * max(cv2.contourArea(c) for c in contours) 
    contours = [c for c in contours if cv2.contourArea(c) > size_threshold]
     
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        card = im[y:y + h, x:x + w]
        if w > h:
            card = cv2.rotate(card, cv2.cv2.ROTATE_90_CLOCKWISE)
        pil_card = transform_opencv_to_pil(card)
        cards.append(pil_card)
    log_extraction(im, contours, cards)
    return cards

def transform_pil_to_opencv(pil_image):
    converted_image = pil_image.convert('RGB')
    opencv_image = np.array(converted_image)
    opencv_image = opencv_image[:, :, ::-1].copy() 
    return opencv_image

def transform_opencv_to_pil(opencv_image):
    converted_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(converted_image)
    return pil_image

def get_card_prediction(image):
    mean = [0.6, 0.6, 0.6]
    std = [0.2, 0.2, 0.2]
    input_transforms = transforms.Compose([
        transforms.Resize((250,160)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    timg = input_transforms(image)
    timg = torch.unsqueeze(timg, 0)
    return get_prediction(timg)


def get_prediction(input_tensor):
    log_probabilities = clf.forward(input_tensor)
    probabilities = torch.exp(log_probabilities)
    top_prob, top_class = probabilities.topk(1, dim=1)
    return top_prob.item(), top_class.item()

def log_extraction(orig_image, contours, extracted_cards):
    folder = f"./setsolverapi/log/{int(time.time())}"
    os.makedirs(folder)
    # cv2.imwrite(f"{folder}/_original.png",orig_image)
    cv2.drawContours(orig_image, contours, -1, (0,255,0), 15)
    cv2.imwrite(f"{folder}/_original_contours.png",orig_image)
    idx = 0
    for card in extracted_cards:
        idx += 1
        card.save(f"{folder}/card_{str(idx)}.png")