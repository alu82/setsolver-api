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
import itertools as it

from setsolverapi import app
from setsolverapi.classifier import classifier as clf

cards = dict()
card_id = 0
for number in range(3):
    for color in range(3):
        for form in range(3):
            for filling in range(3):
                cards[card_id] = (number, color, form, filling)
                card_id += 1

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

    found_sets = find_valid_set([p["card"] for p in predictions])

    response = {}
    response["extractedCards"] = predictions
    response["foundSets"] = found_sets
    return jsonify(response)

def find_valid_set(cards):
    valid_sets = list()
    for triple in it.combinations(cards, 3):
        if is_valid_set(triple):
            valid_sets.append(triple)
    return valid_sets

def is_valid_set(triple):
    card_0 = cards[triple[0]]
    card_1 = cards[triple[1]]
    card_2 = cards[triple[2]]

    for idx in range(len(card_0)):
        diff_values = set([card_0[idx], card_1[idx], card_2[idx]])
        if len(diff_values) == 2:
            return False
    return True

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