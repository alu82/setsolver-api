import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

from setsolverapi import app
from setsolverapi.classifier import classifier as clf

@app.route("/")
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    input_tensor = get_input_tensor(file)
    top_prob, top_class = get_prediction(input_tensor)   
    return jsonify({
        'class' : top_class,
        'probability' : top_prob
        })

def get_input_tensor(image_file):
    mean = [0.6, 0.6, 0.6]
    std = [0.2, 0.2, 0.2]
    input_transforms = transforms.Compose([
        transforms.Resize((250,160)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    image = Image.open(image_file)
    timg = input_transforms(image)
    timg = torch.unsqueeze(timg, 0)
    return timg

def get_prediction(input_tensor):
    log_probabilities = clf.forward(input_tensor)
    probabilities = torch.exp(log_probabilities)
    top_prob, top_class = probabilities.topk(1, dim=1)
    return top_prob.item(), top_class.item()