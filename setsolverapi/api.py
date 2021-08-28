from flask import jsonify, request
from PIL import Image
from setsolverapi import app
import setsolverapi.service.classifier as clf
import setsolverapi.service.extractor as ex
import setsolverapi.service.setfinder as sf

@app.route("/")
def root():
    return jsonify({'msg' : 'Try POSTing to the /solve endpoint with an RGB image attachment.'})

@app.route('/solve', methods=['POST'])
def solve():
    image = Image.open(request.files['file'])
    cards = ex.extract_cards(image)
    classifications = [clf.classify_card(card) for card in cards]
    found_sets = sf.find_sets([p[1] for p in classifications])
    return to_json_response(classifications, found_sets)

def to_json_response(classifications, sets):
    clf_struct = [dict(probability=pred[0], card=pred[1]) for pred in classifications]
    response = {
        "extractedCards": clf_struct,
        "foundSets": sets
    }
    return jsonify(response)