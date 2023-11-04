from flask import Flask, request, jsonify
import json
import logging
import torch
import numpy as np
from scipy.signal import butter, sosfiltfilt

app = Flask(__name__)


def build_model():
    """Gets the trained PyTorch TorchScript model"""
    model = torch.jit.load('model.pt')
    return model

model = build_model()

def preprocess(pressure, compliance):
    """Pre-processes tympanogram data"""
    if np.any(np.diff(pressure) <= 0):
        raise ValueError('Pressure sweep is not monotonic')

    default_p = np.arange(-399, 201, dtype=float)
    lpf = butter(4, 0.04, 'lowpass', output='sos')
    
    trace = np.interp(default_p, pressure, compliance)
    trace = np.stack([trace, sosfiltfilt(lpf, trace)])
    trace = torch.from_numpy(trace).float()[None]
    return trace

@app.route("/", methods=['POST'])
def hello_world():
    if request.method == 'POST':
        json_data = request.get_json()
        pressure = json_data.get('pressure')
        compliance = json_data.get('compliance')
        res = classify(pressure,compliance)
    return jsonify(res)

def classify(pressure, compliance):
    
    # Get the length and width parameters from the event object. The 
    # runtime converts the event object to a Python dictionary
    
    # pressure = event['pressure']
    # compliance = event['compliance']
        
     # Perform model inference
    input = preprocess(pressure, compliance)
    with torch.no_grad():
        tymptype, attributes = model(input)
    tymptype = {1: 'A', 2: 'B', 3: 'C'}[tymptype[0].item()]
    ECV, TPP, ECV_std, TPP_std = attributes[0]
    if TPP_std > 100:
        TPP += float('nan')
    res = {"tympType": tymptype,
           "ECV": ECV.numpy().tolist(),
           "TPP": TPP.numpy().tolist(),
           "ECV_std": ECV_std.numpy().tolist(),
           "TPP_std": TPP_std.numpy().tolist()}

    return res



