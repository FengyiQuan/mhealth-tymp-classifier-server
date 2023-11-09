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
        tpp = json_data.get('tpp')
        ecv = json_data.get('ecv')
        sa = json_data.get('sa')
        zeta = json_data.get('zeta')
        slope = json_data.get('slope')
        res = classify(tpp, ecv, sa, zeta, slope)
    return jsonify(res)

def classify(tpp, ecv, sa, zeta, slope):
    # Generate tympanometry data
    pressure, compliance = sim_tracing(tpp, ecv, sa, zeta, slope)
    
    # Perform model inference
    input_ = preprocess(pressure, compliance)
    with torch.no_grad():
        tymptype, attributes = model(input_)
    tymptype = {1:'A', 2:'B', 3:'C'}[tymptype[0].item()]
    ECV, TPP, ECV_std, TPP_std = attributes[0]
    if TPP_std > 100:
        TPP += float('nan')



    res = {"tympType": tymptype,
           "ECV": ECV.numpy().tolist(),
           "TPP": TPP.numpy().tolist(),
           "ECV_std": ECV_std.numpy().tolist(),
           "TPP_std": TPP_std.numpy().tolist(),
           "pressure": pressure.tolist(),
           "compliance": compliance.tolist()}

    return res



def sim_tracing(tpp=0, ecv=1.0, sa=1.0, zeta=2e-3, slope=5e-4):
    """An analytic formula for generating simulated tympanograms"""
    p = np.linspace(-399, 200, 600)
    atm = 1e5 / 10  # 1 atm in decaPascals
    a = 1 / (1 + (tpp - p)**2 / (zeta**2 * (2*atm + tpp + p)**2))
    a200 = a[-1]
    amax = sa / (1 - a200)
    a *= amax
    a += ecv - amax * a200
    a += slope * (p - tpp) * (p < tpp)
    return p, a