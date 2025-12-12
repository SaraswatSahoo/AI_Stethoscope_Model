from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import io, json
import soundfile as sf
import librosa
import tensorflow as tf

app = FastAPI(title="AI Stethoscope TFLite Server")

MODEL_PATH = "models/model_int8.tflite"
LABEL_MAP_PATH = "models/label_map.json"
SR = 8000
N_MFCC = 20

interpreter = None
input_details = None
output_details = None
label_map = None

def load_model():
    global interpreter, input_details, output_details
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def load_labels():
    global label_map
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

def extract_features_from_bytes(wav_bytes):
    data, orig_sr = sf.read(io.BytesIO(wav_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    if orig_sr != SR:
        data = librosa.resample(y=data, orig_sr=orig_sr, target_sr=SR)

    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1).astype(np.float32)

def predict_with_interpreter(sample):
    inp = sample.reshape(1, -1).astype(input_details[0]['dtype'])

    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        inp = (inp / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        out = scale * (out.astype(np.float32) - zero_point)

    return out

@app.on_event("startup")
def startup_event():
    load_model()
    load_labels()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()

    try:
        feat = extract_features_from_bytes(content)
        out = predict_with_interpreter(feat)
        pred_idx = int(np.argmax(out, axis=1)[0])

        return {
            "label": label_map.get(pred_idx, "unknown"),
            "scores": out.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
