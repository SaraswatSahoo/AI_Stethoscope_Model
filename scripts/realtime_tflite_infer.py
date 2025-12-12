import argparse
import numpy as np
import sounddevice as sd
import librosa
import json
import tensorflow as tf

SR = 8000
DURATION = 2
N_MFCC = 20

def record():
    print("Recording...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1).astype(np.float32)

def main():
    interpreter = tf.lite.Interpreter(model_path="models/model_int8.tflite")
    interpreter.allocate_tensors()

    with open("models/label_map.json") as f:
        labels = json.load(f)

    audio = record()
    feat = extract_mfcc(audio)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = feat.reshape(1, -1).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])

    pred = int(np.argmax(out))
    print("Prediction:", labels[str(pred)])

if __name__ == "__main__":
    main()
