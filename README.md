AI Stethoscope — Full Code Package
=================================

This package contains a full, ready-to-run set of scripts to train a small Keras model on MFCC features,
convert it to TFLite (float + int8 quantized), run realtime inference from microphone, and serve predictions via FastAPI.

Directory layout (after unzip):
  ai_stethoscope_full_code/
    scripts/
      train_tf_model.py            # training + conversion (requires TensorFlow)
      realtime_tflite_infer.py     # record from mic -> MFCC -> TFLite predict
      app_tflite.py                # FastAPI server (POST wav file to /predict)
    ai_stethoscope/utils/
      features.py                  # helper MFCC functions
    data/                          # place your data here (wavs/ + metadata.csv)
    models/                        # models will be written here by train script
    requirements.txt
    README.md

Quickstart (Linux / macOS):
1. create venv & install deps
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Prepare data (example):
   data/
     wavs/
       sample1.wav
       sample2.wav
     metadata.csv   # columns: filename,label
   Labels should be: tb, pneumonia, normal (or any mapping)

3. Train & convert:
   python scripts/train_tf_model.py --data_dir data --out_dir models --epochs 40

4. Test realtime inference (mic):
   python scripts/realtime_tflite_infer.py --model models/model_int8.tflite --label_map models/label_map.json

5. Run server:
   uvicorn scripts.app_tflite:app --host 0.0.0.0 --port 8000
   curl -F "file=@data/wavs/sample1.wav" http://localhost:8000/predict

Notes & tips:
- Use tflite-runtime on embedded devices (smaller footprint).
- For production, validate model performance rigorously and consult medical device regulations.
- If you want, I can optionally add a small web UI that records in-browser and sends audio to the FastAPI server.