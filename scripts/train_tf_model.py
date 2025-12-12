import os
import argparse
import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Extract MFCC features
# -----------------------------
def extract_mfcc(path, sr=8000, n_mfcc=20):
    try:
        # Load audio
        y, orig_sr = librosa.load(path, sr=None, mono=True)

        # Resample if needed (NEW librosa API)
        if orig_sr != sr:
            y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)

        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
        return mfcc_mean

    except Exception as e:
        print(f"[ERROR] Failed to load {path} — {e}")
        return None


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(data_dir, metadata, sr, n_mfcc):
    meta_path = os.path.join(data_dir, metadata)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"[ERROR] metadata.csv not found at: {meta_path}")

    df = pd.read_csv(meta_path, header=None, names=["filename", "label"])
    print(f"[INFO] Loading metadata: {len(df)} entries found.")

    wav_folder = os.path.join(data_dir, "wavs")

    X = []
    y = []

    for i, row in df.iterrows():
        wav_path = os.path.join(wav_folder, row["filename"])

        if not os.path.exists(wav_path):
            print(f"[WARN] Missing file: {wav_path}")
            continue

        features = extract_mfcc(wav_path, sr=sr, n_mfcc=n_mfcc)
        if features is None:
            continue

        X.append(features)
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Loaded {len(X)} valid audio samples.")
    return X, y


# -----------------------------
# Build Model
# -----------------------------
def build_model(input_dim, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# -----------------------------
# Main Training Script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    X, y = load_dataset(args.data_dir, args.metadata, args.sr, args.n_mfcc)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    class_map = {i: label for i, label in enumerate(le.classes_)}
    print("\n[INFO] Class mapping:", class_map)

    # Save class map
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(class_map, f)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    # Build model
    model = build_model(input_dim=X.shape[1], n_classes=len(le.classes_))
    model.summary()

    # Train Model
    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # Save SavedModel
    saved_path = os.path.join(args.out_dir, "saved_model")
    model.save(saved_path, include_optimizer=False)
    print(f"[INFO] Saved model at {saved_path}")

    # -------------------------
    # Convert to TFLite (FLOAT)
    # -------------------------
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    tflite_float = converter.convert()
    float_path = os.path.join(args.out_dir, "model_float.tflite")
    open(float_path, "wb").write(tflite_float)
    print(f"[INFO] Saved float TFLite model: {float_path}")

    # -------------------------
    # Convert to TFLite (INT8)
    # -------------------------
    def rep_ds():
        for i in range(min(500, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    int8_path = os.path.join(args.out_dir, "model_int8.tflite")
    open(int8_path, "wb").write(tflite_int8)
    print(f"[INFO] Saved int8 TFLite model: {int8_path}")

    print("\n🎉 TRAINING + TFLITE CONVERSION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
