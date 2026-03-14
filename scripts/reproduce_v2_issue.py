
import os
import time
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import kagglehub

def test_perch_v1():
    print("\n--- TESTING PERCH V1 (version 4) ---")
    try:
        model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4')
        print(f"Model path: {model_path}")
        model = tf.saved_model.load(model_path)
        infer = model.signatures['serving_default']
        print(f"Signatures: {list(model.signatures.keys())}")
        print(f"Structured Input: {infer.structured_input_signature[1]}")
        print(f"Structured Output: {infer.structured_outputs}")
        
        # Test Inference
        dummy_input = tf.zeros((1, 160000), dtype=tf.float32)
        input_key = list(infer.structured_input_signature[1].keys())[0]
        outputs = infer(**{input_key: dummy_input})
        print(f"Output Keys: {list(outputs.keys())}")
        if 'embedding' in outputs:
            print(f"Embedding Shape: {outputs['embedding'].shape}")
        elif 'output_1' in outputs:
            print(f"Output 1 Shape: {outputs['output_1'].shape}")
        return True
    except Exception as e:
        print(f"❌ V1 FAILED: {e}")
        return False

def test_perch_v2():
    print("\n--- TESTING PERCH V2 (version 2) ---")
    # Perch v2 is usually accessed via hub
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
    print(f"Loading from: {model_url}")
    try:
        # Try hub.load first
        model = hub.load(model_url)
        infer = model.signatures['serving_default']
        print(f"Signatures: {list(model.signatures.keys())}")
        print(f"Structured Input: {infer.structured_input_signature[1]}")
        print(f"Structured Output: {infer.structured_outputs}")
        
        # Test Inference
        dummy_input = tf.zeros((1, 160000), dtype=tf.float32)
        input_key = list(infer.structured_input_signature[1].keys())[0]
        outputs = infer(**{input_key: dummy_input})
        print(f"Output Keys: {list(outputs.keys())}")
        if 'embedding' in outputs:
            print(f"Embedding Shape: {outputs['embedding'].shape}")
        return True
    except Exception as e:
        print(f"❌ V2 FAILED: {e}")
        # Try kagglehub download as fallback
        try:
            print("Trying kagglehub download as fallback...")
            p = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/perch_v2/2')
            m = tf.saved_model.load(p)
            print("Successfully loaded via kagglehub!")
            return True
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
        return False

if __name__ == "__main__":
    # Test with XLA OFF (default for V1 success)
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
    print("Running with XLA AUTO JIT = -1")
    v1_ok = test_perch_v1()
    v2_ok = test_perch_v2()
    
    if not v2_ok:
        print("\n--- RETRYING V2 WITH XLA ON ---")
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=1"
        test_perch_v2()
