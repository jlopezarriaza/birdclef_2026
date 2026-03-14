import os
import tensorflow as tf
import tensorflow_text as tf_text
import kagglehub
import numpy as np

def run_test():
    print(f"--- VERTEX PERCH V2 (TF 2.20+) TEST ---")
    print(f"TensorFlow: {tf.__version__}")
    print(f"TensorFlow Text: {tf_text.__version__}")
    
    try:
        # 1. Download/Load Model using kagglehub (matches examples)
        print("Downloading/Loading Perch v2 (version 2)...")
        model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/perch_v2/2')
        print(f"Model path: {model_path}")
        
        # Load the SavedModel
        model = tf.saved_model.load(model_path)
        infer = model.signatures['serving_default']
        print("✅ Model loaded successfully!")
        
        # 2. Inspect Input (XLA/StableHLO friendly)
        input_spec = infer.structured_input_signature[1]
        input_key = list(input_spec.keys())[0]
        print(f"Input Key: {input_key}")
        
        # 3. Inference with dummy audio (160,000 samples for 5s @ 32kHz)
        dummy_audio = np.zeros((1, 160_000), dtype=np.float32)
        print("Running inference (Zero input test)...")
        
        # Call the signature
        outputs = infer(**{input_key: tf.constant(dummy_audio)})
        
        # 4. Success verification
        emb = outputs['embedding'].numpy()
        print(f"✅ INFERENCE SUCCESSFUL!")
        print(f"Embedding Shape: {emb.shape}")
        
        if emb.shape[1] == 1536:
            print("🚀 Dimension Match: This is Perch v2.")
            
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
