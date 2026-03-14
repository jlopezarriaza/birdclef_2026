import os
import tensorflow as tf
import numpy as np
import kagglehub

def run_test_tflite():
    print("--- VERTEX PERCH V2 TFLITE SMOKE TEST (CPU) ---")
    
    try:
        # 1. Download TFLite version of Perch v2
        # Corrected slug: The TFLite variation often uses version 1 or default
        print("Downloading Perch v2 TFLite...")
        model_path = kagglehub.model_download('google/bird-vocalization-classifier/tfLite/bird-vocalization-classifier/1')
        
        # TFLite models usually have a .tflite file in the root of the download
        tflite_files = [f for f in os.listdir(model_path) if f.endswith('.tflite')]
        if not tflite_files:
            print(f"Contents of {model_path}: {os.listdir(model_path)}")
            raise FileNotFoundError("No .tflite file found in downloaded model.")
            
        tflite_path = os.path.join(model_path, tflite_files[0])
        print(f"TFLite Path: {tflite_path}")
        
        # 2. Initialize Interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input Shape: {input_details[0]['shape']}")
        
        # 3. Dummy Inference (5s @ 32kHz)
        dummy_audio = np.zeros((1, 160_000), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_audio)
        
        print("Running TFLite inference...")
        interpreter.invoke()
        
        # 4. Extract Embedding (Find the 1536-dim output)
        for i, det in enumerate(output_details):
            output_data = interpreter.get_tensor(det['index'])
            print(f"Output {i} shape: {output_data.shape}")
            if output_data.shape[-1] == 1536:
                print(f"✅ SUCCESS! Found 1536-dim embedding at output {i}")
                print(f"Embedding head: {output_data[0][:5]}")
                return
        
        print("❌ FAILED: Could not find 1536-dim output.")
            
    except Exception as e:
        print(f"❌ TFLITE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test_tflite()
