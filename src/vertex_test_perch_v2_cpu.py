import os
import tensorflow as tf
import tensorflow_text as tf_text
import kagglehub
import numpy as np

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"

def run_test():
    print(f"--- VERTEX PERCH V2 CPU SMOKE TEST (FIXED SLUG) ---")
    print(f"TensorFlow: {tf.__version__}")
    
    try:
        # 1. Download/Load the CPU-SPECIFIC variant discovered in the notebooks
        print("Downloading Perch v2 CPU variant...")
        model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
        print(f"Model path: {model_path}")
        
        # Load the SavedModel
        model = tf.saved_model.load(model_path)
        infer = model.signatures['serving_default']
        print("✅ CPU Model loaded successfully!")
        
        # 2. Dummy Inference (5s @ 32kHz)
        dummy_audio = np.zeros((1, 160_000), dtype=np.float32)
        print("Running CPU inference...")
        
        # The notebook uses 'inputs' as the key
        outputs = infer(inputs=tf.constant(dummy_audio))
        
        # 3. Success verification
        # Let's check both 'embedding' and 'label'
        if 'embedding' in outputs:
            emb = outputs['embedding'].numpy()
            print(f"✅ EMBEDDING SUCCESS! Shape: {emb.shape}")
        
        if 'label' in outputs:
            logits = outputs['label'].numpy()
            print(f"✅ LOGITS SUCCESS! Shape: {logits.shape}")
            
    except Exception as e:
        print(f"❌ CPU TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
