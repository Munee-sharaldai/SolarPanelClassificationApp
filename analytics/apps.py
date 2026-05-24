from django.apps import AppConfig
import os
from pathlib import Path

class AnalyticsConfig(AppConfig):
    name = 'analytics'
    mlp_model = None
    scaler = None

    def ready(self):
        print("SESS-AI: AnalyticsConfig.ready() is being called...")
        # Import here to avoid potential issues during Django startup
        try:
            import joblib
            import tensorflow as tf
            
            # Path to the models folder relative to this file (analytics/apps.py)
            base_path = Path(__file__).resolve().parent / 'models_data'
            print(f"SESS-AI: Searching for models in {base_path}")
            
            scaler_path = base_path / 'scaler.joblib'
            model_path = base_path / 'mlp_model.keras'
            
            # Load models into class attributes for global access
            AnalyticsConfig.scaler = joblib.load(scaler_path)
            AnalyticsConfig.mlp_model = tf.keras.models.load_model(model_path)
            
            print("SESS-AI: Models (Scaler and MLP) successfully loaded into RAM.")
        except Exception as e:
            print(f"SESS-AI: Error loading models into RAM: {e}")
