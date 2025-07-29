"""
Debug script to isolate Sprint 3 training issues
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import Settings
from src.models.regime_model import RegimeAwareModel


async def debug_training():
    """Debug the model training process."""
    print("🔍 Debugging model training...")
    
    try:
        # Initialize settings and model
        settings = Settings()
        model = RegimeAwareModel(settings)
        
        print("✅ Model created successfully")
        
        # Initialize model
        init_result = await model.initialize()
        print(f"Model initialization: {init_result}")
        
        if not init_result['success']:
            print(f"❌ Model initialization failed: {init_result.get('error')}")
            return
        
        print("✅ Model initialized successfully")
        
        # Try training
        print("🎯 Starting model training...")
        training_result = await model.train_model("AAPL", lookback_days=50)
        
        print(f"Training result: {training_result}")
        
        if training_result['success']:
            print("✅ Model training successful!")
            
            # Try making a prediction
            print("🔮 Testing prediction...")
            prediction = await model.predict("AAPL", explain=True)
            print(f"Prediction: {prediction}")
            
        else:
            print(f"❌ Model training failed: {training_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_training())
