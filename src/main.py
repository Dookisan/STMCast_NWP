from __future__ import annotations
import os
from utils.error import ERROR, MEASUREMENT
from utils.data_improved import DatasetFactory, ModelDataset, CastDataset
import sys


def _lookup_models(): 
    print(sys.prefix)
    print("python looks for models in this locations:")

    for path in sys.path:
        print(f"-{path}")


def main():
    """
    Streamlined main function using the new Dataset classes with file support.
    """
    print("🌤️ STMCast NWP - Weather Data Processing")
    print("=" * 60)
    
    # Define path to weather data file
    data_file_path = os.path.join("src", "data", "api.json")
    
    if not os.path.exists(data_file_path):
        print(f"❌ Error: Weather data file not found at {data_file_path}")
        return None, None
    
    print(f"\n📥 Loading data from: {data_file_path}")
    
    # Create Model Dataset (automatically loads data from file)
    model_dataset = ModelDataset.from_json_file(data_file_path)
    print(f"✅ {model_dataset}")
    
    # Create Cast Dataset with synthetic noise from model file
    cast_dataset = DatasetFactory.create_cast_from_model_file(
        file_path=data_file_path,
        station_id="VILLACH_SYNTHETIC", 
        noise_level=0.08  # 8% noise
    )
    print(f"✅ {cast_dataset}")
    
    # Show detailed summaries
    print(model_dataset.summary())
    print(cast_dataset.summary())
    
    # Calculate errors for first few measurements
    print("\n🔍 Error Analysis (first 5 measurements):")
    print("=" * 50)
    
    for i in range(min(5, len(model_dataset))):
        error_calc = ERROR(model_dataset[i], cast_dataset[i])
        error_vector = error_calc.error
        print(f"Measurement {i+1}: Wind={error_vector[0]:+6.2f}, Temp={error_vector[1]:+6.2f}, Pressure={error_vector[2]:+6.1f}")
    
    print(f"\n🎯 Analysis complete!")
    print(f"   Model validation: {'✅ PASSED' if model_dataset.validate_data() else '❌ FAILED'}")
    print(f"   Cast validation:  {'✅ PASSED' if cast_dataset.validate_data() else '❌ FAILED'}")
    
    return model_dataset, cast_dataset




if __name__ == "__main__":
    main()
