# Example usage of the enhanced plot_error_analysis function
import sys
import os

# Add the src/utils directory to the path
utils_path = os.path.join(os.path.dirname(__file__), 'src', 'utils')
sys.path.insert(0, utils_path)

from src.utils.error import MEASUREMENT, ERROR
import numpy as np

def create_sample_data():
    """Create sample data for demonstration"""
    # Create sample measurements
    model_measurements = []
    cast_measurements = []
    actual_errors = []
    predicted_errors = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(20):
        # Model measurements (baseline)
        model_wind = 10 + np.sin(i * 0.3) * 2
        model_temp = 20 + np.cos(i * 0.2) * 5
        model_pressure = 1013 + np.sin(i * 0.1) * 10
        
        model_meas = MEASUREMENT(model_wind, model_temp, model_pressure, i)
        
        # Cast measurements (with some error)
        cast_wind = model_wind + np.random.normal(0, 1.5)
        cast_temp = model_temp + np.random.normal(0, 2.0)
        cast_pressure = model_pressure + np.random.normal(0, 3.0)
        
        cast_meas = MEASUREMENT(cast_wind, cast_temp, cast_pressure, i)
        
        # Actual errors
        actual_error = ERROR(model_meas, cast_meas)
        
        # Predicted errors (simulate prediction algorithm)
        pred_wind_error = actual_error.error[0] + np.random.normal(0, 0.5)
        pred_temp_error = actual_error.error[1] + np.random.normal(0, 0.8)
        pred_pressure_error = actual_error.error[2] + np.random.normal(0, 1.0)
        
        # Create predicted error object
        pred_cast_wind = model_wind + pred_wind_error
        pred_cast_temp = model_temp + pred_temp_error
        pred_cast_pressure = model_pressure + pred_pressure_error
        
        pred_cast_meas = MEASUREMENT(pred_cast_wind, pred_cast_temp, pred_cast_pressure, i)
        predicted_error = ERROR(model_meas, pred_cast_meas)
        
        model_measurements.append(model_meas)
        cast_measurements.append(cast_meas)
        actual_errors.append(actual_error)
        predicted_errors.append(predicted_error)
    
    return model_measurements, cast_measurements, actual_errors, predicted_errors

def main():
    """Demonstrate the enhanced plotting functionality"""
    print("🔬 Creating sample weather data...")
    model_measurements, cast_measurements, actual_errors, predicted_errors = create_sample_data()
    
    print("📊 Generating enhanced error analysis plot...")
    
    # Generate plot with both actual and predicted errors using ERROR class static method
    fig = ERROR.plot_error_analysis(
        model_measurements=model_measurements,
        cast_measurements=cast_measurements,
        errors=actual_errors,
        predicted_errors=predicted_errors,  # This is the new parameter!
        title="Weather Prediction Error Analysis - Actual vs Predicted",
        save_path="weather_error_comparison.png"
    )
    
    print("✅ Plot generated successfully!")
    print("\n🎯 Key Features:")
    print("• Wind, Temperature, and Pressure analysis with dual error visualization")
    print("• Linear trend comparison between actual and predicted errors")
    print("• Error magnitude comparison with grouped bar charts")
    print("• Comprehensive statistics including prediction accuracy metrics (MAE, RMSE)")
    print("• Professional styling with color-coded actual vs predicted data")

if __name__ == "__main__":
    main()
