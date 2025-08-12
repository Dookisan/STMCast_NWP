from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

class MEASUREMENT:
    def __init__(self, wind, temp, pressure, time): 
        self.wind = wind
        self.temp = temp
        self.pressure = pressure
        self.time = time

    def __str__(self):
        return f"Wind: {self.wind}, Temp: {self.temp}, Pressure: {self.pressure}, Time: {self.time}"

    def __repr__(self):
        return f"ERROR(wind={self.wind}, temp={self.temp}, pressure={self.pressure}, time={self.time})"
    

class ERROR (): 
    def __init__(self, MODEL: MEASUREMENT, CAST:MEASUREMENT): 
        self.MODEL = MODEL
        self.CAST = CAST

    
    @property
    def _estimation_error(self)->np.array:
        """
        Calculate the estimation error of the current attributes mk. 
        Returns: 
        --------------------------
        The vector of estimation errors ek
        """

        dwind = self.MODEL.wind - self.CAST.wind
        dtemp = self.MODEL.temp - self.CAST.temp
        dpress = self.MODEL.pressure - self.CAST.pressure  

        return np.array([dwind, dtemp, dpress])

    @property
    def error(self):
        return self._estimation_error   

    def predict(self, input: MEASUREMENT) -> MEASUREMENT:
        """
        Predict the output based on the input measurement and the current error.
        This implementation follows an linear model consisting of guessed 
        Lagrange polynoms. They will be calculated by using linear regression. 

        Args:
            input (MEASUREMENT): The input measurement to base the prediction on.

        Returns:
            MEASUREMENT: The predicted measurement.
        """

        coefficients = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Error_matrix = np.array([self.MODEL.wind, self.MODEL.temp, self.MODEL.pressure])

        # New matrix multiplication
        estimated = coefficients @ Error_matrix

        prod1 =  Error_matrix.T @ Error_matrix
        prod2 = Error_matrix.T @ estimated

        coefficients = np.linalg.inv(prod1) @ prod2

    def plot_error_analysis(model_measurements, cast_measurements, errors, save_path=None):
        """
        Beautiful visualization of error estimators with linear functions and real values.
        
        Args:
            model_measurements: List of MEASUREMENT objects (model data)
            cast_measurements: List of MEASUREMENT objects (cast/observed data) 
            errors: List of ERROR objects
            save_path: Optional path to save the plot
        """
        
        # Set style for beautiful plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🌤️ Weather Prediction Error Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # Extract data for plotting
        time_indices = np.arange(len(model_measurements))
        
        # Wind data
        model_wind = [m.wind for m in model_measurements]
        cast_wind = [c.wind for c in cast_measurements]
        wind_errors = [e.error[0] for e in errors]
        
        # Temperature data
        model_temp = [m.temp for m in model_measurements]
        cast_temp = [c.temp for c in cast_measurements]
        temp_errors = [e.error[1] for e in errors]
        
        # Pressure data
        model_pressure = [m.pressure for m in model_measurements]
        cast_pressure = [c.pressure for c in cast_measurements]
        pressure_errors = [e.error[2] for e in errors]
        
        # --- Wind Speed Plot ---
        ax1.plot(time_indices, model_wind, 'o-', linewidth=3, markersize=8, 
                label='Model Prediction', color='#2E86C1', alpha=0.8)
        ax1.plot(time_indices, cast_wind, 's-', linewidth=3, markersize=8,
                label='Observed Data', color='#E74C3C', alpha=0.8)
        
        # Linear trend line for wind
        z_wind = np.polyfit(time_indices, wind_errors, 1)
        p_wind = np.poly1d(z_wind)
        ax1.plot(time_indices, model_wind + p_wind(time_indices), '--', 
                linewidth=2, color='#F39C12', label=f'Linear Error Trend (slope: {z_wind[0]:.3f})')
        
        ax1.fill_between(time_indices, model_wind, cast_wind, alpha=0.2, color='gray', label='Error Region')
        ax1.set_title('💨 Wind Speed Analysis', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Time Index', fontsize=12)
        ax1.set_ylabel('Wind Speed (m/s)', fontsize=12)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # --- Temperature Plot ---
        ax2.plot(time_indices, model_temp, 'o-', linewidth=3, markersize=8,
                label='Model Prediction', color='#2E86C1', alpha=0.8)
        ax2.plot(time_indices, cast_temp, 's-', linewidth=3, markersize=8,
                label='Observed Data', color='#E74C3C', alpha=0.8)
        
        # Linear trend line for temperature
        z_temp = np.polyfit(time_indices, temp_errors, 1)
        p_temp = np.poly1d(z_temp)
        ax2.plot(time_indices, model_temp + p_temp(time_indices), '--',
                linewidth=2, color='#F39C12', label=f'Linear Error Trend (slope: {z_temp[0]:.3f})')
        
        ax2.fill_between(time_indices, model_temp, cast_temp, alpha=0.2, color='orange', label='Error Region')
        ax2.set_title('🌡️ Temperature Analysis', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # --- Pressure Plot ---
        ax3.plot(time_indices, model_pressure, 'o-', linewidth=3, markersize=8,
                label='Model Prediction', color='#2E86C1', alpha=0.8)
        ax3.plot(time_indices, cast_pressure, 's-', linewidth=3, markersize=8,
                label='Observed Data', color='#E74C3C', alpha=0.8)
        
        # Linear trend line for pressure
        z_pressure = np.polyfit(time_indices, pressure_errors, 1)
        p_pressure = np.poly1d(z_pressure)
        ax3.plot(time_indices, model_pressure + p_pressure(time_indices), '--',
                linewidth=2, color='#F39C12', label=f'Linear Error Trend (slope: {z_pressure[0]:.3f})')
        
        ax3.fill_between(time_indices, model_pressure, cast_pressure, alpha=0.2, color='purple', label='Error Region')
        ax3.set_title('🌀 Pressure Analysis', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Time Index', fontsize=12)
        ax3.set_ylabel('Pressure (hPa)', fontsize=12)
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        # --- Error Magnitude Plot ---
        error_magnitudes = [np.linalg.norm(e.error) for e in errors]
        colors = plt.cm.viridis(np.linspace(0, 1, len(error_magnitudes)))
        
        bars = ax4.bar(time_indices, error_magnitudes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Linear trend for error magnitude
        z_error = np.polyfit(time_indices, error_magnitudes, 1)
        p_error = np.poly1d(z_error)
        ax4.plot(time_indices, p_error(time_indices), 'r--', linewidth=3,
                label=f'Linear Trend (slope: {z_error[0]:.3f})')
        
        ax4.set_title('📊 Error Magnitude Distribution', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Time Index', fontsize=12)
        ax4.set_ylabel('Error Magnitude', fontsize=12)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for error magnitude
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(error_magnitudes), vmax=max(error_magnitudes)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax4, shrink=0.8)
        cbar.set_label('Error Intensity', fontsize=10)
        
        # Adjust layout and styling
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add subtle background gradient
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#f8f9fa')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
        
        # Statistics box
        mean_error = np.mean(error_magnitudes)
        std_error = np.std(error_magnitudes)
        max_error = np.max(error_magnitudes)
        
        stats_text = f"""📈 Statistics:
        Mean Error: {mean_error:.3f}
        Std Error: {std_error:.3f}
        Max Error: {max_error:.3f}
        Total Points: {len(errors)}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Plot saved to: {save_path}")
        
        plt.show()
        
        return fig

