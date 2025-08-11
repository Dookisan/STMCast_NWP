from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import json 
import os

# Import your MEASUREMENT class
from .error import MEASUREMENT

# Utility class for common JSON parsing functionality
class WeatherDataParser:
    """
    Utility class for parsing weather JSON data from strings or files.
    """
    
    @staticmethod
    def parse_weather_json_file(file_path: str) -> List[MEASUREMENT]:
        """
        Parse JSON weather data from a file and convert to MEASUREMENT objects.
        
        Args:
            file_path: Path to JSON file containing weather data
            
        Returns:
            List of MEASUREMENT objects created from the weather data
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                weather_data = json.load(f)
            
            measurements = WeatherDataParser._parse_weather_data(weather_data, f"file: {file_path}")
            return measurements
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {e}")
            return []
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
    
    @staticmethod
    def parse_weather_json(json_source: str) -> List[MEASUREMENT]:
        """
        Parse JSON weather data from a string and convert to MEASUREMENT objects.
        
        Args:
            json_source: JSON string containing weather data
            
        Returns:
            List of MEASUREMENT objects created from the weather data
        """
        if not json_source or json_source == "unknown":
            print("No valid JSON source provided for loading data")
            return []
            
        try:
            weather_data = json.loads(json_source)
            measurements = WeatherDataParser._parse_weather_data(weather_data, "JSON string")
            return measurements
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []
        except Exception as e:
            print(f"Error processing weather data: {e}")
            return []
    
    @staticmethod
    def _parse_weather_data(weather_data: dict, source_info: str) -> List[MEASUREMENT]:
        """
        Internal method to parse weather data dictionary.
        
        Args:
            weather_data: Parsed JSON data dictionary
            source_info: Information about the data source (for logging)
            
        Returns:
            List of MEASUREMENT objects
        """
        measurements = []
        
        # Extract data array from JSON
        data_points = weather_data.get("data", [])
        
        for data_point in data_points:
            # Extract required values with fallbacks
            wind_speed = data_point.get("wind_spd", 0)  # wind speed in m/s
            temperature = data_point.get("temp", 0)     # temperature in Celsius
            pressure = data_point.get("pres", 0)        # pressure in hPa
            timestamp = data_point.get("datetime", "")  # timestamp
            
            # Create MEASUREMENT object
            measurement = MEASUREMENT(
                wind=wind_speed,
                temp=temperature, 
                pressure=pressure,
                time=timestamp
            )
            
            measurements.append(measurement)
        
        print(f"Successfully loaded {len(measurements)} measurements from {source_info}")
        return measurements

# Abstract base class for datasets
class DATASET(ABC):
    """
    Abstract base class for handling weather measurement datasets.
    """
    def __init__(self, measurements: List[MEASUREMENT] = None, auto_load: bool = True):
        self.measurements = measurements or []
        
        # Automatisches Laden wenn keine measurements vorhanden
        if auto_load and not self.measurements:
            try:
                loaded_data = self.load_data()
                if loaded_data:
                    self.measurements = loaded_data
            except Exception as e:
                print(f"Warning: Auto-load failed: {e}")
    
    @abstractmethod
    def load_data(self) -> List[MEASUREMENT]:
        """Load data specific to the dataset type."""
        pass
       
    @abstractmethod
    def validate_data(self) -> bool:
        """Validate the loaded data."""
        pass
    
    def add_measurement(self, measurement: MEASUREMENT):
        """Add a single measurement to the dataset."""
        self.measurements.append(measurement)
    
    def get_measurements(self) -> List[MEASUREMENT]:
        """Get all measurements in the dataset."""
        return self.measurements
    
    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, index):
        return self.measurements[index]
    
    def __str__(self):
        """String representation of the dataset."""
        return f"{self.__class__.__name__}(measurements={len(self.measurements)}, type='{self.dataset_type}')"
    
    def __repr__(self):
        """Detailed representation of the dataset."""
        return self.__str__()
    
    def summary(self) -> str:
        """Get detailed summary of the dataset."""
        if not self.measurements:
            return f"{self.__class__.__name__}: No measurements loaded"
        
        winds = [m.wind for m in self.measurements]
        temps = [m.temp for m in self.measurements]
        pressures = [m.pressure for m in self.measurements]
        
        summary_text = f"""
{self.__class__.__name__} Summary:
{'='*50}
Total measurements: {len(self.measurements)}
Dataset type: {self.dataset_type}

Weather Statistics:
- Wind: {min(winds):.1f} - {max(winds):.1f} m/s (avg: {np.mean(winds):.1f})
- Temperature: {min(temps):.1f} - {max(temps):.1f}°C (avg: {np.mean(temps):.1f})
- Pressure: {min(pressures):.1f} - {max(pressures):.1f} hPa (avg: {np.mean(pressures):.1f})

Time range: {self.measurements[0].time} → {self.measurements[-1].time}
Data validation: {'✓ PASSED' if self.validate_data() else '✗ FAILED'}
{'='*50}"""
        return summary_text

# Concrete implementation for MODEL data
class ModelDataset(DATASET):
    """
    Dataset for handling weather model predictions.
    """
    def __init__(self, measurements: List[MEASUREMENT] = None, model_source: str = "", 
                 file_path: str = "", auto_load: bool = True):
        self.model_source = model_source
        self.file_path = file_path
        self.dataset_type = "MODEL"
        
        # Rufe die Basis-Klasse auf - sie übernimmt das Auto-Loading!
        super().__init__(measurements, auto_load)
    
    def load_data(self) -> List[MEASUREMENT]:
        """
        Load weather data from file or JSON string.
        Returns:
            List of MEASUREMENT objects created from the weather data
        """
        if self.file_path:
            # Load from file
            return WeatherDataParser.parse_weather_json_file(self.file_path)
        else:
            # Load from JSON string (legacy support)
            return WeatherDataParser.parse_weather_json(self.model_source)

    def validate_data(self) -> bool:
        """Validate model data integrity."""
        if not self.measurements:
            return False
        
        # Check if all measurements have required fields
        for measurement in self.measurements:
            if not all([
                hasattr(measurement, 'wind'),
                hasattr(measurement, 'temp'), 
                hasattr(measurement, 'pressure'),
                hasattr(measurement, 'time')
            ]):
                return False
        return True
    
    @classmethod
    def from_json_string(cls, json_string: str) -> 'ModelDataset':
        """
        Convenience method to create ModelDataset directly from JSON string.
        """
        return cls(model_source=json_string)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'ModelDataset':
        """
        Convenience method to create ModelDataset directly from JSON file.
        """
        return cls(file_path=file_path)
    
    @classmethod
    def from_weather_api_data(cls, weather_json: str, source_name: str = "API") -> 'ModelDataset':
        """
        Create ModelDataset from weather API JSON data.
        """
        dataset = cls(model_source=weather_json)
        # Override the default source name
        dataset.model_source = source_name
        return dataset

    def __str__(self):
        """String representation of ModelDataset."""
        if self.file_path:
            source_info = f"file='{os.path.basename(self.file_path)}'"
        else:
            source_info = f"source='{self.model_source[:50]}...'" if len(self.model_source) > 50 else f"source='{self.model_source}'"
        return f"ModelDataset({len(self.measurements)} measurements, {source_info})"
    
    def __repr__(self):
        """Detailed representation of ModelDataset."""
        return f"ModelDataset(measurements={len(self.measurements)}, model_source='{self.model_source}', dataset_type='{self.dataset_type}')"

# Concrete implementation for CAST data  
class CastDataset(DATASET):
    """
    Dataset for handling actual weather measurements (observations).
    """
    def __init__(self, measurements: List[MEASUREMENT] = None, station_id: str = "", 
                 json_source: str = "", file_path: str = "", noise_level: float = 0.1, auto_load: bool = False):
        self.station_id = station_id
        self.json_source = json_source  # JSON source für Cast-Daten
        self.file_path = file_path      # File path für Cast-Daten
        self.noise_level = noise_level  # Noise level für realistische Cast-Daten
        self.dataset_type = "CAST"
        
        # CAST data normalerweise nicht auto-laden (meist aus externen Quellen)
        super().__init__(measurements, auto_load)
    
    def load_data(self) -> List[MEASUREMENT]:
        """
        Load actual measurement data from weather stations.
        If file_path or json_source is provided, parse it and add realistic noise.
        """
        base_measurements = []
        
        if self.file_path:
            # Lade Basis-Daten aus Datei mit dem gemeinsamen Parser
            base_measurements = WeatherDataParser.parse_weather_json_file(self.file_path)
        elif self.json_source:
            # Lade Basis-Daten aus JSON String mit dem gemeinsamen Parser
            base_measurements = WeatherDataParser.parse_weather_json(self.json_source)
        else:
            # Fallback für andere Cast-Datenquellen
            print(f"Loading CAST data from station {self.station_id}")
            return self.measurements or []
        
        if not base_measurements:
            return []
            
        # Füge realistisches Noise hinzu um Cast-Daten zu simulieren
        cast_measurements = []
        for measurement in base_measurements:
            # Noise hinzufügen (normalverteilt)
            noisy_wind = measurement.wind + np.random.normal(0, self.noise_level * measurement.wind)
            noisy_temp = measurement.temp + np.random.normal(0, self.noise_level * 2.0)  # Temperatur-Noise
            noisy_pressure = measurement.pressure + np.random.normal(0, self.noise_level * 5.0)  # Druck-Noise
            
            cast_measurement = MEASUREMENT(
                wind=max(0, noisy_wind),  # Wind kann nicht negativ sein
                temp=noisy_temp,
                pressure=max(900, noisy_pressure),  # Minimaler atmosphärischer Druck
                time=measurement.time
            )
            cast_measurements.append(cast_measurement)
        
        source_info = f"file '{os.path.basename(self.file_path)}'" if self.file_path else "JSON string"
        print(f"Generated {len(cast_measurements)} cast measurements from {source_info} with noise level {self.noise_level}")
        return cast_measurements
    
    def validate_data(self) -> bool:
        """Validate cast data integrity."""
        if not self.measurements:
            return False
            
        # Check data quality specific to observations
        for measurement in self.measurements:
            # Add specific validation for observation data
            if measurement.pressure <= 0 or measurement.temp < -100:
                return False
        return True
    
    def __str__(self):
        """String representation of CastDataset."""
        noise_info = f"noise={self.noise_level}" if self.json_source else "no_noise"
        return f"CastDataset({len(self.measurements)} measurements, station='{self.station_id}', {noise_info})"
    
    def __repr__(self):
        """Detailed representation of CastDataset."""
        return f"CastDataset(measurements={len(self.measurements)}, station_id='{self.station_id}', json_source={'Yes' if self.json_source else 'No'}, noise_level={self.noise_level})"

# Factory class for creating datasets
class DatasetFactory:
    """
    Factory for creating different types of datasets.
    """
    
    @staticmethod
    def create_dataset(dataset_type: str, **kwargs) -> DATASET:
        """
        Create a dataset based on the specified type.
        
        Args:
            dataset_type: Type of dataset ('MODEL' or 'CAST')
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            DATASET: Instance of the appropriate dataset type
            
        Raises:
            ValueError: If dataset_type is not supported
        """
        dataset_type = dataset_type.upper()
        
        if dataset_type == "MODEL":
            return ModelDataset(
                measurements=kwargs.get('measurements', []),
                model_source=kwargs.get('model_source', 'unknown'),
                file_path=kwargs.get('file_path', '')
            )
        elif dataset_type == "CAST":
            return CastDataset(
                measurements=kwargs.get('measurements', []),
                station_id=kwargs.get('station_id', 'unknown'),
                json_source=kwargs.get('json_source', ''),
                noise_level=kwargs.get('noise_level', 0.1),
                auto_load=kwargs.get('auto_load', False)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'MODEL' or 'CAST'.")
    
    @staticmethod
    def create_model_dataset(measurements: List[MEASUREMENT] = None, 
                           model_source: str = "NWP") -> ModelDataset:
        """Convenient method to create a model dataset."""
        return ModelDataset(measurements, model_source)
    
    @staticmethod
    def create_model_dataset_from_file(file_path: str) -> ModelDataset:
        """Convenient method to create a model dataset from JSON file."""
        return ModelDataset(file_path=file_path)
    
    @staticmethod  
    def create_cast_dataset(measurements: List[MEASUREMENT] = None,
                          station_id: str = "default", 
                          json_source: str = "",
                          noise_level: float = 0.1) -> CastDataset:
        """Convenient method to create a cast dataset."""
        return CastDataset(measurements, station_id, json_source, noise_level)
    
    @staticmethod
    def create_cast_from_model_data(model_json: str, station_id: str = "synthetic", 
                                   noise_level: float = 0.1) -> CastDataset:
        """Create realistic cast data from model data by adding noise."""
        return CastDataset(json_source=model_json, station_id=station_id, 
                         noise_level=noise_level, auto_load=True)
    
    @staticmethod
    def create_cast_from_model_file(file_path: str, station_id: str = "synthetic", 
                                   noise_level: float = 0.1) -> CastDataset:
        """Create realistic cast data from model JSON file by adding noise."""
        return CastDataset(file_path=file_path, station_id=station_id, 
                         noise_level=noise_level, auto_load=True)


