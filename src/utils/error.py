from __future__ import annotations
import numpy as np

class MEASUREMENT:
    def __init__(self, wind, temp, pressure, time): 
        self.wind = wind
        self.temp = temp
        self.pressure = pressure
        self.time = time
        self.error = self._estimation_error
        
    def __str__(self):
        return f"Wind: {self.wind}, Temp: {self.temp}, Pressure: {self.pressure}, Time: {self.time}"

    def __repr__(self):
        return f"ERROR(wind={self.wind}, temp={self.temp}, pressure={self.pressure}, time={self.time})"
    

class ERROR (): 
    def __init__(self, MODEL: MEASUREMENT, CAST:MEASUREMENT): 
        self.MODEL = MODEL
        self.CAST = CAST

        self.error = self._estimation_error

    
    @property
    def _estimation_error(self)->np.array:
        """
        Calculate the estimation error of the current attributes mk. 
        Returns: 
        --------------------------
        The vector of estimation errors ek
        """
        print("Not implemented yet")

        
