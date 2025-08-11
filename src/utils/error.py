from __future__ import annotations
import numpy as np

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
