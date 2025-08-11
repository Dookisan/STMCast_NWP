from __future__ import annotations
import numpy as np
from utils.error import ERROR
import sys

def _lookup_models(): 
    print(sys.prefix)
    print("python looks for models in this locations:")

    for path in sys.path:
        print(f"-{path}")


def main():

    #_lookup_models()
        
    error = ERROR(wind=5, temp=20, pressure=1013, time=0)
    print(error)
    print(error.error)

if __name__ == "__main__":
    main()
