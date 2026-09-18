# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 01:18:24 2025

@author: Bruno
"""
import os

files = [f for f in os.listdir("./") if f.endswith('.txt')]

print(f"Procesando {len(files)} archivos...")

for filename in files:
    try:
        with open(filename, 'r') as f:
            lineas = f.readlines()
            if lineas[4].startswith("Iteraciones:"):
                lineas[4]="N"+lineas[4];
                    
                    
            with open(filename, 'w') as f:
                f.writelines(lineas)
                
    except Exception as e:
        print(f"Error leyendo {filename}: {e}")
