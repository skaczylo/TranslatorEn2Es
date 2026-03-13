"""
Script para convertir y unificar los dataset en un archivo .json

Creamos dos archivos:

    -full_dataset.json : alrededor de 2 millones de frases
    -dataset.json : subconjunto de 600k pares de frases
    
"""

import os
import json
import pandas as pd

tatoeba_path =os.path.join("dataset","Parejas de oraciones en InglésEspañol - 2026-03-13.tsv")
full_dataset_path = os.path.join("dataset","full_dataset.json")
europarl_en_path = os.path.join("dataset", "europarl-v7.es-en.en")
europarl_es_path = os.path.join("dataset", "europarl-v7.es-en.es")
dataset_path = os.path.join("dataset","dataset.json")


json_data = []

with open(tatoeba_path, 'r', encoding='utf-8') as tsvfile:
    for line in tsvfile:
        # Quitamos espacios en blanco extra y dividimos por tabulador
        columns = line.strip().split('\t')
            
        # Verificamos que la línea tenga las 4 columnas esperadas
        if len(columns) >= 4:
            entry = {
                "en": columns[1], # La segunda columna es el inglés
                "es": columns[3]  # La cuarta columna es el español
            }
            json_data.append(entry)


with open(europarl_en_path, 'r', encoding='utf-8') as f_en, open(europarl_es_path, 'r', encoding='utf-8') as f_es:
            
    # Recorremos ambos archivos de Europarl a la vez
    for line_en, line_es in zip(f_en, f_es):
            
        json_data.append({
            "en": line_en.strip(),
            "es": line_es.strip()
        })

# Guardamos el resultado en un archivo JSON
with open(full_dataset_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)


print(f"Se han procesado {len(json_data)} frases.")

#Reducimos a 600.000 frases

sample = 600000

# 1. Leer el archivo JSON original
df = pd.read_json(full_dataset_path)
df_600k = df.sample(n=sample, random_state=42)

# 4. Guardar en un nuevo archivo JSON
# 'orient=records' mantiene el formato de lista de diccionarios [{en:..., es:...}]
df_600k.to_json(dataset_path, orient='records', force_ascii=False, indent=2)
