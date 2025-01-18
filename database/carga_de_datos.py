import os

def save_data(data, file_path="database/datos.txt"):
    """
    Guarda los datos proporcionados en un archivo de texto.
    """
    with open(file_path, "w") as file:
        for key, value in data.items():
            file.write(f"{key}:{value}\n")

def load_data(file_path="database/datos.txt"):
    """
    Carga los datos desde un archivo de texto. Devuelve un diccionario con los datos cargados.
    Si el archivo no existe, devuelve un diccionario vacío.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = {}
            for line in file:
                key, value = line.strip().split(":")
                data[key] = int(value)
            return data
    else:
        print("No se encontró el archivo de estado. Usando valores predeterminados.")
        return {}