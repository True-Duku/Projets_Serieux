import os
import pandas as pd

# Spécifiez le chemin du dossier contenant les fichiers Excel
input_folder = 'C:\\Users\\thier\\Desktop\\Excel_Ingestion\\Classeurs_Excel'
# Spécifiez le chemin du dossier où les fichiers CSV seront stockés
output_folder = 'C:\\Users\\thier\\Desktop\\Excel_Ingestion\\Fichiers_CSV'

# Créez le dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parcourez tous les fichiers dans le dossier d'entrée
for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # Chargez le fichier Excel
        excel_file_path = os.path.join(input_folder, filename)
        xls = pd.ExcelFile(excel_file_path)

        # Parcourez tous les onglets dans le fichier Excel
        for sheet_name in xls.sheet_names:
            # Chargez l'onglet dans un DataFrame
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Créez un nom de fichier pour le CSV basé sur le nom de l'onglet
            csv_filename = f"{os.path.splitext(filename)[0]}_{sheet_name}.csv"
            csv_file_path = os.path.join(output_folder, csv_filename)

            # Enregistrez le DataFrame en tant que fichier CSV
            df.to_csv(csv_file_path, index=False)
            print(f"Converti {sheet_name} de {filename} en {csv_filename}")

print("Conversion terminée !")