from airflow import DAG 
from airflow.operators.python import PythonOperator 
from datetime import datetime 
import pandas as pd 
import sqlite3 

# Programme qui scan un fichier tsv et le convertis en table de données sqlite
# Automatisation avec Apache Airflow mais pas nécessaire pour moi (NiFi, Hop)

DATABASE_PATH = 'C:\\Users\thier\Desktop\Metabase_Dashboard\dbeaver_base\gazdomestique.sqlite' 
CSV_FILE_PATH = 'C:\\Users\thier\Desktop\Metabase_Dashboard\Rag_Install_Diverse\estat_sdg_13_10.tsv/data.tsv' 
TABLE_NAME = 'Gaz domestique' 
def import_csv_to_sqlite(): 
  try: 
   conn = sqlite3.connect(DATABASE_PATH) 
   df = pd.read_csv(CSV_FILE_PATH, sep='\t') 
   df.to_sql(TABLE_NAME, conn, if_exists='append', index=False) 
   conn.close() 
   print(f"Le fichier CSV '{CSV_FILE_PATH}' a été importé avec succès dans la table '{TABLE_NAME}'.") 
  except Exception as e: 
        print(f"Une erreur est survenue lors de l'import du CSV : {e}") 
        raise  # Permet à Airflow de marquer la tâche comme failed
with DAG( 
    dag_id='simple_csv_to_sqlite', 
    start_date=datetime(2023, 1, 1), 
    schedule_interval=None,  # Je ne souhaite pas l'exécuter 
    catchup=False, 
    tags=['ingestion', 'sqlite', 'csv'], 
) as dag: 
    import_data = PythonOperator( 
        task_id='import_csv_data', 
        python_callable=import_csv_to_sqlite, 
    ) 
    
    