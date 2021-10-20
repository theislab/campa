import os
import numpy as np

# --- set local paths ---
for BASE_DIR in [
    os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')),
    # insert other paths to base dir (containing local_data/raw_data dirs) 
]:
    if os.path.exists(BASE_DIR):
        print('Setting BASE_DIR to ' + BASE_DIR)
        break
if not os.path.exists(BASE_DIR):
    print('WARNING: BASE_DIR not found, setting to None')
    BASE_DIR = None
    
DATA_DIR = os.path.join(BASE_DIR, "local_data/NascentRNA")  
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "local_experiments")

# --- dataset specific constants ---
# name of column in metadata.csv that contains a unique object identifier
OBJ_ID = 'mapobject_id'

# --- conditions ---
# definition of conditions to be used for cVAE models.
# keys are column names in metadata.csv, and values are all possible values for this condition
# this will be used to convert conditions to one-hot encoded vector
CONDITIONS = {
    # 10 perturbation
    'perturbation_duration': ['AZD4573-120', 'AZD4573-30', 'CX5461-120', 'DMSO-120', 'DMSO-720',
    'Meayamycin-720', 'TSA-30', 'Triptolide-120', 'Wnt-C59-720',
    'normal'],
    # 4 cell cycle
    'cell_cycle': ['M', 'G1', 'S', 'G2'],
    # 28 wells
    'well_name': ['H06', 'H07', 'I03', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I14',
                    'I16', 'I17', 'I18', 'I20', 'J06', 'J07', 'J08', 'J09', 'J10', 'J12',
                    'J13', 'J14', 'J15', 'J16', 'J18', 'J20', 'J21', 'J22'],
    # 2 siRNA perturbation
    'siRNA': ['scrambled', 'SBF2'],
}
