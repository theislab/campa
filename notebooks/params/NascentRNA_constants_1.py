# Constants for NascentRNA dataset

# --- dataset specific constants ---
import os
from miann.constants import BASE_DATA_DIR

# --- dataset specific constants ---
DATA_DIR = BASE_DATA_DIR
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
# name of column in metadata.csv that contains a unique object identifier
OBJ_ID = 'mapobject_id'
# name of csv file containing channels metadata (relative to DATA_DIR).
# is expected to contain channel names in column "name".
CHANNELS_METADATA = 'channels_metadata.csv'

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

