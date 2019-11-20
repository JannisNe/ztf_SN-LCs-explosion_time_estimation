import os
import json


sncosmo_fit_dir = os.path.dirname(os.path.realpath(__file__))

with open(sncosmo_fit_dir + '/model_names_dict.json') as f:
    sncosmo_model_names = json.load(f)


def get_sncosmo_fit_path():
    return f'{sncosmo_fit_dir}/sncosmo_fit.py'


