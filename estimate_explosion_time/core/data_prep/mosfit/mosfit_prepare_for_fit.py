import pickle
import sncosmo_utils

with open('../simsurvey-paper-scripts/lcs/lcs_Ibc_nugent_000000.pkl', 'rb') as fin:
    sim = pickle.load(fin, encoding='latin1')

add_columns = {
    'instrument': 'ZTF_camera',
    'telescope': 'ZTF',
    'name': 'arb',
    'reference': 'JannisNecker',
    'u_time': 'MJD',
    'redshift': 'arb',
    'ebv': 'arb',
    'ID': 'arb'
}

sncosmo_utils.write_pkl_to_csv('../simsurvey-paper-scripts/lcs/lcs_Ibc_nugent_000000.pkl',
                               # folder_suffix='hehe',
                               # indices = [21],
                               add_columns=add_columns)

