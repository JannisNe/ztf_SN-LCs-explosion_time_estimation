import logging
from estimate_explosion_time.shared import simulation_dir, plots_dir
from estimate_explosion_time.core.data_prep.data import DataHandler, load_dh
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter


logging.getLogger().setLevel("DEBUG")
logging.debug('logging level is DEBUG')

method = 'sncosmo_chi2'
simulation_name = 'simsurvey_simulation'

simsurvey_path = f'{simulation_dir}/{simulation_name}'

simsurveyDH = DataHandler(simsurvey_path, simulation_name)

fitter = Fitter(method)

fitter.fit_lcs(simsurveyDH)

# simsurveyDH = load_dh(simulation_name, method)

simsurveyDH.results()

