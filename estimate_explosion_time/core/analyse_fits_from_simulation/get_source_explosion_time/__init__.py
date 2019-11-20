import os
import json
import logging
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time \
    import get_explosion_time_from_timeseries


find_explosion_time_dir = os.path.dirname(os.path.realpath(__file__))
explosion_time_dic_path = f'{find_explosion_time_dir}/explosion_times.json'


def get_explosion_time(source):
    """
    Get the explosion time from a spectral time series source
    :param source: str, SNCosmo source name or path to Spectral time series
    :return: float, explosion time relative to the source's t_0
    """

    with open(explosion_time_dic_path, 'r') as f:
        explosion_time_dict = json.load(f)

    # if the source has no entry already in the explosion time dictionary, get it and add it to the dictionary
    if not source in explosion_time_dict.keys():
        logging.info(f'getting explosion time for {source}')
        t_exp = get_explosion_time_from_timeseries(source)
        explosion_time_dict[source] = t_exp

        with open(explosion_time_dic_path, 'w') as f:
            json.dump(explosion_time_dict, f)

    return explosion_time_dict[source]
