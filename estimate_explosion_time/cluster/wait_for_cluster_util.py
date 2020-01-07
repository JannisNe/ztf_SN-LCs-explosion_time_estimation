import logging
from estimate_explosion_time.shared import get_custom_logger, main_logger_name

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)

from estimate_explosion_time.cluster import wait_for_cluster
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('job_id', type=int)
args = parser.parse_args()
logger.info(f'waiting on job {args.job_id}')
left = wait_for_cluster(args.job_id)
