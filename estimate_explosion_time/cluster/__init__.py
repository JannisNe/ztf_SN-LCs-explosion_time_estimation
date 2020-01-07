import os
import time
import subprocess
import logging
import numpy as np
from tqdm import tqdm
from estimate_explosion_time.cluster.make_cluster_submit_script import make_desy_submit_file
from estimate_explosion_time.shared import log_dir, cache_dir, get_custom_logger, main_logger_name, TqdmToLogger


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
tqdm_deb = TqdmToLogger(logger, level=logging.DEBUG)
tqdm_info = TqdmToLogger(logger, level=logging.INFO)


username = os.path.basename(os.environ['HOME'])
cmd = f'qstat -u {username}'


def submit_to_desy(method_name, simulation_name=None, **kwargs):

    name = f'{method_name}_{simulation_name}'

    # remove old logs
    logger.debug('removing old log files')
    for file in tqdm(os.listdir(log_dir), desc='removing old log files', file=tqdm_deb, mininterval=30):
        if name in file:
            os.remove(log_dir + '/' + file)

    submit_file = make_desy_submit_file(method_name, **kwargs)
    submit_cmd = 'qsub -o {0} '.format(log_dir)

    if simulation_name:
        submit_cmd += f'-N {name} '

    submit_cmd += submit_file

    logger.debug(f'submit command is {submit_cmd}')

    # os.system(submit_cmd)

    process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
    msg = process.stdout.read().decode()
    logger.info(str(msg))
    job_id = int(str(msg).split('job-array')[1].split('.')[0])

    return job_id


def wait_for_cluster(job_id):
    """
    Runs the command cmd, which queries the status of the job on the
    cluster, and reads the output. While the output is not an empty
    string (indicating job completion), the cluster is re-queried
    every 30 seconds. Occasionally outputs the number of remaining sub-tasks
    on cluster, and outputs full table result every ~ 8 minutes. On
    completion of job, terminates function process and allows the script to
    continue.
    :param job_id: int, the ID of the job to be waited for
    :return: bool, if True the job has no tasks left
    """
    try:

        time.sleep(10)
        n_total = n_tasks(job_id)
        i = 31
        j = 6

        while n_total != 0:
            if i > 3:

                n_running = n_tasks(job_id, " -s r")
                n_waiting = n_tasks(job_id, " -s p")
                if n_waiting < 1:
                    waiting_str = 'no'
                else:
                    waiting_str = 'still'

                logger.info(f'{time.asctime(time.localtime())} - Job-ID {job_id}: '
                            f'{n_total} entries in queue. '
                            f'Of these, {n_running} are running tasks, and there are '
                            f'{waiting_str} jobs waiting to be executed.')
                i = 0
                j += 1

            if j > 7:
                n_tasks(job_id, print_full=True)
                j = 0

            time.sleep(30)
            i += 1
            n_total = n_tasks(job_id)

    finally:
        # check again if there are still jobs left which might be true in case of raised exception
        n_left = n_tasks(job_id)

        # if there are jobs left, delete them if desired by user
        if n_left > 0:

            inpt = input(f'Delete remaining tasks of job {job_id}?: ')
            if inpt in ['y', 'yes']:
                logger.info(f'deleting job {job_id}')
                delete_cmd = f'qdel {job_id}'
                os.system(delete_cmd)
                ret = True

            else:
                ret = False

        else:
            ret = True

    # give the system time to copy all results
    logger.info('waiting ...')
    time.sleep(20)

    return ret


def n_tasks(job_id=None, flags='', print_full=False):
    """
    Returns the number of tasks for a given job ID
    :param job_id: int, if given only tasks belonging to this job will we counted
    :param flags: str, optional, flags to be added to qstat
    :param print_full: bool, if True the full output of qstat will be printed
    :return: int, number of tasks left
    """

    process = subprocess.Popen(cmd+flags, stdout=subprocess.PIPE, shell=True)
    tmp = process.stdout.read().decode()

    st = str(tmp)
    ids = np.array([int(s.split(' ')[2]) for s in st.split('\n')[2:-1]])

    if print_full:
        logger.info('\n' + st)

    if job_id:
        return len(ids[ids == job_id])
    else:
        return len(ids)
