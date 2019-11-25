import os
import time
import subprocess
import logging
import numpy as np
from estimate_explosion_time.cluster.make_cluster_submit_script import make_desy_submit_file
from estimate_explosion_time.shared import log_dir, cache_dir, get_custom_logger, main_logger_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


username = os.path.basename(os.environ['HOME'])


def submit_to_desy(method_name, indir, outdir, njobs, cache=cache_dir, simulation_name=None):

    name = f'{method_name}_{simulation_name}'

    # remove old logs
    logger.debug('removing old log files')
    for file in os.listdir(log_dir):
        if name in file:
            logger.debug(f'removing {log_dir}/{file}')
            os.remove(log_dir + '/' + file)

    submit_file = make_desy_submit_file(method_name, indir, outdir, cache)

    submit_cmd = 'qsub ' \
                 '-t 1-{0} ' \
                 '-o {1} '.format(njobs, log_dir,)

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
    """

    try:

        cmd = f'qstat -u {username}'
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        tmp = process.stdout.read().decode()
        n_total = n_tasks(tmp, job_id)
        i = 31
        j = 6
        while n_total != 0:
            if i > 3:

                running_process = subprocess.Popen(
                    cmd + " -s r", stdout=subprocess.PIPE, shell=True)
                running_tmp = running_process.stdout.read().decode()

                if running_tmp != '':
                    n_running = n_tasks(running_tmp, job_id)
                else:
                    n_running = 0

                logger.info(f'{time.asctime(time.localtime())} - Job-ID {job_id}: '
                            f'{n_total} entries in queue. '
                            f'Of these, {n_running} are running tasks, and '
                            f'{n_total-n_running} are jobs still waiting to be executed.')
                i = 0
                j += 1

            if j > 7:
                logger.info(str(tmp))
                j = 0

            time.sleep(30)
            i += 1
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            tmp = process.stdout.read().decode()
            n_total = n_tasks(tmp, job_id)

    except KeyboardInterrupt:
        delete_cmd = f'qdel {job_id}'
        process = subprocess.Popen(delete_cmd, stdout=subprocess.PIPE, shell=True)
        msg = process.stdout.read().decode()
        logger.info(str(msg))


def n_tasks(tmp, job_id):
    """
    Returns the number of tasks given the output of qsub
    :param tmp: output of qsub
    :param job_id: int, optional, if given only tasks belonging to this job will we counted
    :return: int
    """
    st = str(tmp)
    ids = np.array([int(s.split(' ')[2]) for s in st.split('\n')[2:-1]])

    if job_id:
        return len(ids[ids == job_id])
    else:
        return len(ids)
