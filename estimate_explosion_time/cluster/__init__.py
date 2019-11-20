import os
import time
import subprocess
import logging
from estimate_explosion_time.cluster.make_cluster_submit_script import make_desy_submit_file
from estimate_explosion_time.shared import log_dir, cache_dir


username = os.path.basename(os.environ['HOME'])


def submit_to_desy(method_name, indir, outdir, njobs, cache=cache_dir, simulation_name=None):

    # remove old logs
    logging.debug('removing old log files')
    for file in os.listdir(log_dir):
        os.remove(log_dir + '/' + file)

    submit_file = make_desy_submit_file(method_name, indir, outdir, cache)

    submit_cmd = 'qsub ' \
                 '-t 1-{0} ' \
                 '-o {1} ' \
                 '-j y ' \
                 '-m ae ' \
                 '-l h_rss=4G ' \
                 '-l h_cpu=30:00:00 ' \
                 '-l s_rt=24:00:00 '.format(njobs, log_dir,)

    if simulation_name:
        submit_cmd += f'-N {method_name}_{simulation_name} '

    submit_cmd += submit_file

    logging.debug(f'submit command is {submit_cmd}')

    os.system(submit_cmd)


def wait_for_cluster():
    """Runs the command cmd, which queries the status of the job on the
    cluster, and reads the output. While the output is not an empty
    string (indicating job completion), the cluster is re-queried
    every 30 seconds. Occasionally outputs the number of remaining sub-tasks
    on cluster, and outputs full table result every ~ 8 minutes. On
    completion of job, terminates function process and allows the script to
    continue.
    """
    cmd = f'qstat -u {username}'
    time.sleep(10)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    tmp = process.stdout.read().decode()
    i = 31
    j = 6
    while tmp != '':
        if i > 3:

            n_total = len(str(tmp).split('\n')) - 3

            running_process = subprocess.Popen(
                cmd + " -s r", stdout=subprocess.PIPE, shell=True)
            running_tmp = running_process.stdout.read().decode()

            if running_tmp != '':
                n_running = len(running_tmp.split('\n')) - 3
            else:
                n_running = 0

            print(time.asctime(time.localtime()), n_total, "entries in queue. ", end=' ')
            print("Of these,", n_running, "are running tasks, and", end=' ')
            print(n_total-n_running, "are jobs still waiting to be executed.")
            print(time.asctime(time.localtime()), "Waiting for Cluster")
            i = 0
            j += 1

        time.sleep(30)
        i += 1
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        tmp = process.stdout.read().decode()
