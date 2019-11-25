import logging
import os
from estimate_explosion_time.shared import \
    root_dir, es_scratch_dir, activate_path, environment_path, mosfit_environment_path, \
    get_custom_logger, main_logger_name
from estimate_explosion_time.core.fit_data.sncosmo import get_sncosmo_fit_path


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
multiprocess_dir = os.path.dirname(os.path.realpath(__file__))
desy_submit_file = f'{multiprocess_dir}/submitDESY.sh'


def make_desy_submit_file(method_name, indir, outdir, cache):

    if 'sncosmo' in method_name:

        script = get_sncosmo_fit_path()
        path_to_environment = environment_path

        fill = "OUTNAME=${ID}.pkl \n" \
               f"python {script} $ID $INDIR $OUTNAME"

        if 'mcmc' in method_name:
            fill += f' --method mcmc'

        if 'nester' in method_name:
            fill += f' --method nester'

    elif 'mosfit' in method_name:

        path_to_environment = mosfit_environment_path

        iterations = 600
        extrapolate = [20, 0]
        walkers = 700

        fill = 'OUTNAME=products/${ID}.json \n' \
               'printf \'n\\n%d\\nn\\nn\\nn\' $ID | ' \
               'mosfit -e ${INDIR}/${ID}.csv -m default --prefer-fluxes --quiet ' \
               f'-i {iterations} -E {extrapolate[0]} {extrapolate[1]} -N {walkers}'

    else:
        raise ValueError(f'Input {method_name} for method_name not understood!')

    txt1 = "#!/bin/zsh \n" \
           "#-o /lustre/fs23/group/icecube/necker/software/job_logs/ \n" \
           "#-notify \n" \
           "#-j y \n" \
           "#-m ae \n" \
           "#-l h_rss=4G \n" \
           "#-l h_cpu=30:00:00 \n" \
           "#-l s_rt=24:00:00 \n" \
           "#-t 1-200 \n" \
           "\n" \
           "date \n" \
           "\n" \
           "ID=$SGE_TASK_ID \n" \
           f"INDIR={indir} \n" \
           f"OUTDIR={outdir} \n" \
           "TMPDIR=" + cache + "/tmp${ID} \n" \
           "\n" \
           "mkdir $TMPDIR \n" \
           "cd $TMPDIR \n" \
           f"export PYTHONPATH={root_dir} \n" \
           f"export EXPLOSION_TIME_ESTIMATION_SCRATCH_DIR={es_scratch_dir} \n" \
           f"source {activate_path} \n" \
           f"conda activate {path_to_environment} \n" \
           "\n" \
           "#trap \'printf \"\\n\\n exiting normally\\n\";exit 0\' 0 \n" \
           "#trap \'printf \"\\n\\n got notification after soft limit was reached \\n\"\' SIGUSR1 \n" \
           "#trap \'printf \"\\n\\n got notification of coming kill \\n commence clean-up\\n\";" \
           "printf \"%d\\n\" $ID >> ~/mosfitNotDone.txt;" \
           "cd ..;" \
           "rm -r $TMPDIR;" \
           "exit 3\' SIGUSR2 \n" \
           "\n"

    txt2 = "\n" \
           "mv $OUTNAME $OUTDIR \n" \
           "cd .. \n" \
           "rm -r $TMPDIR \n" \
           "exit 0 \n" \
           "\n" \
           "date "

    text = txt1 + fill + txt2

    logger.info(f'making submit file at {desy_submit_file}')
    logger.debug(f'root directory is {root_dir}')

    with open(desy_submit_file, 'w') as f:
        f.write(text)

    cmd = f'chmod +x {desy_submit_file}'
    os.system(cmd)

    return desy_submit_file


if __name__ == '__main__':
    make_desy_submit_file('mosfit', 'test_indir', 'test_outdir')
