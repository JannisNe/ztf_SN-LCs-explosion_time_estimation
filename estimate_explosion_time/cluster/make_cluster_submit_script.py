import logging
import os
import math
from estimate_explosion_time.shared import \
    root_dir, es_scratch_dir, activate_path, environment_path, mosfit_environment_path, \
    get_custom_logger, main_logger_name
from estimate_explosion_time.core.fit_data.sncosmo import get_sncosmo_fit_path


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
multiprocess_dir = os.path.dirname(os.path.realpath(__file__))
desy_submit_file = f'{multiprocess_dir}/submitDESY.sh'


def make_desy_submit_file(method_name, indir, outdir, ntasks, cache,
                          hrss='8G', hcpu='23:59:00', tasks_in_group=100):

    njobs = int(math.ceil( ntasks / tasks_in_group))

    if 'sncosmo' in method_name:

        script = get_sncosmo_fit_path()
        path_to_environment = environment_path

        hrss = '1G'

        fill = "    OUTNAME=${c}.pkl \n" \
               "    echo 'doing sncosmo fit' \n" \
              f"    python {script} $c $INDIR $OUTNAME"

        if 'mcmc' in method_name:
            fill += f' --method mcmc'

        if 'nester' in method_name:
            fill += f' --method nester'

    elif 'mosfit' in method_name:

        path_to_environment = mosfit_environment_path

        iterations = 500
        extrapolate = [20, 0]
        walkers = 100

        fill = '    OUTNAME=products/${c}.json \n' \
               '    echo "doing mosfit fit" \n' \
               '    printf \'n\\n%d\\nn\\nn\\nn\' $c | ' \
               'mosfit -e ${INDIR}/${c}.csv -m default --prefer-fluxes --quiet ' \
               f'-i {iterations} -E {extrapolate[0]} {extrapolate[1]} -N {walkers}'

    else:
        raise ValueError(f'Input {method_name} for method_name not understood!')

    txt1 = "#!/bin/zsh \n" \
           "#$-notify \n" \
           f"#$-t 1-{njobs} \n" \
           "#$-j y \n" \
           "#$-m a \n" \
           f"#$-l h_rss={hrss} \n" \
           f"#$-l h_cpu={hcpu} \n" \
           "\n" \
           "date \n" \
           "\n" \
           "ID=$SGE_TASK_ID \n" \
           "echo 'task-ID is ' $ID '\\n'\n" \
           f"ntasks={ntasks} \n" \
           f"tasks_in_group={tasks_in_group} \n" \
           f"INDIR={indir} \n" \
           f"OUTDIR={outdir} \n" \
           "\n" \
           f"export PYTHONPATH={root_dir} \n" \
           f"export EXPLOSION_TIME_ESTIMATION_SCRATCH_DIR={es_scratch_dir} \n" \
           f"source {activate_path} \n" \
           f"conda activate {path_to_environment} \n" \
           "\n" \
           "trap \'printf \"\\n\\n exiting normally\\n\";exit 0\' 0 \n" \
           "trap \'printf \"\\n\\n got notification after soft limit was reached \\n\"\' SIGUSR1 \n" \
           "trap \'printf \"\\n\\n got notification of coming kill \\n clean-up\\n\";" \
           "cd ..;" \
           "rm -r $TMPDIR;" \
           "exit 3\' SIGUSR2 \n" \
           "\n" \
           "echo 'PYTHONPATH is ' $PYTHONPATH \n" \
           "\n" \
           "itr1=$(expr $ID \* $tasks_in_group - $tasks_in_group + 1) \n" \
           "itr2=$(expr $ID \* $tasks_in_group) \n" \
           "echo 'iterating from ' $itr1 ' to ' $itr2 \n" \
           "\n" \
           "for (( c=$itr1; c<=$itr2; c++ )) \n" \
           "do \n" \
           "    echo 'loop variable is ' $c '\\n'\n" \
           "    if [[ $c == $(expr $ntasks + 1) ]] \n" \
           "    then \n" \
           "        echo 'exceeded n_tasks at ' $c \n" \
           "        break \n" \
           "    fi \n" \
           "    TMPDIR=" + cache + "/tmp${c} \n" \
           "    mkdir $TMPDIR \n" \
           "    cd $TMPDIR \n"

    txt2 = "\n" \
           "    mv $OUTNAME $OUTDIR \n" \
           "    cd .. \n" \
           "    rm -r $TMPDIR \n" \
           "done \n" \
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
