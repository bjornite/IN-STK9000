from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

RUN_FILE = "TestRecommender.py"
num_rollouts = 1000
num_runs = 100
num_workers = 6

commands = []
for i in range(num_runs):
    commands.append("python {0} --run_id={1} --num_rollouts={2} --num_runs={3}".format(
            RUN_FILE,
            i,
            num_rollouts,
            1))

pool = Pool(num_workers)  # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
        print("%d command failed: %d" % (i, returncode))