import subprocess
import time
import numpy as np


if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []
    temperature = np.exp(1.7)
    for tau in [2]:
        for game in ['random',]:
            for variant in ['QRE', "RE", "BE"]:
                p = subprocess.Popen(['python',
                                      '../.././RH_main_fp.py',
                                      f'--game={game}',
                                      f'--fp_iterations={1000}',
                                      f'--method={"pFP"}',
                                      f'--variant={variant}',
                                      f'--tau={tau}',
                                      f'--temperature={temperature}',
                                      f'--lookahead',
                                      ])
                child_processes.append(p)

                time.sleep(1)

                while len(child_processes) >= max_tasks:
                    for p in list(child_processes):
                        if p.poll() is not None:
                            child_processes.remove(p)
                        time.sleep(1)

    for p in child_processes:
        p.wait()
