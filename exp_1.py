import subprocess
import time
import numpy as np

if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []

    for variant in["RE_fp","NE_fp","QRE_fp","BE_fp"]:
        for game in ["SIS"]:
            p = subprocess.Popen(['python',
                                  './main_fp.py',
                                  f'--game={game}',
                                  f'--fp_iterations={2000}',
                                  f'--variant={variant}',
                                  f'--temperature={9.5}',
                                  f'--softmax',
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
