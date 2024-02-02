import subprocess
import time
import numpy as np

if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []

    for variant in ["fpi",]:
        for game in ['LR',]:
            for temperature in np.linspace(0.01, 1.01, 11):
                p = subprocess.Popen(['python',
                                      './main_fp_valuebased.py',
                                      f'--game={game}',
                                      f'--fp_iterations={100}',
                                      f'--variant={variant}',
                                      f'--temperature={temperature}',
                                      f'--softmax',
                                      ])
                child_processes.append(p)

                time.sleep(3)

                while len(child_processes) >= max_tasks:
                    for p in list(child_processes):
                        if p.poll() is not None:
                            child_processes.remove(p)
                        time.sleep(1)

    for p in child_processes:
        p.wait()
