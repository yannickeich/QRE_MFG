import subprocess
import time
import numpy as np

if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []

    iterations = 100
    for variant in["QRE","RE","NE","BE"]:
        for method in ["FPI"]:
            if variant  == "NE":
                method = "pFP"

            for temperature in [0.1,0.5,1.0,2.0,3.0]:
                for game in ["random"]:
                    p = subprocess.Popen(['python',
                                              '../.././main_fp.py',
                                              f'--game={game}',
                                              f'--fp_iterations={iterations}',
                                              f'--method={method}',
                                              f'--temperature={temperature}',
                                              f'--variant={variant}',
                                              ])
                    child_processes.append(p)

                    #time.sleep(1)

                    while len(child_processes) >= max_tasks:
                            for p in list(child_processes):
                                if p.poll() is not None:
                                    child_processes.remove(p)
                                time.sleep(1)

    for p in child_processes:
        p.wait()
