import subprocess
import time
import numpy as np

if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []

    for variant in["QRE","RE","NE","BE"]:
        for method in ["pFP"]:
            for tau in range(2,6):
                for temperature in [0.5]:
                    for game in ["RPS"]:
                        p = subprocess.Popen(['python',
                                              '../.././main_fp.py',
                                              f'--game={game}',
                                              f'--fp_iterations={1000}',
                                              f'--method={method}',
                                              f'--temperature={temperature}',
                                              f'--variant={variant}',
                                              f'--tau={tau}',
                                              #f'--lookahead',
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
