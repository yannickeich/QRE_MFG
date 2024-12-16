import subprocess
import time
import numpy as np

if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []

    iterations = 10000
    temperature = 0.5

    time_steps = 10
    mu_0 = np.array([0.0,0.3,0.3,0.4])
    for script in ['../.././RH_main_fp.py','../.././parallel_RH_main_fp.py']:
    #for script in [ '../.././parallel_RH_main_fp.py']:
        for variant in["QRE",]:
            for method in ["pFP"]:
                for tau in [5]:
                    for game in ["RPS"]:
                        p = subprocess.Popen(['python',
                                                  script,
                                                  f'--game={game}',
                                                  f'--fp_iterations={iterations}',
                                                  f'--method={method}',
                                                  f'--temperature={temperature}',
                                                  f'--variant={variant}',
                                                  f'--tau={tau}',
                                                  f"--time_steps={time_steps}",
                                                  f"--mu_0={mu_0}"
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
