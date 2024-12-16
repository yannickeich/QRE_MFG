# Bounded Rationality Equilibrium Learning in Mean Field Games
Accompanying code for the paper "Bounded Rationality Equilibrium Learning in Mean Field Games" by Y. Eich, C. Fabian, K. Cui, and H. Koeppl.
Proceedings of the AAAI Conference on Artificial Intelligence, 39.

### Install Python packages in venv
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run example
Run a example, i.e. solving random problem with GFP for different equilibria.
First change directory to experiments/random
```
python exp1_GFP.py
```

### Plot example exploitabilities
Plot the results. For generated figures, see the figures folder. (The "figures" folder needs to be created first)
```
python plot1_GFP.py
```

### Run custom experiments
Options can be found in args_parser.
```
python main_fp.py --game=random --variant=QRE_fp  --temperature=0.1
```

### These figures from the paper are created by the following scripts

Figure 2 -> experiments/SIS/exp1_GFP.py -> experiments/SIS/plot1_GFP.py

Figure 3 -> experiments/RPS/exp4_simplex.py -> experiments/RPS/plot4_simplex.py

Figure 4 -> experiments/random/exp4_RH_QRE.py -> experiments/random/plot4_RH_QRE.py

Figure 5 -> experiments/random/exp2_GFPI.py -> experiments/random/plot2_GFPI.py

Figure 6 -> experiments/SIS/exp2_GFPI.py -> experiments/SIS/plot2_GFPI.py

Figure 7 -> experiments/RPS/exp2_GFPI.py -> experiments/SIS/plot2_GFPI.py

Figure 8 -> experiments/random/exp1_GFP.py -> experiments/random/plot1_GFP.py

Figure 9 -> experiments/RPS/exp1_GFP.py -> experiments/RPS/plot1_GFP.py

Figure 10 -> experiments/RPS/exp5_RH_seq_vs_parallel.py -> experiments/RPS/plot5_RH_seq_vs_parallel.py
