# DTMFGs

### Install Python packages in venv
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run example
Run a example, i.e. solving random problem with GFPI for different equilibria.
```
python exp_1_gfpi.py
```

### Plot example exploitabilities
Plot the example's exploitabilities For generated figures, see the figures folder. (The "figures" folder needs to be created first)
```
python plot_1_exploitability.py
```

### Run custom experiments
Options can be found in args_parser.
```
python main_fp.py --game=random --variant=QRE_fp  --temperature=0.1
```
