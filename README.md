# Simple Mean Field Games

### Install Python packages in venv
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run example
Run a simple example, i.e. solving SIS problem for fixed-point iteration (FPI), fictitious play (FP) and online mirror descent (OMD).
```
python exp_1.py
```

### Plot example exploitabilities
Plot the simple example's exploitabilities and trajectories. For generated figures, see the figures folder.
```
python plot_1_exploitability.py
```

### Run custom experiments
Options can be found in args_parser.
```
python main_fp.py --game=SIS --variant=fp --softmax --temperature=0.01
```
