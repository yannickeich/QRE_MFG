import argparse

from env.SIS import SIS
from env.LR import LR
from env.RandomMFG import RandomMFG
from env.RPS import RPS
from env.A2_MDP import A2_MDP
from env.A3_MDP import A3_MDP
from env.riskRPS import riskRPS


def parse_args():
    parser = argparse.ArgumentParser(description="Simple-MFG")
    parser.add_argument('--game', help='game to solve', default="random")
    parser.add_argument('--fp_iterations', type=int, help='number of fp iterations', default=300)
    parser.add_argument('--id', type=int, help='experiment id', default=0)

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
   # parser.add_argument("--softmax", action="store_true", default=True, help="Use softmax instead of argmax")
    parser.add_argument("--inf", action="store_true", default=False, help="infinite horizon")
    parser.add_argument("--temperature", type=float, default=0.05, help="Softmax temperature")
    parser.add_argument("--variant", default="NE", choices=["NE", "BE","RE","QRE"])
    parser.add_argument("--method",default = "pFP", choices=["FPI","FP","expFPv1","expFPv2","pFP"])
    parser.add_argument("--lookahead",action="store_true", default=False, help="lookahead")
    parser.add_argument("--tau",type = int,default = 5,help="receding horizon")
    parsed, unknown = parser.parse_known_args()
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    def isint(num):
        try:
            int(num)
            return True
        except ValueError:
            return False
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0],
                                type=int if isint(arg.split('=')[1]) else float if isfloat(arg.split('=')[1]) else str)

    return parser.parse_args()


def generate_config(args,mf_method=None):
    return generate_config_from_kw(**vars(args),mf_method=mf_method)


def generate_config_from_kw(temperature=0.1, inf=False, mf_method=None,**kwargs):
    kwargs['temperature'] = temperature

    kwargs['exp_dir'] = "./results/%s_%s_%s_%d_%f_%d_%d" \
               % (kwargs['game'], kwargs['variant'],kwargs['method'], kwargs['fp_iterations'], kwargs['temperature'],kwargs['lookahead'],kwargs['tau'])
    kwargs['exp_dir'] += f"/"

    if mf_method == "RH":
        kwargs['exp_dir'] = kwargs['exp_dir'].replace("/results/", "/RH_results/")

    from pathlib import Path

    Path(f"{kwargs['exp_dir']}").mkdir(parents=True, exist_ok=True)


    if kwargs['game'] == 'SIS':
        kwargs['game'] = SIS
    elif kwargs['game'] == 'LR':
        kwargs['game'] = LR
    elif (kwargs['game'] == 'random')|(kwargs['game'] == "<class 'env.RandomMFG.RandomMFG'>"):
        kwargs['game'] = RandomMFG
    elif kwargs['game'] == 'RPS':
        kwargs['game'] = RPS
    elif kwargs['game'] == 'A2_MDP':
        kwargs['game'] = A2_MDP
    elif kwargs['game'] == 'A3_MDP':
        kwargs['game'] = A3_MDP
    elif kwargs['game'] == 'riskRPS':
        kwargs['game'] = riskRPS
    else:
        raise NotImplementedError

    return kwargs


def parse_config(mf_method=None):
    args = parse_args()
    return generate_config(args,mf_method)
