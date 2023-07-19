import argparse

from env.SIS import SIS


def parse_args():
    parser = argparse.ArgumentParser(description="Simple-MFG")
    parser.add_argument('--game', help='game to solve', default="SIS")
    parser.add_argument('--fp_iterations', type=int, help='number of fp iterations', default=300)
    parser.add_argument('--id', type=int, help='experiment id', default=0)

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--softmax", action="store_true", default=False, help="Use softmax instead of argmax")
    parser.add_argument("--inf", action="store_true", default=False, help="infinite horizon")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature")
    parser.add_argument("--variant", default="fpi", choices=["fpi", "fp", "omd"])

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


def generate_config(args):
    return generate_config_from_kw(**vars(args))


def generate_config_from_kw(temperature=0.1, softmax=0, inf=False, **kwargs):
    kwargs['temperature'], kwargs['softmax'], kwargs['inf'] = temperature, softmax, inf

    kwargs['exp_dir'] = "./results/%s_%s_%d_%d_%f_%d" \
               % (kwargs['game'], kwargs['variant'], kwargs['fp_iterations'],
                  kwargs['inf'], kwargs['temperature'], kwargs['softmax'])
    kwargs['exp_dir'] += f"/"

    from pathlib import Path
    Path(f"{kwargs['exp_dir']}").mkdir(parents=True, exist_ok=True)

    if kwargs['game'] == 'SIS':
        kwargs['game'] = SIS
    else:
        raise NotImplementedError

    return kwargs


def parse_config():
    args = parse_args()
    return generate_config(args)
