import json
import argparse
from trainer import train
import setproctitle

proc_name = 'ns_testms'

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    setproctitle.setproctitle(f'{proc_name}_{args["oflocation"]}')

    # for ms in range(800,4000,200):
    #     print(f"memory_size is {ms}")
    #     args["memory_size"]=ms
    #     train(args)
    for mp in range(20,30,10):
        print(f"memory_per_class is {mp}")
        args["memory_per_class"] = mp
        args["memory_size"]=mp*100
        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    main()
