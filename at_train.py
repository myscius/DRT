import json
import argparse
from trainer import train
import setproctitle
import numpy as np
import random

tips = "exp_ex_Inc50_ep70" # n1 is not norm to 1 
proc_name = tips

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
    # for mp in range(20,30,10):
    #     print(f"memory_per_class is {mp}")
    #     args["memory_per_class"] = mp
    #     args["memory_size"]=mp*100
    #     train(args)
    for i in range(1):
        args["logtips"] = f'{tips}_{i}'
        # args["init_cls"] = 50
        # args["increment"] = 50
        # args["addlist"] = genetate_atlist(70,1,1)
        # args["addlist"]  = random_atlist(70)
        # args["addlist"]  = []
        args["addlist"]  = get_atlist(70,0,0)
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

def genetate_atlist(ep,firstd,secondd):
    # assert firstd in [0,-1,1]
    if firstd==0 and secondd==0:
        return np.array([1]*ep)

    at_list = []
    x = 0
    top = 10 #max Derivative

    # init fd0
    fd = random.uniform(0.00001,top)

    for i in range(ep):
        at_list.append(x)
        x+=fd
        if secondd>0:
            sd = random.uniform(0.00001,top) 
            fd+= sd
        elif secondd<0:
            sd = random.uniform(0.00001,top) 
            fd*=(1-sd/top)
        elif secondd==0:
            pass

    if firstd<0:
        at_list = at_list[::-1]

    return np.array(at_list)

def get_atlist(ep,a,b):
    if a==0 and b==0:
        return np.array([1]*ep)
    #get a exact list
    x=np.arange(0,ep,1)
    pull_arr = a*(x/ep)**b
    # pull_arr = a*(x)**b
    return pull_arr

def random_atlist(ep):
    at_list = []
    top = 10
    for i in range(ep):
        x = random.uniform(0.00001,top)
        at_list.append(x)
    return np.array(at_list)

if __name__ == '__main__':
    main()
