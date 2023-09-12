import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    # seed_list = copy.deepcopy(args["seed"])
    # device = copy.deepcopy(args["device"])

    # for seed in seed_list:
    #     args["seed"] = seed
    #     args["device"] = device
    #     _train(args)

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    args["device"] = device
    args["seed"] = int(seed_list)
        
    _train(args)


def _train(args):

    logs_name = "logs/{}/".format(args["model_name"])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_mz{}_mp{}".format(
        args["model_name"],
        args["logtips"],
        args["prefix"],
        args["seed"],
        args["model_name"],
        args["convnet_type"],
        args["dataset"],
        args["init_cls"],
        args["increment"],
        args["oflocation"],
        args["memory_size"],
        args["memory_per_class"],
    )
    handlers = logging.getLogger().handlers
    if len(handlers)==0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        handlers[0] = logging.FileHandler(filename=logfilename + ".log") #change a new logfile

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        # model.incremental_train(data_manager)
        if args["oflocation"]=='before':
            model.outfit_before_train(data_manager)
            logging.info("method: outfit before")
        elif args["oflocation"]=='after':
            model.outfit_after_train(data_manager)
            logging.info("method: outfit after")
        elif args["oflocation"]=='both':
            model.outfit_both_train(data_manager)
            logging.info("method: outfit both")
        elif args["oflocation"]=='pull':
            # model.pull_train(data_manager)
            model.pull_train_da(data_manager)
            logging.info("method: pull train")
        else:
            # model.incremental_train(data_manager)
            model.incremental_train_ex(data_manager)
            # logging.info("method: general incremental")
            logging.info("method: extend data incremental")

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            if "cuda" in "{}".format(device):
                pass
            else:
                device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
