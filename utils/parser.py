import argparse
import yaml

# python train.py --data_path dataset --save_path result_lr0001

def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    # parser.add_argument("--train_path", require=True, type=str, help='path to train dataset')
    # parser.add_argument("--val_path", require=True, type=str, help='path to validation dataset')
    parser.add_argument("--data_path", required=True, help='')
    parser.add_argument("--save_path", required=True, help='path to save model')
    # parser.add_argument("--use_ddp", default=True, help='dist training')
    # parser.add_argument("--device", default="0,1", help='gpu ids')
    parser.add_argument("--n_classes", default=8, type=int, help='object classes')
    # parser.add_argument("--port", default="8894", type=str, help='dist training port')
    parser.add_argument("--batch_size", default=1, type=int, help='batch size of dataloader')
    parser.add_argument("--num_workers", default=0, type=int, help='num_workers of dataloader')
    parser.add_argument("--seed", default=8603, type=int, help='random seed')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--momentum", default=0.937, type=float, help='')
    parser.add_argument("--lrf", default=0.0005, type=float, help='')
    parser.add_argument("--weight_decay", default=0.00005, type=float, help='')
    parser.add_argument("--cosanneal_cycle", default=50, type=int, help='')
    parser.add_argument("--epoch", default=120, type=int, help='total epoch')
    parser.add_argument("--accumulation", default=16, type=int, help='accumulation steps')
    
    # parser.add_argument("", default="", help='')
    args = parser.parse_args()
    return args

def parser_args():

    args = create_parser()

    if args.config_path:
        with open(args.config_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        arg_dict = args.__dict__
        for key, value in data.items():
            # print(key, value)
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].extend(value)
            else:
                arg_dict[key] = value
    return args
