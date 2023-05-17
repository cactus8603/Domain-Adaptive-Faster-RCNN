import argparse
import yaml

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("data_path", required=True, help='path to dataset')
    parser.add_argument("use_ddp", default=True, help='dist training')
    parser.add_argument("device", default="0,1,2,3", help='gpu ids')
    parser.add_argument("n_classes", default="8", help='object classes')
    parser.add_argument("port", default="8894", help='dist training port')
    parser.add_argument("batch_size", default="64", help='batch size of dataloader')
    parser.add_argument("num_workers", default="6", help='num_workers of dataloader')
    parser.add_argument("seed", default="8603", help='random seed')
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
