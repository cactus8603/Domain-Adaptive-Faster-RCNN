import argparse
import yaml

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", default="config.yaml", nargs='?', help="path to config file")
    
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