""" Convert dataset into WebDataset (WDS) format """
import argparse
import json
import os
import webdataset as wds
from tqdm import tqdm

def convert_to_wds(args):
    """ Convert dataset to wds format """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    with wds.ShardWriter(os.path.join(args.output_dir, 'pretrain-%d.tar'), maxcount=args.maxcount) as shard_writer:
        for entry in tqdm(data):
            image_path = entry.get('image') or entry.get('images')[0]
            with open(os.path.join(args.image_dir, image_path), "rb") as img_file:
                image_data = img_file.read()
            sample = {
                "__key__": entry.get('id') or image_path.split('.')[0],
                "jpg": image_data,
                "json": json.dumps(entry['messages']).encode("utf-8"),
            }
            shard_writer.write(sample)

    print(f"Dataset successfully converted to wds")


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments"""
    group = parser.add_argument_group(title='wds')
    group.add_argument('--output_dir', type=str, required=True, help='Output directory')
    group.add_argument('--json_file', type=str, required=True, help='Json file')
    group.add_argument('--image_dir', type=str, required=True, help='Image directory')
    group.add_argument('--maxcount', type=int, default=10000, help='Number of samples per shard')

    return parser


def parse_args():
    """arguments"""
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    """main function"""
    args = parse_args()
    convert_to_wds(args)


if __name__ == '__main__':
    main()