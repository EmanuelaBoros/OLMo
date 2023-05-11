import argparse
import json
import os
from typing import List

import pandas as pd

S3_location = "s3://ai2-llm/pretraining-data/sources/stack-dedup"


def merge(urls: List[str], output_path: str, version: str):
    if os.path.exists(output_path):
        return
    all_dfs = []
    for url in urls:
        path = os.path.join(S3_location, version, "attributes", url + ".tsv")
        try:
            df = pd.read_csv(path, sep="\t")
            all_dfs.append(df)
        except FileNotFoundError:
            import traceback

            traceback.print_exc()

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(output_path, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge file statistics")
    parser.add_argument("--lang-files-path", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    with open(args.lang_files_path) as f:
        lang_paths = json.load(f)

    merge(lang_paths[args.lang], args.output_path, args.version)