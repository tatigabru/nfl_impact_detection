import os
import json
import pandas as pd
from typing import Dict, Union

DATA_DIR = "../../../archive/"

metric_dir = "checkpoints_metrics_impact_f1/"
metric_file = "_metrics.json"


def decode_json(json_file: str):
    """
    Helper, decodes json file
    """
    input_file = open(json_file, "r")
    try:
        json_decode = json.load(input_file)
    except:
        print(f"Can not decode: {input_file}")
        return

    return json_decode


def read_json_metrics(json_decode) -> Dict[str, float]:
    """
    Helper, reads json meta to a dictionary

    helmet_f1/fp

    impact_f1
    """
    s = "impact"
    json_dict = {}
    json_dict["f1"] = json_decode["best"][f"metrics/{s}_f1"]
    json_dict["tp"] = json_decode["best"][f"metrics/{s}_f1/tp"]
    json_dict["fp"] = json_decode["best"][f"metrics/{s}_f1/fp"]
    json_dict["fn"] = json_decode["best"][f"metrics/{s}_f1/fn"]
    json_dict["recall"] = recall(json_dict["tp"], json_dict["fn"])
    json_dict["precision"] = precision(json_dict["tp"], json_dict["fp"])

    return json_dict


def recall(tp: int, fn: int) -> float:
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def precision(tp: int, fp: int) -> float:
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


if __name__ == "__main__":

    folders = os.listdir(DATA_DIR)
    print(folders)

    # Create a dataframe to store meta
    df = pd.DataFrame(columns=["f1", "tp", "fp", "fn", "recall", "precision"])

    for folder in folders[:-1]:
        filepath = os.path.join(DATA_DIR, folder, metric_dir, metric_file)
        json_decode = decode_json(filepath)
        if not json_decode:
            continue
        json_dict = read_json_metrics(json_decode)
        df = df.append(json_dict, ignore_index=True)

    df["model"] = folders[:-1]
    print(df.head(50))
    df.to_csv(f"{DATA_DIR}/impact_det_metrics.csv", index=False)
