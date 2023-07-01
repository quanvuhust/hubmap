# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

HUBMAP_CATEGORIES = [{'id': 1, 'name': 'blood_vessel'}]


_PREDEFINED_SPLITS = {
    # point annotations without masks
    "hubmap_instance_train": (
        "/kaggle/input/hubmap-hacking-the-human-vasculature/train",
        "/kaggle/input/split-coco-kfold-cross-eval/hubmap_train_fold_0.json",
    ),
    "hubmap_instance_val": (
        "/kaggle/input/hubmap-hacking-the-human-vasculature/train",
        "/kaggle/input/split-coco-kfold-cross-eval/hubmap_val_fold_0.json",
    ),
}


def _get_hubmap_instances_meta():
    thing_ids = [k["id"] for k in HUBMAP_CATEGORIES]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous HUBMAP category id to an id in [0, 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in HUBMAP_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_hubmap_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_hubmap_instances_meta(),
            json_file,
            image_root,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hubmap_instance(_root)
