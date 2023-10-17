# Modify
* Add model ema:
```
utils/model_ema.py
```
* Register custom dataset:
```
maskdino/data/datasets/register_hubmap_instance.py
maskdino/data/datasets/__init__.py
```
* Dataset mappers
```
maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
```
# Prepare dataset
* https://www.kaggle.com/quan0095/split-coco-kfold-cross-eval
* segmentation type: polygon (bbox mode: 1)
# Inference notebook
* https://www.kaggle.com/code/quan0095/inference-of-maskdino-p100
