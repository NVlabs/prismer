from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 13, 'name': 'trafficlight', 'supercategory': ''}, 
    {'id': 16, 'name': 'firehydrant', 'supercategory': ''}, 
    {'id': 17, 'name': 'chair', 'supercategory': ''}, 
    {'id': 19, 'name': 'trashcan', 'supercategory': ''}, 
    {'id': 20, 'name': 'person', 'supercategory': ''}, 
    {'id': 23, 'name': 'motorcycle', 'supercategory': ''}, 
    {'id': 24, 'name': 'car', 'supercategory': ''}, 
    {'id': 25, 'name': 'van', 'supercategory': ''}, 
    {'id': 26, 'name': 'bus', 'supercategory': ''}, 
    {'id': 27, 'name': 'truck', 'supercategory': ''},
]


def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_VIPER = {
    "viper_train": ("viper/train/img", "viper/train/viper_instances_train.json"),
    "viper_val": ("viper/val/img", "viper/val/viper_instances_val.json"),
    "viper_test": ("viper/test/img", "viper/test/viper_instances_test_image_info.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIPER.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
