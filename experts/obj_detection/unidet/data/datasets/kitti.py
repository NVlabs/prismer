from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 24, 'name': 'person'},
    {'id': 25, 'name': 'rider'},
    {'id': 26, 'name': 'car'},
    {'id': 27, 'name': 'truck'},
    {'id': 28, 'name': 'bus'},
    {'id': 31, 'name': 'train'},
    {'id': 32, 'name': 'motorcycle'},
    {'id': 33, 'name': 'bicycle'},
]

def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_ = {
    "kitti_train": ("kitti/data_semantics/training/image_2/", "kitti/train.json"),
    "kitti_test": ("kitti/data_semantics/testing/image_2/", "kitti/kitti_test.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
