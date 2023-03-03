from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 1, 'name': 'person'},
]


def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_CROWDHUMAN = {
    "crowdhuman_train": ("crowdhuman/CrowdHuman_train/Images/", "crowdhuman/annotations/train.json"),
    "crowdhuman_val": ("crowdhuman/CrowdHuman_val/Images/", "crowdhuman/annotations/val.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_CROWDHUMAN.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
