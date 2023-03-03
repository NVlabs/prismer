from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': i, 'name': x} for i, x in enumerate(
        ["person", "rider", "car", "truck","bus", "train", \
         "motorcycle", "bicycle"])
]

def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_CITYSCAPES = {
    "cityscapes_cocoformat_val": ("", "cityscapes/annotations/cityscapes_fine_instance_seg_val_coco_format.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_CITYSCAPES.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join(image_root),
    )
