from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 1, 'name': 'aeroplane'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'bird'},
    {'id': 4, 'name': 'boat'},
    {'id': 5, 'name': 'bottle'},
    {'id': 6, 'name': 'bus'},
    {'id': 7, 'name': 'car'},
    {'id': 8, 'name': 'cat'},
    {'id': 9, 'name': 'chair'},
    {'id': 10, 'name': 'cow'},
    {'id': 11, 'name': 'diningtable'},
    {'id': 12, 'name': 'dog'},
    {'id': 13, 'name': 'horse'},
    {'id': 14, 'name': 'motorbike'},
    {'id': 15, 'name': 'person'},
    {'id': 16, 'name': 'pottedplant'},
    {'id': 17, 'name': 'sheep'},
    {'id': 18, 'name': 'sofa'},
    {'id': 19, 'name': 'train'},
    {'id': 20, 'name': 'tvmonitor'},
]

def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_VOC = {
    "voc_cocoformat_test": ("voc/images/", "voc/annotations/pascal_test2007.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_VOC.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
