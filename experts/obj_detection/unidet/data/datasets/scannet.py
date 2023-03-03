from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 3, 'name': 'cabinet', 'supercategory': 'furniture'}, 
    {'id': 4, 'name': 'bed', 'supercategory': 'furniture'}, 
    {'id': 5, 'name': 'chair', 'supercategory': 'furniture'}, 
    {'id': 6, 'name': 'sofa', 'supercategory': 'furniture'}, 
    {'id': 7, 'name': 'table', 'supercategory': 'furniture'}, 
    {'id': 8, 'name': 'door', 'supercategory': 'furniture'}, 
    {'id': 9, 'name': 'window', 'supercategory': 'furniture'}, 
    {'id': 10, 'name': 'bookshelf', 'supercategory': 'furniture'}, 
    {'id': 11, 'name': 'picture', 'supercategory': 'furniture'}, 
    {'id': 12, 'name': 'counter', 'supercategory': 'furniture'}, 
    {'id': 14, 'name': 'desk', 'supercategory': 'furniture'}, 
    {'id': 16, 'name': 'curtain', 'supercategory': 'furniture'}, 
    {'id': 24, 'name': 'refrigerator', 'supercategory': 'appliance'}, 
    {'id': 28, 'name': 'shower curtain', 'supercategory': 'furniture'},
    {'id': 33, 'name': 'toilet', 'supercategory': 'furniture'},
    {'id': 34, 'name': 'sink', 'supercategory': 'appliance'}, 
    {'id': 36, 'name': 'bathtub', 'supercategory': 'furniture'}, 
    {'id': 39, 'name': 'otherfurniture', 'supercategory': 'furniture'},
]


def _get_builtin_metadata():
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(sorted(categories, key=lambda x: x['id']))}
    thing_classes = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_SCANNET = {
    "scannet_trainval": ("scannet/scannet_frames_25k/", "scannet/scannet_instances.json"),
    "scannet_train": ("scannet/scannet_frames_25k/", "scannet/scannet_instances_0.json"),
    "scannet_val": ("scannet/scannet_frames_25k/", "scannet/scannet_instances_1.json"),
    "scannet_test": ("scannet/scannet_frames_test/", "scannet/scannet_instances_test_image_info.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_SCANNET.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
