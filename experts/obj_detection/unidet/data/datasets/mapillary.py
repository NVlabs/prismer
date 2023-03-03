from detectron2.data.datasets.register_coco import register_coco_instances
import os

'''
categories = [
    {'id': 28, 'name': 'animal--bird'} ,
    {'id': 29, 'name': 'animal--ground-animal'} ,
    {'id': 30, 'name': 'construction--flat--crosswalk-plain'} ,
    {'id': 31, 'name': 'human--person'} ,
    {'id': 32, 'name': 'human--rider--bicyclist'} ,
    {'id': 33, 'name': 'human--rider--motorcyclist'} ,
    {'id': 34, 'name': 'human--rider--other-rider'} ,
    {'id': 35, 'name': 'marking--crosswalk-zebra'} ,
    {'id': 36, 'name': 'object--banner'} ,
    {'id': 37, 'name': 'object--bench'} ,
    {'id': 38, 'name': 'object--bike-rack'} ,
    {'id': 39, 'name': 'object--billboard'} ,
    {'id': 40, 'name': 'object--catch-basin'} ,
    {'id': 41, 'name': 'object--cctv-camera'} ,
    {'id': 42, 'name': 'object--fire-hydrant'} ,
    {'id': 43, 'name': 'object--junction-box'} ,
    {'id': 44, 'name': 'object--mailbox'} ,
    {'id': 45, 'name': 'object--manhole'} ,
    {'id': 46, 'name': 'object--phone-booth'} ,
    {'id': 47, 'name': 'object--street-light'} ,
    {'id': 48, 'name': 'object--support--pole'} ,
    {'id': 49, 'name': 'object--support--traffic-sign-frame'} ,
    {'id': 50, 'name': 'object--support--utility-pole'} ,
    {'id': 51, 'name': 'object--traffic-light'} ,
    {'id': 52, 'name': 'object--traffic-sign--back'} ,
    {'id': 53, 'name': 'object--traffic-sign--front'} ,
    {'id': 54, 'name': 'object--trash-can'} ,
    {'id': 55, 'name': 'object--vehicle--bicycle'} ,
    {'id': 56, 'name': 'object--vehicle--boat'} ,
    {'id': 57, 'name': 'object--vehicle--bus'} ,
    {'id': 58, 'name': 'object--vehicle--car'} ,
    {'id': 59, 'name': 'object--vehicle--caravan'} ,
    {'id': 60, 'name': 'object--vehicle--motorcycle'} ,
    {'id': 61, 'name': 'object--vehicle--other-vehicle'} ,
    {'id': 62, 'name': 'object--vehicle--trailer'} ,
    {'id': 63, 'name': 'object--vehicle--truck'} ,
    {'id': 64, 'name': 'object--vehicle--wheeled-slow'} ,
]
'''
categories = [
    {'id': 1, 'name': 'animal--bird'},
    {'id': 2, 'name': 'animal--ground-animal'},
    {'id': 9, 'name': 'construction--flat--crosswalk-plain'},
    {'id': 20, 'name': 'human--person'},
    {'id': 21, 'name': 'human--rider--bicyclist'},
    {'id': 22, 'name': 'human--rider--motorcyclist'},
    {'id': 23, 'name': 'human--rider--other-rider'},
    {'id': 24, 'name': 'marking--crosswalk-zebra'},
    {'id': 33, 'name': 'object--banner'},
    {'id': 34, 'name': 'object--bench'},
    {'id': 35, 'name': 'object--bike-rack'},
    {'id': 36, 'name': 'object--billboard'},
    {'id': 37, 'name': 'object--catch-basin'},
    {'id': 38, 'name': 'object--cctv-camera'},
    {'id': 39, 'name': 'object--fire-hydrant'},
    {'id': 40, 'name': 'object--junction-box'},
    {'id': 41, 'name': 'object--mailbox'},
    {'id': 42, 'name': 'object--manhole'},
    {'id': 43, 'name': 'object--phone-booth'},
    {'id': 45, 'name': 'object--street-light'},
    {'id': 46, 'name': 'object--support--pole'},
    {'id': 47, 'name': 'object--support--traffic-sign-frame'},
    {'id': 48, 'name': 'object--support--utility-pole'},
    {'id': 49, 'name': 'object--traffic-light'},
    {'id': 50, 'name': 'object--traffic-sign--back'},
    {'id': 51, 'name': 'object--traffic-sign--front'},
    {'id': 52, 'name': 'object--trash-can'},
    {'id': 53, 'name': 'object--vehicle--bicycle'},
    {'id': 54, 'name': 'object--vehicle--boat'},
    {'id': 55, 'name': 'object--vehicle--bus'},
    {'id': 56, 'name': 'object--vehicle--car'},
    {'id': 57, 'name': 'object--vehicle--caravan'},
    {'id': 58, 'name': 'object--vehicle--motorcycle'},
    {'id': 60, 'name': 'object--vehicle--other-vehicle'},
    {'id': 61, 'name': 'object--vehicle--trailer'},
    {'id': 62, 'name': 'object--vehicle--truck'},
    {'id': 63, 'name': 'object--vehicle--wheeled-slow'},
]


def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {categories[i]['id']: i for i in range(37)}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS = {
    "mapillary_train": ("mapillary/training/images/", "mapillary/annotations/training_fix_id.json"),
    # "mapillary_train": ("mapillary/training/images/", "mapillary/annotations/training.json"),
    "mapillary_val": ("mapillary/validation/images/", "mapillary/annotations/validation_fix_id.json"),
    # "mapillary_val": ("mapillary/validation/images/", "mapillary/annotations/validation.json"),
    "mapillary_960_train": ("mapillary/training/images960/", "mapillary/annotations/training960_fix_id.json"),
    'mapillary_test': ('mapillary/testing/images/', 'mapillary/annotations/test_image_info_fix_id.json')
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )

