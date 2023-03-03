from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES as coco_categories
from .objects365 import categories_v1 as objects365_categories
from .oid import categories as oid_categories
from .mapillary import categories as mapillary_categories

categories = {
    'coco': [x for x in coco_categories if x['isthing'] == 1],
    'objects365': objects365_categories,
    'oid': oid_categories,
    'mapillary': mapillary_categories,
}


if __name__ == '__main__':
    import json
    json.dump(categories, 'datasets/metadata/det_categories.json')