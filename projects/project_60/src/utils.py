import os
import sys
import re
import numpy as np
import brickschema


class Schema:
    point_label_col = 'PointLabel'
    ahu_col = 'UpstreamAHU'
    zone_col = 'ZoneName'
    vav_col = 'VAVName'
    brick_class_col = 'BrickClass'
    temp_col = '_'
    col_list = [point_label_col, ahu_col, zone_col, vav_col, brick_class_col]
    ahu_prefix = 'AHU_'
    vav_prefix = 'VAV_'


def random_idx(n):
    return np.random.randint(0, n)


metadata = {
    "name": "Brick Reconciliation Service",
    "defaultTypes": [
        {"id": "EquipmentClass", "name": "EquipmentClass"},
        {"id": "PointClass", "name": "PointClass"}
    ]
}

inf = brickschema.inference.TagInferenceSession(approximate=True)
# TODO: Provide framework to customize in config
LIMIT = 1


def flatten(lol):
    """flatten a list of lists"""
    return [x for sl in lol for x in sl]


def recon_api_inference(q):
    """
    q has fields:
    - query: string of the label that needs to be converted to a Brick type
    - type: optional list of 'types' (e.g. "PointClass" above)
    - limit: optional limit on # of returned candidates (default to 10)
    - properties: optional map of property idents to values
    - type_strict: [any, all, should] for strictness on the types returned
    """
    from abbrmap import abbrmap as tagmap # to avoid issues while running clean target
    # limit = int(q.get('limit', 10))
    # break query up into potential tags
    tags = map(str.lower, re.split(r'[.:\-_ ]', q))
    tags = list(tags)
    brick_tags = flatten([tagmap.get(tag.lower(), [tag]) for tag in tags])
    brick_tags = list(filter(lambda x: x != '', brick_tags))

    # not needed at this time
    # if q.get('type') == 'PointClass':
    #     brick_tags += ['Point']
    # elif q.get('type') == 'EquipmentClass':
    #     brick_tags += ['Equipment']

    # res = []
    most_likely, leftover = inf.most_likely_tagsets(brick_tags, LIMIT)
    # Disabling framework to return multiple likelihood scores
    # for ml in most_likely:
    #     res.append({
    #         'id': q,
    #         'name': ml,
    #         'score': (len(brick_tags) - len(leftover)) / len(brick_tags),
    #         'match': len(leftover) == 0,
    #         'type': [{"id": "PointClass", "name": "PointClass"}],
    #     })
    # print('returning', res)
    return most_likely[0]


def clean_extra_contents():
    os.system('rm -rf ../brick-builder ../reconciliation-api abbrmap.py output.ttl test/testdata/table_999_v1_processed.csv')
    # os.system('rm -rf src/building_depot data plot logs ./config/sensor_uuids.json')
