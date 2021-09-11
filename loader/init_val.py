# segmentation decoding colors
name2color = {"person": [[135, 169, 180]],
              "sidewalk": [[242, 107, 146]],
              "road": [[156, 198, 23], [43, 79, 150]],
              "sky": [[209, 247, 202]],
              "pole": [[249, 79, 73], [72, 137, 21], [45, 157, 177], [67, 266, 253], [206, 190, 59]],
              "building": [[161, 171, 27], [61, 212, 54], [151, 161, 26]],
              "car": [[153, 108, 6]],
              "bus": [[190, 225, 64]],
              "truck": [[112, 105, 191]],
              "vegetation": [[29, 26, 199], [234, 21, 250], [145, 71, 201], [247, 200, 111]]
              }

name2id = {"person": 1,
           "sidewalk": 2,
           "road": 3,
           "sky": 4,
           "pole": 5,
           "building": 6,
           "car": 7,
           "bus": 8,
           "truck": 9,
           "vegetation": 10}

id2name = {i: name for name, i in name2id.items()}

split_label_dict = {'train': [0, 1, 2, 4, 6, 7],
                    'val': [0, 1, 2, 4, 6, 7],
                    'test': [0, 3, 5, 8, 9, 10]}

# list of nodes on the maps (this needs for loading from the folder)
all_edges = [
    ((0, 0), (16, -74)),
    ((16, -74), (-86, -78)),
    ((-86, -78), (-94, -58)),
    ((-94, -58), (-94, 24)),
    ((-94, 24), (-143, 24)),
    ((-143, 24), (-219, 24)),
    ((-219, 24), (-219, -68)),
    ((-219, -68), (-214, -127)),
    ((-214, -127), (-336, -132)),
    ((-336, -132), (-335, -180)),
    ((-335, -180), (-216, -205)),
    ((-216, -205), (-226, -241)),
    ((-226, -241), (-240, -252)),
    ((-240, -252), (-440, -260)),
    ((-440, -260), (-483, -253)),
    ((-483, -253), (-494, -223)),
    ((-494, -223), (-493, -127)),
    ((-493, -127), (-441, -129)),
    ((-441, -129), (-443, -222)),
    ((-443, -222), (-339, -221)),
    ((-339, -221), (-335, -180)),
    ((-219, 24), (-248, 24)),
    ((-248, 24), (-302, 24)),
    ((-302, 24), (-337, 24)),
    ((-337, 24), (-593, 25)),
    ((-593, 25), (-597, -128)),
    ((-597, -128), (-597, -220)),
    ((-597, -220), (-748, -222)),
    ((-748, -222), (-744, -128)),
    ((-744, -128), (-746, 24)),
    ((-744, -128), (-597, -128)),
    ((-593, 25), (-746, 24)),
    ((-746, 24), (-832, 27)),
    ((-832, 27), (-804, 176)),
    ((-804, 176), (-747, 178)),
    ((-747, 178), (-745, 103)),
    ((-745, 103), (-696, 104)),
    ((-696, 104), (-596, 102)),
    ((-596, 102), (-599, 177)),
    ((-599, 177), (-747, 178)),
    ((-599, 177), (-597, 253)),
    ((-596, 102), (-593, 25)),
    ((-337, 24), (-338, 172)),
    ((-337, 172), (-332, 251)),
    ((-337, 172), (-221, 172)),
    ((-221, 172), (-221, 264)),
    ((-221, 172), (-219, 90)),
    ((-219, 90), (-219, 24)),
    ((-221, 172), (-148, 172)),
    ((-148, 172), (-130, 172)),
    ((-130, 172), (-57, 172)),
    ((-57, 172), (-57, 194)),
    ((20, 192), (20, 92)),
    ((20, 92), (21, 76)),
    ((21, 76), (66, 22)),
    ((66, 22), (123, 28)),
    ((123, 28), (123, 106)),
    ((123, 106), (123, 135)),
    ((123, 135), (176, 135)),
    ((176, 135), (176, 179)),
    ((176, 179), (210, 180)),
    ((210, 180), (210, 107)),
    ((210, 107), (216, 26)),
    ((216, 26), (118, 21)),
    ((118, 21), (118, 2)),
    ((118, 2), (100, -62)),
    ((100, -62), (89, -70)),
    ((89, -70), (62, -76)),
    ((62, -76), (28, -76)),
    ((28, -76), (16, -74)),
    ((16, -74), (14, -17)),
    ((-494, -223), (-597, -220)),
    ((-597, -128), (-493, -127)),
    ((-493, -127), (-493, 25)),
    ((-336, -132), (-337, 24)),
    ((14, -17), (66, 22)),
    ((-597, 253), (-443, 253)),
    ((-443, 253), (-332, 251)),
    ((-332, 251), (-221, 264)),
    ((-221, 264), (-211, 493)),
    ((-211, 493), (-129, 493)),
    ((-129, 493), (23, 493)),
    ((23, 493), (20, 274)),
    ((176, 274), (176, 348)),
    ((176, 348), (180, 493)),
    ((180, 493), (175, 660)),
    ((175, 660), (23, 646)),
    ((23, 646), (-128, 646)),
    ((-128, 646), (-134, 795)),
    ((-134, 795), (-130, 871)),
    ((-130, 871), (20, 872)),
    ((175, 872), (175, 795)),
    ((252, 799), (175, 795)),
    ((175, 795), (23, 798)),
    ((23, 798), (-134, 795)),
    ((-134, 795), (-128, 676)),
    ((-128, 676), (-129, 493)),
    ((23, 493), (23, 646)),
    ((23, 646), (23, 798)),
    ((23, 798), (20, 872)),
    ((-338, 172), (-332, 251)),
    ((-57, 255), (20, 255)),
    ((-57, 194), (20, 192)),
    ((20, 255), (20, 274)),
    ((20, 274), (176, 267)),
    ((23, 493), (180, 493)),
    ((176, 267), (176, 348))]

mean_rgb = {"airsim": [103.939, 116.779, 123.68], }

def label_region_n_compute_distance(i, path_tuple):
    begin = path_tuple[0]
    end = path_tuple[1]

    # computer distance
    distance = ((begin[0] - end[0]) ** 2 + (begin[1] - end[1]) ** 2) ** 0.5

    # label region
    if begin[0] <= -400 or end[0] < -400:
        region = 'suburban'
    else:
        if begin[1] >= 300 or end[1] >= 300:
            region = 'shopping'
        else:
            region = 'skyscraper'

    # update tuple
    path_tuple = (i,) + path_tuple + (distance, region,)

    return path_tuple
