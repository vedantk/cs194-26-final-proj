import argparse
import json
from pprint import pprint
import feature_descriptor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 194-26 Final Project by Vedant Kumar, Melanie Cebula, and Siddhartho Bhattacharya")
    parser.add_argument("filename")

    args = parser.parse_args()

    json_filename = args.filename
    with open(json_filename) as f:
        data = json.load(f)
        #uncomment to print json data format
        #print(data)

        images = [cv2.imread(image) for image, _ in data]
        features = [feature_descriptor.find_features(image) for image in image]
        for feature in features:
            #pos is (row, col) in image
            #desc is description
            pos, desc = feature.pos, feature.desc

