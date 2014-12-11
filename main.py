import argparse
import json
import feature_descriptor
import cv2
from pprint import pprint

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
        #Using SIFT by default
        print images
        features = [feature_descriptor.find_features(image) for image in images]
        for i in range(1, len(features) - 1):
            feature_descriptor.find_matches(features[i], features[i + 1])

