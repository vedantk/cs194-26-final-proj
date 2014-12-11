import argparse
import json
import feature_descriptor
import cv2
import math
from pprint import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 194-26 Final Project by Vedant Kumar, Melanie Cebula, and Siddhartho Bhattacharya")
    parser.add_argument("filename")

    args = parser.parse_args()

    json_filename = args.filename
    with open(json_filename) as f:
        #tuples of image names and camera pose (theta, phi, calibration matrix)
        data = json.load(f)
        #uncomment to print json data format
        #pprint(data)

        images = [cv2.imread(image) for image, _ in data]
        #NOTE: Using SIFT by default
        features = [feature_descriptor.find_features(image) for image in images]

        #Find all corr. points between adjacent pairs of images
        corr_pts = []
        theta = math.pi/2
        phi = math.pi/2
        for i in range(1, len(features) - 1):
            corr_pts.append(feature_descriptor.find_matches(features[i], features[i + 1]))

