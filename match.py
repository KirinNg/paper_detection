import cv2 # 3.4.2.17
import numpy as np
import os

MAX_FEATURES = 40000
GOOD_MATCH_PERCENT = 0.2
IN_PATH = "IN"
OUT_PATH = "OUT"
orb = cv2.ORB_create(MAX_FEATURES)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def alignImages(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg, h

if __name__ == '__main__':
    refFilename = "target.png"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    for dirpath, dirnames, filenames in os.walk(IN_PATH):
        for filepath in filenames:
            try:
                imFilename = os.path.join(IN_PATH, filepath)
                im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
                imReg, h = alignImages(im, imReference)
                outFilename = os.path.join(OUT_PATH, "out_" + filepath)
                cv2.imwrite(outFilename, imReg)
                print(filepath+"    success!")
            except:
                print(filepath+"    failed!")
