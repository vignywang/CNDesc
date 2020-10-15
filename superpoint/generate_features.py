#!/usr/bin/env python3

import argparse
import numpy as np
import os
import shutil
from tqdm import tqdm
import types
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from reconstruction_pipeline import recover_database_images_and_ids

from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from SuperPointPretrainedNetwork.demo_superpoint import myjet

#from r2d2.extract import r2d2_set_gpu, r2d2_load_model, r2d2_extract_keypoints

def export_features(images, paths, args):
    # Export the features.
    print('Exporting features...')

    for image_name, _ in tqdm(images.items(), total=len(images.items())):
        image_path = os.path.join(paths.image_path, image_name)
        if os.path.exists(image_path):
            sift = cv.xfeatures2d.SIFT_create()
            img0 = cv.imread(image_path)
            gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
            keypoints, descs = sift.detectAndCompute(gray, None)
            kps = np.asarray([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            img = cv.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                   cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if args.plot:
                _, (ax1, ax2) = plt.subplots(2, 1)
                ax1.imshow(img)
                ax2.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
                ax2.scatter(kps[:, 0], kps[:, 1])
                plt.show()
        features_path = os.path.join(paths.image_path, '%s.%s' % (image_name, args.method_name))
        #np.savez(features_path, keypoints=kps, descriptors=descs)


def export_super_features(images, paths, args):
    print('Exporting features...')
    super_net = SuperPointFrontend(weights_path=args.weights_path,
                                   nms_dist=4,
                                   conf_thresh=0.015,
                                   nn_thresh=0.7,
                                   cuda=True)
    for image_name, _ in tqdm(images.items(), total=len(images.items())):
        image_path = os.path.join(paths.image_path, image_name)
        if os.path.exists(image_path):
            img0 = cv.imread(image_path)
            gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
            input_image = gray.astype('float32') / 255.0
            kps, descs, heatmap = super_net.run(input_image)
            kps = kps.transpose()
            descs = descs.transpose()
            if args.plot:
                _, (ax1, ax2) = plt.subplots(2, 1)
                if heatmap is not None:
                    min_conf = 0.001
                    heatmap[heatmap < min_conf] = min_conf
                    heatmap = -np.log(heatmap)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
                    out3 = myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype('int'), :]
                    out3 = (out3 * 255).astype('uint8')
                    #print(heatmap.shape)
                    ax1.imshow(heatmap)
                ax2.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
                ax2.scatter(kps[:, 0], kps[:, 1])
                plt.show()
        #features_path = os.path.join(paths.image_path, '%s.%s' % (image_name, args.method_name))
        features_path = os.path.join('/data/localization/aachen/Aachen-Day-Night/scalepoint1.1/', '%s.%s' % (image_name.split('.')[0], args.method_name))
        #print((image_name.split('/')[:-1]))
        #Path('/data/localization/aachen/Aachen-Day-Night/superpoint/',str(image_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        #print(type(kps))
        #print(kps.shape)
        #np.savez(open(features_path, 'wb'), keypoints=kps, descriptors=descs)



def export_r2d2_features(images, paths, args):
    print('Exporting features...')
    iscuda = r2d2_set_gpu()
    r2d2_net = r2d2_load_model(args.weights_path, iscuda)
    top_k = 5000
    for image_name, _ in tqdm(images.items(), total=len(images.items())):
        image_path = os.path.join(paths.image_path, image_name)
        if os.path.exists(image_path):
            img0 = cv.imread(image_path)
            (kps, descs, scores) =  r2d2_extract_keypoints(r2d2_net, iscuda, image_path)
            idxs = scores.argsort()[-top_k or None:]
            kps = kps[idxs]
            descs = descs[idxs]
            scores = scores[idxs]

            idxs = np.nonzero(scores > 0.98)
            kps = kps[idxs]
            descs = descs[idxs]
            scores = scores[idxs]

            if args.plot:
                _, (ax1, ax2) = plt.subplots(2, 1)
                ax2.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
                ax2.scatter(kps[:, 0], kps[:, 1])
                plt.show()
        features_path = os.path.join(paths.image_path, '%s.%s' % (image_name, args.method_name))
        np.savez(features_path, keypoints=kps, descriptors=descs, scores=scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--method_name', required=True, help='Name of the method')
    parser.add_argument('--weights_path', type=str,
                        help='Path to pretrained weights file.')
    parser.add_argument('--plot', action='store_true')
    #parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(args.dataset_path, 'database_v1_1.db')
    # paths.database_path = os.path.join(args.dataset_path, 'aachen.db')
    paths.image_path = os.path.join(args.dataset_path, 'images', 'images_upright')

    images, _ = recover_database_images_and_ids(paths, args)
    # export_features(images, paths, args)
    export_super_features(images, paths, args)
   # export_r2d2_features(images, paths, args)

    return


if __name__ == "__main__":
    main()
