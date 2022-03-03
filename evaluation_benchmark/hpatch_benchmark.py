#
# Created by ZhangYuyang on 2020/6/27
#
import os
import yaml
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

from utils.evaluator import evaluate, Evaluator


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def benchmark_features(read_feats, dataset_path):
    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        _, keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            _, keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]


def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )


def generate_read_function(prediction_path, method):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(prediction_path, method, seq_name, '%d.npz' % im_idx))
        if top_k is None:
            return aux['shape'], aux['keypoints'], aux['descriptors']
        else:
            assert ('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k:]
            return aux['shape'], aux['keypoints'][ids, :], aux['descriptors'][ids, :]

    return read_function


def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)


def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    if top_k is None:
        return keypoints, descriptors
    else:
        assert ('scores' in mat)
        ids = np.argsort(mat['scores'][0])[-top_k:]
        return keypoints[ids, :], descriptors[ids, :]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, required=True)

    args = args.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    methods = config['methods'].split(',')
    names = config['names'].split(',')
    colors = config['colors'].split(',')
    linestyles = config['linestyles'].split(',')
    metrics = config['metrics'].split(',')

    assert len(methods) == len(names) == len(colors) == len(linestyles)

    top_k = None

    dataset_path = '../evaluation_hpatch/hpatches_sequences/hpatches-sequences-release'
    prediction_path = config['feature_path']

    lim = [1, 15]
    rng = np.arange(lim[0], lim[1] + 1)

    if top_k is None:
        cache_dir = 'cache/cache'
    else:
        cache_dir = 'cache/cache-top'
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    errors = {}

    for method in methods:
        output_file = os.path.join(cache_dir, method + '.npz')
      #  print(method)

        read_function = generate_read_function(prediction_path, method)

        if os.path.exists(output_file):
         #   print('Loading precomputed errors...')
            error = np.load(output_file, allow_pickle=True)
            error = {key: error[key].item() for key in error}
            errors[method] = error
        else:
            errors[method] = evaluate(read_function, dataset_path, evaluator=Evaluator())
            np.savez(output_file, **errors[method])

    def generate_plt(metric):
        print(metric, 'at 3,6,9 thr:')
        plt_lim = [1, 10]
        plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

        plt.rc('axes', titlesize=25)
        plt.rc('axes', labelsize=25)

        plt.figure(figsize=(15, 5))

        linewidth = 2
        plt.subplot(1, 3, 1)
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            i_err, v_err = errors[method]['i_err'][metric], errors[method]['v_err'][metric]
            n_i = errors[method]['i_count']
            n_v = errors[method]['v_count']
            plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / (n_i + n_v) for thr in plt_rng], color=color, ls=ls,
                     linewidth=linewidth, label=name)

            print(name,[(i_err[3] + v_err[3]) / (n_i + n_v)])
            print(name, [(i_err[6] + v_err[6]) / (n_i + n_v)])
            print(name, [(i_err[9] + v_err[9]) / (n_i + n_v)])

        if metric == 'MMA':
            plt.title('Overall')
         #   plt.ylabel('MMA (SS)')
        #else:
        plt.ylabel(metric)
        #plt.title('Overall')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend()

        plt.subplot(1, 3, 2)
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            i_err, v_err = errors[method]['i_err'][metric], errors[method]['v_err'][metric]
            n_i = errors[method]['i_count']
            plt.plot(plt_rng, [i_err[thr] / n_i for thr in plt_rng], color=color, ls=ls, linewidth=linewidth, label=name)
        if metric == 'MMA':
             plt.title('Illumination')
        # if metric == 'MMA':
        #     plt.xlabel('threshold [px]')
        #plt.title('Illumination')
        #plt.xlabel('threshold [px]')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.gca().axes.set_yticklabels([])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.subplot(1, 3, 3)
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            i_err, v_err = errors[method]['i_err'][metric], errors[method]['v_err'][metric]
            n_v = errors[method]['v_count']
            plt.plot(plt_rng, [v_err[thr] / n_v for thr in plt_rng], color=color, ls=ls, linewidth=linewidth, label=name)
        if metric == 'MMA':
             plt.title('Viewpoint')
        #plt.title('Viewpoint')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.gca().axes.set_yticklabels([])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)

        if top_k is None:
            plt.savefig('results/hseq_%s.pdf' % metric, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('results/hseq-top_%s.pdf' % metric, bbox_inches='tight', dpi=300)

    for met in metrics:
        generate_plt(met)



