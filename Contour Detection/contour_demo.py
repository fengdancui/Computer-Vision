# Code from Saurabh Gupta
from tqdm import tqdm
import os, sys, numpy as np, cv2

sys.path.insert(0, 'pybsds')
from scipy import signal
from skimage.util import img_as_float
from skimage.io import imread
from pybsds.bsds_dataset import BSDSDataset
from pybsds import evaluate_boundaries
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

GT_DIR = os.path.join('contour-data', 'groundTruth')
IMAGE_DIR = os.path.join('contour-data', 'images')
N_THRESHOLDS = 99


def get_imlist(name):
    imlist = np.loadtxt('contour-data/{}.imlist'.format(name))
    return imlist.astype(np.int)


def compute_edges_dxdy(I):
    """Returns the norm of dx and dy as the edge response function."""
    I = I.astype(np.float32) / 255
    dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
    dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')

    # after Q1 another way for convolution
    # dx, dy = sobel_edge_dxdy(I)

    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = mag / np.max(mag)
    mag = mag * 255
    mag = np.clip(mag, 0, 255)
    mag = mag.astype(np.uint8)
    return mag


def sobel_edge_dxdy(gray):
    dx = signal.convolve2d(gray, np.mat([1, 2, 1]).T * np.mat([1, 0, -1]), mode='same')
    dy = signal.convolve2d(gray, np.mat([1, 0, -1]).T * np.mat([1, 2, 1]), mode='same')

    return dx, dy


def canny_edge(gray):
    # gaussian blur
    gray_gau = cv2.GaussianBlur(gray, (3, 3), 0.8)

    # sobel edge detection
    dx, dy = sobel_edge_dxdy(gray_gau)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    h, w = mag.shape
    non_max = np.zeros((h, w))

    # non - maximum suppression and interpolation
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mag[i, j] != 0:
                grad_x = dx[i, j]
                grad_y = dy[i, j]

                # if the gradient along y is larger
                # the derivative tends to y

                if np.abs(grad_x) < np.abs(grad_y):
                    weight = np.abs(grad_x) / np.abs(grad_y)
                    grad2 = mag[i - 1, j]
                    grad4 = mag[i + 1, j]

                    # if the signs of both derivate of x and y are same
                    # the position of pixel are:
                    # g1    g2
                    #       c
                    #       g4    g3
                    if grad_x * grad_y > 0:
                        grad1 = mag[i - 1, j - 1]
                        grad3 = mag[i + 1, j + 1]

                    # the signs are different
                    #       g2    g1
                    #       c
                    # g3    g4
                    else:
                        grad1 = mag[i - 1, j + 1]
                        grad3 = mag[i + 1, j - 1]

                # if the gradient along x is larger
                else:
                    weight = np.abs(grad_y) / np.abs(grad_x)
                    grad2 = mag[i, j - 1]
                    grad4 = mag[i, j + 1]

                    # signs same
                    #             g3
                    # g2    c     g4
                    # g1
                    if grad_x * grad_y > 0:
                        grad1 = mag[i + 1, j - 1]
                        grad3 = mag[i - 1, j + 1]

                    # signs different
                    # g1
                    # g2    c     g4
                    #             g3
                    else:
                        grad1 = mag[i - 1, j - 1]
                        grad3 = mag[i + 1, j + 1]

                # interpolation
                mag1 = weight * grad1 + (1 - weight) * grad2
                mag2 = weight * grad3 + (1 - weight) * grad4

                if mag[i, j] >= mag1 and mag[i, j] >= mag2:
                    non_max[i, j] = mag[i, j]

    mag = non_max * 255 / np.max(non_max)
    out = np.clip(mag, 0, 255).astype(np.uint8)
    return out


def lab_edge(I):
    lab = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_edge = canny_edge(l)
    a_edge = canny_edge(a)
    b_edge = canny_edge(b)
    merge = np.copy(l_edge)
    merge[a_edge == 255] = 255
    merge[b_edge == 255] = 255

    return merge


def detect_edges(imlist, fn, out_dir):
    for imname in tqdm(imlist):
        I = cv2.imread(os.path.join(IMAGE_DIR, str(imname) + '.jpg'))
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        # after Q2 gaussian blur
        # mag = fn(cv2.GaussianBlur(gray, (3, 3), 0.8))

        # after Q3 non - maximum suppression and interpolation
        # mag = canny_edge(gray)

        # after Q4 using LAB for edge detection
        # mag = lab_edge(I)

        # original simlpe method
        mag = fn(gray)
        out_file_name = os.path.join(out_dir, str(imname) + '.png')
        cv2.imwrite(out_file_name, mag)


def load_gt_boundaries(imname):
    gt_path = os.path.join(GT_DIR, '{}.mat'.format(imname))
    return BSDSDataset.load_boundaries(gt_path)


def load_pred(output_dir, imname):
    pred_path = os.path.join(output_dir, '{}.png'.format(imname))
    return img_as_float(imread(pred_path))


def display_results(ax, f, im_results, threshold_results, overall_result):
    out_keys = ['threshold', 'f1', 'best_f1', 'area_pr']
    out_name = ['threshold', 'overall max F1 score', 'average max F1 score',
                'area_pr']
    for k, n in zip(out_keys, out_name):
        print('{:>20s}: {:<10.6f}'.format(n, getattr(overall_result, k)))
        f.write('{:>20s}: {:<10.6f}\n'.format(n, getattr(overall_result, k)))
    res = np.array(threshold_results)
    recall = res[:, 1]
    precision = res[recall > 0.01, 2]
    recall = recall[recall > 0.01]
    label_str = '{:0.2f}, {:0.2f}, {:0.2f}'.format(
        overall_result.f1, overall_result.best_f1, overall_result.area_pr)
    # Sometimes the PR plot may look funny, such as the plot curving back, i.e,
    # getting a lower recall value as you lower the threshold. This is because of
    # the lack on non-maximum suppression. The benchmarking code does some
    # contour thinning by itself. Unfortunately this contour thinning is not very
    # good. Without having done non-maximum suppression, as you lower the
    # threshold, the contours become thicker and thicker and we lose the
    # information about the precise location of the contour. Thus, a thined
    # contour that corresponded to a ground truth boundary at a higher threshold
    # can end up far away from the ground truth boundary at a lower threshold.
    # This leads to a drop in recall as we decrease the threshold.
    ax.plot(recall, precision, 'r', lw=2, label=label_str)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')


if __name__ == '__main__':

    imset = 'val'
    imlist = get_imlist(imset)
    output_dir = 'contour-output/demo'
    fn = compute_edges_dxdy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Running detector:')
    detect_edges(imlist, fn, output_dir)

    _load_pred = lambda x: load_pred(output_dir, x)
    print('Evaluating:')
    sample_results, threshold_results, overall_result = \
        evaluate_boundaries.pr_evaluation(N_THRESHOLDS, imlist, load_gt_boundaries,
                                          _load_pred, fast=True, progress=tqdm)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    file_name = os.path.join(output_dir + '_out.txt')
    with open(file_name, 'wt') as f:
        display_results(ax, f, sample_results, threshold_results, overall_result)
    fig.savefig(os.path.join(output_dir + '_pr.pdf'), bbox_inches='tight')
