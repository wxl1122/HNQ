import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from bisect import bisect
from scipy.ndimage import label
class FewShotMetric(object):

    '''
    Image-level and pixel-level metrics for few-shot segmentation
    including I-AUROC, I-F1, I-ACC, P-AUROC, P-F1, P-mIoU
    '''
    def __init__(self, products, device=None, gaussian_sigma=4.0, proxy_weight=0.5,
                 image_only=False, workers=1, metric_set='full'):
        if metric_set not in ('full', 'auroc') or (image_only and metric_set != 'full'):
            raise ValueError('Choose the full suite or image/pixel AUROCs')
        self.metric_set = metric_set
        if workers not in (1, 2, 4):
            raise ValueError('Metric workers must be 1, 2, or 4; full-resolution maps use substantial RAM')
        self.workers = workers
        self.products = products
        self.stat_buffer = {}
        self.device=device
        self.gaussian_sigma = gaussian_sigma
        self.proxy_weight = proxy_weight
        self.image_only = image_only
        for product in products:
            self.stat_buffer[product] = {}
            self.stat_buffer[product]['i_score'] = []
            self.stat_buffer[product]['i_gt'] = []
            self.stat_buffer[product]['p_score_map'] = []
            self.stat_buffer[product]['p_gt'] = []

    def update(self, i_logits, i_label, p_logits, p_label, products_list):
        # i_logits = i_logits.softmax(dim=1)
        # p_logits = p_logits.softmax(dim=1)
        for i, product in enumerate(products_list):
            if self.image_only:
                self.stat_buffer[product]['i_score'].append(i_logits[i, 1].item())
                self.stat_buffer[product]['i_gt'].append(i_label[i].item())
                continue
            score_map = (
                self.proxy_weight * p_logits[i, 1, ...]
                + (1.0 - self.proxy_weight) * (1.0 - p_logits[i, 0, ...])
            ).detach().cpu().numpy()
            score_map = gaussian_filter(score_map, self.gaussian_sigma)
            image_score = i_logits[i, 1].item()
            self.stat_buffer[product]['i_score'].append(image_score)
            self.stat_buffer[product]['i_gt'].append(i_label[i].item())
            self.stat_buffer[product]['p_score_map'].append(score_map)
            self.stat_buffer[product]['p_gt'].append(p_label[i].detach().cpu().numpy())

    def get_scores(self, report_progress=False):
        # Each category calculation reads only its own buffer. Keep concurrency
        # explicitly bounded: full-resolution PRO arrays have large RAM peaks.
        def calculate(item):
            index, product = item
            if report_progress:
                print(f'metrics category={index}/{len(self.products)} {product}', flush=True)
            return self.calculate_scores(product)

        indexed = list(enumerate(self.products, 1))
        if self.workers == 1:
            results = [calculate(item) for item in indexed]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # map preserves product order and propagates metric exceptions.
                results = list(executor.map(calculate, indexed))

        self.results = np.array(results)
        mean_results = self.results.mean(0)
        return tuple(mean_results)

    def calculate_scores(self, product):
        i_auroc, ap_clf, f1_clf = self.calculate_image_scores(product)
        if self.image_only:
            return i_auroc, ap_clf, f1_clf

        # pixel level
        p_score_map = np.stack(self.stat_buffer[product]['p_score_map'])
        p_gt = np.stack(self.stat_buffer[product]['p_gt'])
        p_auroc = roc_auc_score(p_gt.flatten(), p_score_map.flatten())
        if self.metric_set == 'auroc':
            return i_auroc, p_auroc

        pro_curve = compute_pro(anomaly_maps=p_score_map, ground_truth_maps=p_gt)
        au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=0.3) / 0.3
        precisions, recalls, _ = precision_recall_curve(p_gt.ravel(), p_score_map.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        return i_auroc, ap_clf, f1_clf, p_auroc, au_pro, f1_px

    def calculate_image_scores(self, product):
        # image level
        i_score = np.array(self.stat_buffer[product]['i_score'])
        i_gt = np.array(self.stat_buffer[product]['i_gt'])
        if len(np.unique(i_gt)) != 2:
            raise ValueError(f'{product}: image AUROC needs both normal and anomalous queries')
        i_auroc = roc_auc_score(i_gt, i_score)
        #ap
        ap_clf = average_precision_score(i_gt, i_score)
        #f1
        precisions, recalls, thresholds = precision_recall_curve(i_gt, i_score)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
        f1_clf = np.max(f1_scores[np.isfinite(f1_scores)])
        # i_auroc = roc_auc_score(i_gt, i_score)

        return i_auroc, ap_clf, f1_clf

    def print_metrics(self):
        metrics = self.results
        if self.metric_set == 'auroc':
            for product, row in zip(self.products, metrics):
                print(f'{product}: I-AUROC={100*row[0]:.4f}, P-AUROC={100*row[1]:.4f}')
            print(f'Mean I-AUROC/P-AUROC: {(100*metrics.mean(0)).tolist()}')
            return
        if self.image_only:
            print(f'Image-only mean I-AUROC/I-AP/I-F1: {metrics.mean(0).tolist()}')
            return
        print(f'{"Product":<20} {"I-AUROC":<20} {"I-AP":<20} {"I-F1_max":<20} {"P-AUROC":<20} {"AUPRO":<20} {"P-F1_max":<20}')
        for i, product in enumerate(self.products):
            print(f'{product:<20} {metrics[i][0]:<20.4f} {metrics[i][1]:<20.4f} {metrics[i][2]:<20.4f} {metrics[i][3]:<20.4f} {metrics[i][4]:<20.4f} {metrics[i][5]:<20.4f}')
        metrics_mean = metrics.mean(0)
        print(f'{"Mean":<20} {metrics_mean[0]:<20.4f} {metrics_mean[1]:<20.4f} {metrics_mean[2]:<20.4f} {metrics_mean[3]:<20.4f} {metrics_mean[4]:<20.4f} {metrics_mean[5]:<20.4f}')

    def reset(self):
        self.stat = np.zeros((self.n_class + 1, 3))     # +1 for bg, 3 for tp, fp, fn


def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    # print("Compute PRO curve...", end=" ")

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    # print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid(...)"
              " are finite. Will continue with only the finite values.")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction
