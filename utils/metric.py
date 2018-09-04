""" Collection of methods to compute the score.

1. We start with a true and predicted mask, corresponding to one train image.

2. The true mask is segmented into different objects. Here lies a main source 
of error. Overlapping or touching nuclei are not separated but are labeled as 
one object. This means that the target mask can contain less objects than 
those that have been originally identified by humans.

3. In the same manner the predicted mask is segmented into different objects.

4. We compute all intersections between the objects of the true and predicted 
masks. Starting with the largest intersection area we assign true objects to 
predicted ones, until there are no true/pred objects left that overlap. 
We then compute for each true/pred object pair their corresponding intersection 
over union (iou) ratio. 

5. Given some threshold t we count the object pairs that have an iou > t, which
yields the number of true positives: tp(t). True objects that have no partner are 
counted as false positives: fp(t). Likewise, predicted objects without a counterpart
a counted as false negatives: fn(t).

6. Now, we compute the precision tp(t)/(tp(t)+fp(t)+fn(t)) for t=0.5,0.55,0.60,...,0.95
and take the mean value as the final precision (score).
"""
import numpy as np
from skimage import morphology


min_object_size = 1

def get_labeled_mask(mask, cutoff=.5):
    """Object segmentation by labeling the mask."""
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    lab_mask = morphology.label(mask > cutoff)

    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = morphology.label(lab_mask > cutoff)

    return lab_mask


def get_iou(y_true_labeled, y_pred_labeled):
    """Compute non-zero intersections over unions."""
    # Array of different objects and occupied area.
    (true_labels, true_areas) = np.unique(y_true_labeled, return_counts=True)
    (pred_labels, pred_areas) = np.unique(y_pred_labeled, return_counts=True)

    # Number of different labels.
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    # Each mask has at least one identified object.
    if (n_true_labels > 1) and (n_pred_labels > 1):

        # Compute all intersections between the objects.
        all_intersections = np.zeros((n_true_labels, n_pred_labels))
        for i in range(y_true_labeled.shape[0]):
            for j in range(y_true_labeled.shape[1]):
                m = y_true_labeled[i, j]
                n = y_pred_labeled[i, j]
                all_intersections[m, n] += 1

                # Assign predicted to true background.
        assigned = [[0, 0]]
        tmp = all_intersections.copy()
        tmp[0, :] = -1
        tmp[:, 0] = -1

        # Assign predicted to true objects if they have any overlap.
        for i in range(1, np.min([n_true_labels, n_pred_labels])):
            mn = list(np.unravel_index(np.argmax(tmp), (n_true_labels, n_pred_labels)))
            if all_intersections[mn[0], mn[1]] > 0:
                assigned.append(mn)
            tmp[mn[0], :] = -1
            tmp[:, mn[1]] = -1
        assigned = np.array(assigned)

        # Intersections over unions.
        intersection = np.array([all_intersections[m, n] for m, n in assigned])
        union = np.array([(true_areas[m] + pred_areas[n] - all_intersections[m, n])
                          for m, n in assigned])
        iou = intersection / union

        # Remove background.
        iou = iou[1:]
        assigned = assigned[1:]
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]

        # Labels that are not assigned.
        true_not_assigned = np.setdiff1d(true_labels, assigned[:, 0])
        pred_not_assigned = np.setdiff1d(pred_labels, assigned[:, 1])

    else:
        # in case that no object is identified in one of the masks
        iou = np.array([])
        assigned = np.array([])
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]
        true_not_assigned = true_labels
        pred_not_assigned = pred_labels

    # Returning parameters.
    params = {'iou': iou,
              'assigned': assigned,
              'true_not_assigned': true_not_assigned,
              'pred_not_assigned': pred_not_assigned,
              'true_labels': true_labels,
              'pred_labels': pred_labels}
    return params


def get_score_summary(y_true, y_pred):
    """Compute the score for a single sample including a detailed summary."""

    y_true_labeled = get_labeled_mask(y_true)
    y_pred_labeled = get_labeled_mask(y_pred)

    params = get_iou(y_true_labeled, y_pred_labeled)
    iou = params['iou']
    assigned = params['assigned']
    true_not_assigned = params['true_not_assigned']
    pred_not_assigned = params['pred_not_assigned']
    true_labels = params['true_labels']
    pred_labels = params['pred_labels']
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    summary = []
    for i, threshold in enumerate(np.arange(0.5, 1.0, 0.05)):
        tp = np.sum(iou > threshold)
        fn = n_true_labels - tp
        fp = n_pred_labels - tp
        if (tp + fp + fn) > 0:
            prec = tp / (tp + fp + fn)
        else:
            prec = 0
        summary.append([threshold, prec, tp, fp, fn])

    summary = np.array(summary)
    score = np.mean(summary[:, 1])  # Final score.
    params_dict = {'summary': summary, 'iou': iou, 'assigned': assigned,
                   'true_not_assigned': true_not_assigned,
                   'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
                   'pred_labels': pred_labels, 'y_true_labeled': y_true_labeled,
                   'y_pred_labeled': y_pred_labeled}

    return score, params_dict


def get_score(y_true, y_pred):
    """Compute the score for a batch of samples."""
    scores = []
    for i in range(y_true.shape[0]):
        score, _ = get_score_summary(y_true[i], y_pred[i])
        scores.append(score)
    return np.array(scores)

