# Precision, recall and f1 score calculations

# Calculate TP/TN
def confusion_counts (y_true, y_pred):
    # Return (tp, tn, fp, fn)  for binary labels
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must match")

    tp = tn = fp = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1: 
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else: 
            raise ValueError("y_true and y_pred must contain only binary values (0 or 1)")
    return tp, tn, fp, fn

# Precision is TP / (TP + FP)
def precision(y_true, y_pred):
    tp, _, fp, _ = confusion_counts(y_true, y_pred)
    if tp + fp == 0:
        return 0.0  # Precision is undefined when there are no predicted positives
    return tp / (tp + fp)


# Recall is TP / (TP + FN)
def recall(y_true, y_pred):
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    if tp + fn == 0:
        return 0.0  # Recall is undefined when there are no actual positives
    return tp / (tp + fn)


# F1 score is the harmonic mean of precision and recall
def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0.0  # F1 score is undefined when both precision and recall are zero
    return 2 * (prec * rec) / (prec + rec)
