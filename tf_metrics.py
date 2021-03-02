import tensorflow as tf

def tf_f1_score(Y_true, Y_pred):
    """Computes macro averaged f1 score
    macro: mean of f1 scores per class

    Args:
        Y_true (Tensor): labels, with shape (batch, num_classes)
        Y_pred (Tensor): model's predictions, same shape as y_true

    Returns: macro averaged f1 score
    """

    # convert y_true into integers
    Y_true = tf.cast(Y_true, tf.int32)

    # get the class with the highest probability
    Y_pred_class = tf.math.argmax(Y_pred, axis=1)
    Y_pred_one_hot = tf.one_hot(Y_pred_class, Y_pred.shape[1])
    Y_pred = tf.cast(Y_pred_one_hot, tf.int32)

    # count true positives (TP), false neg (FN) and false pos (FP)
    TP = tf.math.count_nonzero(Y_pred * Y_true, axis=0)
    FP = tf.math.count_nonzero(Y_pred * (Y_true - 1), axis=0)
    FN = tf.math.count_nonzero((Y_pred - 1) * Y_true, axis=0)

    # tf.math.divide_no_nan accepts only floats
    TP = tf.cast(TP, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)

    # calc precision and recall, divide_no_nan is required to take care of
    # the case that the denominator becomes zero
    precision = tf.math.divide_no_nan(TP, TP + FP)
    recall = tf.math.divide_no_nan(TP, TP + FN)

    # calculate f1 score for each categorie
    f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)

    # take the macro average of all categories
    f1_macro_average = tf.reduce_mean(f1)

    return f1_macro_average


def tf_f1_score_sparse(y_true, Y_pred):
    """Computes macro averaged f1 score
    macro: mean of f1 scores per class

    Args:
        y_true (Tensor): sparse representation of labels, with shape (batch, 1)
        Y_pred (Tensor): model's predictions, same shape as y_true

    Returns: macro averaged f1 score
    """

    y_true = tf.cast(y_true, tf.int32)
    Y_true = tf.one_hot(y_true, Y_pred.shape[1])
    Y_true = tf.cast(Y_true, tf.int32)


    return tf_f1_score(Y_true, Y_pred)
