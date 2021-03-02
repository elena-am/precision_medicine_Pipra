import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history, metrics=['loss', 'accuracy', 'tf_f1_score']):
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, plt.ylim()[1]])

        plt.legend()


def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", square=True)
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()


def eval_model(model, x, y, batch_size=None):
    # print the evaluation metrics
    result_eval = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    for name, value in zip(model.metrics_names, result_eval):
        print(name, ': ', value)
    print()

    # plot a confusion matrix
    y_predicted = model.predict(x, batch_size=batch_size)
    if len(y_predicted.shape) > 1:
        y_predicted = y_predicted.argmax(axis=1)
    if len(y.shape) > 1:
        y = y.argmax(axis=1)

    # plot the confusion matrix
    plot_cm(y, y_predicted)
