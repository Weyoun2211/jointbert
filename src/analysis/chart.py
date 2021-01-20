import numpy as np
from matplotlib import pyplot as plt


def average_weight_for_attribute(dataframe, attrname, tuple=None):
    attribute = dataframe[dataframe['attribute'] == attrname]

    if tuple is not None:
        attribute = attribute[attribute['tuple'] == tuple]

    return attribute['weight'].mean(), attribute['weight'].std()


def count_explation_with_attr(dataframe, attrname, tuple=None):
    attribute = dataframe[dataframe['attribute'] == attrname]
    if tuple is not None:
        attribute = attribute[attribute['tuple'] == tuple]

    return len(attribute['exp'].unique())


def count_explanation(dataframe):
    return len(dataframe['exp'].unique())


def chart(data, subplot, ylim, title="", w=True):
    bars = []

    spacing = 2

    N = np.ones((1,))
    ticks = []

    plt.subplot(*subplot)
    plt.grid(True)
    plt.ylim(ylim)

    attr_sorted = sorted(data['attribute'].unique(), reverse=True)

    for j, att in enumerate(attr_sorted, start=1):

        att_m, att_s = average_weight_for_attribute(data, att, None)
        count = count_explation_with_attr(data, att)
        alpha = count / count_explanation(data)
        color = (0.8, 0.1, 0, 1) if att_m < 0 else (0, 0.8, 0.1, 1)

        xs = ((N+spacing) * j) + np.arange(1)

        ticks.append(xs[0])

        alpha = alpha * 2.5 if w else 2

        plt.bar(xs,
                [att_m],
                color=[color],
                yerr=[att_s],
                width=2,
                align='center')

    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(ticks, attr_sorted, rotation=90)