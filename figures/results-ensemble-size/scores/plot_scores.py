import numpy as np
from matplotlib import pyplot as plt

from scores import *

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 16})

    max_size = len(mean_f1[0][0])

    m1 = np.mean(mean_f1[0], axis=0)
    s1 = np.mean(std_f1[0], axis=0)
    plt.plot(range(1, max_size + 1), m1, '-', color='darkviolet', label='GELECTRA (multi-label)')
    # plt.fill_between(range(1, max_size + 1), m1 - s1, m1 + s1,
    #                  linestyle='-', alpha=0.3, edgecolor='darkviolet', facecolor='violet')

    m2 = np.mean(mean_f1[1], axis=0)
    s2 = np.mean(std_f1[1], axis=0)
    plt.plot(range(2, max_size + 1, 2), m2, '-.', color='deepskyblue', label='GBERT + GELECTRA (multi-label)')
    plt.fill_between(range(2, max_size + 1, 2), m2 - s2, m2 + s2, linestyle='-.', alpha=0.3,
                     edgecolor='deepskyblue', facecolor='skyblue')

    m3 = np.mean(mean_f1[2], axis=0)
    s3 = np.mean(std_f1[2], axis=0)
    plt.plot(range(1, max_size + 1), m3, '--', color='blue', label='GBERT (multi-label)')
    # plt.fill_between(range(1, max_size + 1), m3 - s3, m3 + s3,
    #                  linestyle='--', alpha=0.3, edgecolor='blue', facecolor='cornflowerblue')

    m4 = np.mean(mean_f1[3], axis=0)
    s4 = np.mean(std_f1[3], axis=0)
    plt.plot(range(2, max_size + 1, 2), m4, '-*', color='green', label='GBERT + GELECTRA (single-label)')
    # plt.fill_between(range(2, max_size + 1, 2), m4 - s4, m4 + s4, linestyle='-.', alpha=0.3,
    #                  edgecolor='green', facecolor='green')

    plt.xlabel('ensemble size')
    plt.ylabel('macro-F1 score')
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig('results-ensemble-size_all-models.png', transparent=False, bbox_inches='tight', pad_inches=0)
    plt.close()
