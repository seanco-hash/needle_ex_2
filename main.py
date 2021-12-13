from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
NUM_POINTS = 500
REPEAT = 2


def plot_2d_points(points, title, is_color=False, axis_lim=False, lim=(-2.5, 2.5)):
    if is_color:
        plt.scatter(points[:, 0], points[:, 1], 10, c=points[:, 2])
        cbar = plt.colorbar()
        cbar.set_label("gaussian center", labelpad=+1)
    else:
        plt.scatter(points[:, 0], points[:, 1], 10)
    plt.title(title)
    if axis_lim:
        plt.xlim(lim)
        plt.ylim(lim)
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.show()


def create_gaussian(mean, std, n_points=NUM_POINTS):
    points = np.random.normal(mean, std, (n_points, 2))
    return points


def create_uniform_data(x_limits=(-10, 2), y_limits=(18, 45), n_points=NUM_POINTS):
    _low = [x_limits[0], y_limits[0]]
    _high = [x_limits[1], y_limits[1]]
    points = np.random.uniform(_low, _high, (n_points, 2))
    return points


def q_3_a(is_plot=True, repeat=REPEAT):
    points = []
    for i in range(repeat):
        points = create_uniform_data()
        if is_plot:
            plot_2d_points(points, "Uniform Distribution Data x ∈ [−10, 2], y ∈ [18, 45]")
    return points


def q_3_b(is_plot=True, repeat=REPEAT):
    i_list = [1, 2, 4]
    points = []
    for k in range(repeat):
        points = []
        for i in i_list:
            points.append(create_gaussian([i, -1], [0.5*i, 0.5*i]))
            new_points = np.zeros((NUM_POINTS, 3))
            new_points[:, 0] = points[-1][:, 0]
            new_points[:, 1] = points[-1][:, 1]
            new_points[:, 2] = i
            points[-1] = new_points
        points = np.concatenate(points)
        if is_plot:
            plot_2d_points(points, "Gaussians at 1, 2, 4", True)
            return points
    return points[:, :2]


def q_3_c(is_plot=True, repeat=REPEAT):
    points = []
    for k in range(repeat):
        t_limits = [[[0.5, 3.5], [1, 2]], [[1.5, 2.5], [-1, 1]]]
        points = []
        n_points = NUM_POINTS // 2
        for lim in t_limits:
            points.append(create_uniform_data(lim[0], lim[1], ceil(n_points / len(t_limits))))

        s_centers = [[-1, 1.5], [-2, 1.8], [-3, 1.5], [-2.5, 1], [-2, 0.5], [-1.5, 0.0], [-1, -0.5],
                     [-2, -0.8], [-3, -0.5]]
        s_std = [[0.25, 0.2], [0.35, 0.2], [0.25, 0.3], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.25, 0.3],
                 [0.35, 0.2], [0.25, 0.2]]
        for i in range(len(s_centers)):
            points.append(create_gaussian(s_centers[i], s_std[i], ceil(n_points / len(s_centers))))
        points = np.concatenate(points)
        np.random.shuffle(points)
        points = points[:NUM_POINTS]
        if is_plot:
            plot_2d_points(points, "Data over the letters S and T")
    return points


def q_3_d(is_plot=True, repeat=REPEAT):
    points = []
    for k in range(repeat):
        centers = [[-1, 0.5], [-1, -0.5], [1, 0.5], [1, -0.5]]
        std = [0.35, 0.1]
        points = []
        for center in centers:
            points.append(create_gaussian(center, std, ceil(NUM_POINTS / len(centers))))
        points = np.concatenate(points)
        np.random.shuffle(points)
        points = points[:NUM_POINTS]
        if is_plot:
            plot_2d_points(points, "Four Horizontal Clamps", is_color=False, axis_lim=True)
    return points


def q_3_e(is_plot=True, repeat=REPEAT):
    points = []
    for k in range(repeat):
        centers = []
        points = []
        for i in range(101):
            x1 = -1 + (i / 50)
            y1 = 0 - x1**2 + 1
            x2 = i / 50
            y2 = x1**2 - 0.3
            centers.append([x1, y1])
            centers.append([x2, y2])
        for center in centers:
            points.append(create_gaussian(center, [0.025, 0.025], ceil(NUM_POINTS / len(centers))))
        points = np.concatenate(points)
        np.random.shuffle(points)
        points = points[:NUM_POINTS]
        if is_plot:
            plot_2d_points(points, "Two Moons", is_color=False, axis_lim=True, lim=(-1.1, 2.1))
    return points


def q_3_f(is_plot=True, repeat=REPEAT):
    points = []
    for k in range(repeat):
        points = [q_3_e(False)[:488]]
        points.append(create_gaussian([0, 0.85], [0, 0.075], 6))
        points.append(create_gaussian([1, -0.15], [0, 0.075], 6))
        points = np.concatenate(points)
        if is_plot:
            plot_2d_points(points, "Two Moons Sparsely Connected", is_color=False, axis_lim=True,
                           lim=(-1.1, 2.1))
    return points


def k_means_on_dataset(data, dataset_name):
    centers = [2, 3, 4, 5]
    for k in centers:
        for i in range(REPEAT):
            kmeans = KMeans(n_clusters=k, init='random')
            kmeans = kmeans.fit(data)
            merged = np.zeros((len(data), 3))
            merged[:, :2] = data
            merged[:, 2] = kmeans.labels_
            title = "K-means for dataset " + dataset_name + " with k = " + str(k)
            plot_2d_points(merged, title, is_color=True)


def q_4_a():
    datasets = {'a': q_3_a, 'b': q_3_b, 'c': q_3_c, 'd': q_3_d, 'e': q_3_e, 'f': q_3_f}
    for name, func in datasets.items():
        data = func(False, 1)
        k_means_on_dataset(data, name)


def main():
    q_4_a()


if __name__ == "__main__":
    main()
