# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math

class RRT():
    def __init__(self, init_x, init_y):
        # 指定移動距離
        self.stretch_distance = 0.075

        # 探索範囲最大
        self.MAX_x = 3
        self.MAX_y = 3

        # 探索範囲最小
        self.MIN_x = -3
        self.MIN_y = -3

        # ノード(点)作成
        self.Nodes = np.array([[init_x, init_y]])

        # path
        self.path_x = np.empty((0, 2), float)
        self.path_y = np.empty((0, 2), float)
        
        # samples
        self.samples = np.empty((0, 2), float)

        self.nearest_node = None
        self.new_node = None

    def search(self):
        # (-self.MAX/2 ~ self.MAX/2)の範囲で選択
        search_x = (np.random.rand() * self.MAX_x) - self.MAX_x/2.
        search_y = (np.random.rand() * self.MAX_y) - self.MAX_y/2.

        sample = np.array([search_x, search_y])
        self.samples = np.append(self.samples, [[search_x, search_y]], axis=0)
        
        # ノード探索
        distance = float("inf")
        self.nearest_node = None

        for i in range(self.Nodes.shape[0]):
            node = self.Nodes[i, :]
            part_MSE = (sample - node)**2
            RMSE = math.sqrt(sum(part_MSE))
            if RMSE < distance: # 比較的に距離が近い場合
                distance = RMSE
                self.nearest_node = node

        # 新ノード作成
        pull = sample - self.nearest_node
        grad = math.atan2(pull[1], pull[0])

        d_x = math.cos(grad) * self.stretch_distance
        d_y = math.sin(grad) * self.stretch_distance
        self.new_node = self.nearest_node + np.array([d_x, d_y])
        #return self.nearest_node, self.new_node

    def path_make(self):
        # 新ノード追加
        self.Nodes = np.vstack((self.Nodes, self.new_node))

        self.path_x = np.append(self.path_x, np.array([[self.nearest_node[0], self.new_node[0]]]), axis=0)
        self.path_y = np.append(self.path_y, np.array([[self.nearest_node[1], self.new_node[1]]]), axis=0)

        return self.Nodes, self.path_x, self.path_y, self.samples

class Figures():
    def __init__(self):
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111)

    def fig_set(self):
        # 初期設定
        MAX_x = 1.5
        MAX_y = 1.5
        MIN_x = -1.5
        MIN_y = -1.5

        # 各軸の最大最小
        self.axis.set_xlim(MIN_x, MAX_x)
        self.axis.set_ylim(MIN_y, MAX_y)

        # 各軸ラベル
        self.axis.set_xlabel("X")
        self.axis.set_ylabel("Y")

        # グリッド表示
        self.axis.grid(True)
        
        # 縦横比
        self.axis.set_aspect("equal")

    def animation_plot(self, path_x, path_y, Nodes, samples):
        imgs = []
        for i in range(path_x.shape[1]):
            path_imgs = []
            img_sample = self.axis.plot(samples[i, 0], samples[i, 1], '*', color='b')
            path_imgs.extend(img_sample)
            img_nodes = self.axis.plot(Nodes[:i+2, 0], Nodes[:i+2, 1], '.', color='k')
            path_imgs.extend(img_nodes)
            imgs.append(path_imgs)

        animation = ani.ArtistAnimation(self.fig, imgs)
        animation.save('rrt_animation.gif', writer='imagemagick')
        plt.show()

def main():
    fig = Figures()
    fig.fig_set()

    path_planner = RRT(0.0, 0.0)
    iterations = 70

    for k in range(iterations):
        path_planner.search()
        Nodes, path_x, path_y, samples = path_planner.path_make()

    path_x = path_x.transpose()
    path_y = path_y.transpose()
    fig.animation_plot(path_x, path_y, Nodes, samples)

if __name__ == "__main__":
    main()
