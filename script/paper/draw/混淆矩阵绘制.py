from model.analyzer.Drawer import Drawer
import numpy as np

confusion_matrix = np.array([[99.87, 0.13, 0, 0],
                             [0, 88.24, 11.76, 0, ],
                             [0, 7.23, 91.58, 1.19],
                             [0, 0, 0.74, 99.26]
                             ])
Drawer.drawConfusionMatrix(confusion_matrix)
