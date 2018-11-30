from configs import cfg
from utilities import *

a1 = np.array([[10, 90], [20, 80], [30, 70], [40, 60]])
a2 = np.array([[10, 90], [20, 80], [30, 70], [40, 60]])
a3 = np.array([[10, 1], [20, 3], [30, 4], [40, 6]])

save_as_csv(a1, a2, a3, cfg)
