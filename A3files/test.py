from matplotlib import pyplot as plt

from A3codes import minMulDev,classify, calculateAcc
from A3helpers import augmentX, gaussKernel, plotModel, generateData, plotPoints, synClsExperiments


train_acc, test_acc = synClsExperiments(minMulDev, classify, calculateAcc)

print("Train acc:", train_acc)
print("Test acc:", test_acc)
