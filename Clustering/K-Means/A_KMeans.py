import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import csv
from random import uniform
from math import sqrt, inf


class KMeans:
  def __init__(self, X, K):
    self.X = X
    self.K = K 
    self.centers = self.random_centers()
  
  def random_centers(self):
    N = len(self.X[0])
    x_min = [min(x[i] for x in self.X) for i in range(N)]
    x_max = [max(x[i] for x in self.X) for i in range(N)]
    centers = np.array([[uniform(x_min[i], x_max[i]) for i in range(N)] for _ in range(self.K)])
    return centers

  def inertia(self):
    return sum(min(np.linalg.norm(x - u)**2 for u in self.centers) for x in self.X)

  def d(self, x, y, p=2):
    return sqrt(np.dot(x-y, x-y))

  def train(self, steps = 100):
    S_i = self.inertia
    S = 0
    while S_i != S and steps > 0:
      steps -= 1
      # Actualizamos la inercia anterior.
      S = S_i

      # Agregamos cada elemento a su cluster correspondiente.
      self.clusters = [[] for _ in range(self.K)]
      for x in X:
        min_d = inf
        center = None
        for i, c in enumerate(self.centers):
          dist = self.d(x, c)
          if dist < min_d:
            min_d = dist
            center = i
        self.clusters[center].append(x)

      # Los nuevos centros seran los centros son promediados.
      for i in range(len(self.clusters)):
        self.centers[i] = sum(x for x in self.clusters[i])/max(1, len(self.clusters[i]))

      # Se calcula la inercia actual.
      S_i = self.inertia()
 

# Leemos los datos y los graficamos.
with open('../../../MaterialMML/DataSets/Mall_Customers.csv', newline='') as File:  
    reader = csv.reader(File)
    X, Y = [], []
    prim = True
    for row in reader:
      if not prim:
        X.append(np.array([int(r) for r in row[2:]]))
      else: prim = False

# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')
# Agregamos los puntos en el plano 3D
ax1.scatter([x[0] for x in X], [x[1] for x in X], [x[2] for x in X], c='b', marker='o')

# Mostramos el gráfico
plt.show()

# Aplicamos K-Means
KM = KMeans(X, 4)
KM.train()
