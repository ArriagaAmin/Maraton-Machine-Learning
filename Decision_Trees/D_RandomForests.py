# Implementacion generica de Arboles de Decisiones.
from math import log, inf, sqrt
from B_GenericDecisionTree import DecisionTree 
from random import randint

class RandomForest:
  def __init__(self, X, Y, atr_types, atr_names, num_trees):
    self.X = X
    self.Y = Y
    self.atr_types = atr_types
    self.atr_names = atr_names
    # Numero de arboles a usar.
    self.num_trees = num_trees
    # Arboles
    self.trees = []

  def train(self, splits=-1):
    """ Entrenamos los distintos arboles. """

    for i in range(self.num_trees):
      # Escogemos los datos de entrenamiento de forma aleatoria
      N = len(self.X)
      X_i, Y_i = [], []
      for _ in range(N):
        k = randint(0, N-1):
        X_i.append(self.X[k])
        Y_i.append(self.Y[k])

      # Escogemos los atributos a deshabilitar de forma aleatoria
      N = len(self.atr_types)
      M = int(sqrt(N))
      atr = [j for j in range(M)]
      atr_avail = [1]*N
      for _ in range(M):
        k = randint(0, len(atr)-1)
        atr_avail[atr.pop(k)] = 0

      # Escogemos uno de los criterios de forma aleatoria
      criterios = ["Gini", "Entropy"]
      c = criterios[randint(0,1)]

      # Creamos un nuevo arbol
      t = DecisionTree(X_i.copy(), Y_i.copy(), self.atr_types, self.atr_name, atr_avail.copy())
      t.train(splits, c)
      print(str(i) + "-esimo Arbol de Decision entrenado!")
