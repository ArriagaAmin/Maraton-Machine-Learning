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
        k = randint(0, N-1)
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
      t = DecisionTree(X_i.copy(), Y_i.copy(), self.atr_types, self.atr_names, atr_avail.copy())
      t.train(splits, c)
      self.trees.append(t)
      print(str(i) + "-esimo Arbol de Decision entrenado!")

  def predict(self, x, trees = None):
    """ Ponemos a los arboles a votar y la etiqueta con mas votos sera retornada. """
    if trees == None: trees = self.trees

    dic = {}
    for t in trees:
      r = t.predict(x)
      if not r in dic: dic[r] = 1
      else: dic[r] += 1

    max_v = 0
    v = None
    for d in dic:
      if dic[d] > max_v:
        max_v = dic[d]
        v = d 
    return v

  def OOB(self):
    """ Verificamos la calidad del random forest usando el metodo Out Of Bag. """
    acc, N = 0, 0
    for i, x in enumerate(self.X):
      trees = []
      for t in self.trees:
        if not x in t.X: trees.append(t)
      if len(trees) > 0:
        N += 1
        if self.predict(x, trees) == self.Y[i][0]: acc += 1

    if N == 0: return -1
    return acc/N
