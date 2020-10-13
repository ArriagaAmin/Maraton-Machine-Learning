from math import log

class node:
  def __init__(self, parent):
    self.parent = parent
    self.childs = []
    self.cond = []
    self.leaf = True
    self.value = None

class decision_tree:
  def __init__(self, X, Y, atr_name):
    self.X = X
    self.Y = Y
    self.atr_name = atr_name

  def I(self, *A):
    """ Cantidad de informacion que proporciona un atributo A """
    result = 0
    for a in A:
      if a == 1: return 0
      elif a > 0: result -= a*log(a, 2)
    return result

  def get_values(self, X):
    """ Obtenemos los posibles valores de cada atributo de los ejemplos."""
    n = len(X[0])
    values = [[] for _ in range(n)]
    for x in X:
      for i in range(n):
        if not x[i] in values[i]: values[i].append(x[i])
    return values

  def gain(self, a, values, X, Y):
    """ Calculamos la ganancia de un atributo en especifico. """
    p = sum(1 for y in Y if y == 1)
    n = sum(1 for y in Y if y == 0)
    r = self.I(p/(p+n), n/(p+n))
    for v in values:
      p_i = sum(1 for i in range(len(X)) if Y[i]==1 and X[i][a]==v)
      n_i = sum(1 for i in range(len(X)) if Y[i]==0 and X[i][a]==v)
      r -= (p_i+n_i)*self.I(p_i/(p_i+n_i), n_i/(p_i+n_i))/(p+n)
    return r

  def train(self):
    self.train_T(self.X, self.Y, self.get_values(self.X), 0, None)

  def train_T(self, X, Y, values, por_defecto, parent):
    child = node(parent)
    if parent != None: parent.childs.append(child)
    
    if len(X) == 0: child.value = por_defecto
    elif all(Y[0] == y for y in Y): child.value = Y[0]
    elif len(X[0]) == 0: child.value = int(sum(1 for y in Y if y == 1) > sum(1 for y in Y if y == 0))
    else:
      child.leaf = False

      # Obtenemos el mejor atributo.
      best = -1
      best_g = -1
      for a in range(len(X[0])):
        g = self.gain(a, values[a], X, Y)
        if g > best_g:
          best_g = g
          best = a
      a = best

      m = int(sum(1 for y in Y if y == 1) > sum(1 for y in Y if y == 0))

      for v in values[a]:
        X_i = []
        Y_i = []
        for i in range(len(X)):
          if X[i][a] == v: 
            x = X[i].copy()
            x.pop(a)
            X_i.append(x)
            Y_i.append(Y[i])
        child.cond.append((a, v))

        values_i = self.get_values(X_i)
        self.train_T(X_i, Y_i, values_i, m, child)
    self.tree = child

  def predict(self, x):
    node_i = self.tree
    x_i = x.copy()
    while True:
      if node_i.leaf: return node_i.value
      for i, c in enumerate(node_i.cond):
        if x_i[c[0]] == c[1]:
          node_i = node_i.childs[i]
          x_i.pop(c[0])
          break

  def print_tree(self, node_i = None, level = 0, atr = None):
    if node_i == None: node_i = self.tree
    if atr == None: atr_i = self.atr_name.copy()
    else: atr_i = atr.copy()
    
    if node_i.leaf: 
      text = " -> " + str(bool(node_i.value))
    else:
      best = node_i.cond[0][0]
      text = "\n" + level*"|  " + "_ " + atr_i.pop(best)
      for i, c in enumerate(node_i.cond):
        text += "\n" + (level+1)*"|  " + " * " + str(c[1])
        text += self.print_tree(node_i.childs[i], level+1, atr_i)
      text += "\n" + (level)*"|  " + "-"
    return text


if __name__ == "__main__":
  X = [
    [1,0,0,1,"Algunos","$$$",0,1,"Frances",0],
    [1,0,0,1,"Lleno","$",0,0,"Tailandesa",30],
    [0,1,0,0,"Algunos","$",0,0,"Hamburguesa",0],
    [1,0,1,1,"Lleno","$",1,0,"Tailandesa",10],
    [1,0,1,0,"Lleno","$$$",0,1,"Frances",60],
    [0,1,0,1,"Algunos","$$",1,1,"Italiana",0],
    [0,1,0,0,"Vacio","$",1,0,"Hamburguesa",0],
    [0,0,0,1,"Algunos","$$",1,1,"Tailandesa",0],
    [0,1,1,0,"Lleno","$",1,0,"Hamburguesa",60],
    [1,1,1,1,"Lleno","$$$",0,1,"Italiana",10],
    [0,0,0,0,"Vacio","$",0,0,"Tailandesa",0],
    [1,1,1,1,"Lleno","$",0,0,"Hamburguesa",30]
  ]
  Y = [1,0,1,1,0,1,0,1,0,0,0,1]

  IA = decision_tree(X, Y, ["ALTERNATIVA", "BAR", "VIERNES", "HAMBRE", "CLIENTES", 
    "PRECIO", "LLUVIA", "REST", "TIPO"])
  IA.train()
  print(IA.print_tree())
