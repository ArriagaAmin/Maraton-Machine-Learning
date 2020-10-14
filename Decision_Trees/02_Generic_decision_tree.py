# Implementacion generica de Arboles de Decisiones.
from math import log, inf

class Node:
  def __init__(self, parent, X, Y, atr_types, default):
    self.parent = parent

    # Ejemplos del entrenamiento que pertenecen a este nodo.
    self.X = X
    # Etiquetas de los ejemplos.
    self.Y = Y

    # Tipos de atributos de los ejemplos.
    self.atr_types = atr_types
    # Moda de las etiquetas.
    self.default = default

    self.childs = []
    # La i-esima condicion corresponde al i-esimo hijo.
    self.cond = []
    self.leaf = True
    # Etiqueta que recibe el patron al alcanzar esta nodo en caso de ser hoja.
    self.value = None

class DecisionTree:
  def __init__(self, X, Y, atr_types, atr_name):
    # Ejemplos de entrenamiento.
    self.X = X
    # Etiquetas de los ejemplos.
    self.Y = Y 
    # Tipos de atributos de los ejemplos de entrenamiento. Hay dos tipos:
    # "Catg" -> Categorico
    # "Cont" -> Continuo
    self.atr_types = atr_types
    # Nombres de los atributos de los ejemplos de entrenamiento.
    self.atr_name = atr_name

  def gini(self, *P):
    """ Gini impurity."""
    return 1 - sum(p**2 for p in P)

  def entropy(self, *P):
    """ Entropy for measure of randomness."""
    r = 0
    for p in P:
      if p==1: return 0
      elif p>0: r -= p*log(p,2)
    return r 

  def mayoria(self, Y):
    # Retorna la moda de un arreglo de elementos unitarios, ejemplo:
    # [[0], [1], [0]] -> mayoria = 0
    dic = {}
    for y in Y:
      if y[0] in dic: dic[y[0]] += 1
      else: dic[y[0]] = 1

    best = None
    max_c = 0
    for d in dic:
      if dic[d] > max_c:
        max_c = dic[d]
        best = d 
    return d 

  def get_values(self, X, a):
    """ Obtenemos los posibles valores de un determinado atributo."""
    n = len(X[0])
    values = []
    for x in X:
      if not x[a] in values: values.append(x[a])
    return values

  def gain_catg(self, a, values, X, Y):
    """ Calculamos la ganancia de un atributo categorico en especifico. """
    # Calculamos la probabilidad de aparicion de cada etiqueta.
    N = len(Y)
    dic = {}
    for y in Y:
      if y[0] in dic: dic[y[0]] += 1/N
      else: dic[y[0]] = 1/N
    # Calculamos la entropia del nodo actual.
    r = self.crit(*[dic[d] for d in dic])

    # Calculamos la entropia de cada nodo luego de la division
    # y se lo restamos a la entropia del nodo actual.
    # Por cada valor del atributo indicado.
    for v in values:

      # Calculamos la probabilidad de aparicion de cada etiqueta dado
      # que el atributo indicado tiene el valor v.
      dic = {}
      N_i = 0
      for i, y in enumerate(Y):
        if y[0] in dic and X[i][a]==v: 
          dic[y[0]] += 1
          N_i += 1
        elif X[i][a]==v: 
          dic[y[0]] = 1
          N_i += 1

      # Calculamos la entropia de una de las divisiones.
      r -= N_i*self.crit(*[dic[d]/N_i for d in dic])/N
    return r

  def gain_cont(self, a, values, X, Y):
    """ Calculamos la ganancia de un atributo continuo en especifico. """
    # Calculamos la probabilidad de aparicion de cada etiqueta.
    N = len(Y)
    dic = {}
    for y in Y:
      if y[0] in dic: dic[y[0]] += 1/N
      else: dic[y[0]] = 1/N
    # Calculamos la entropia del nodo actual.
    r = self.crit(*[dic[d] for d in dic])

    # Obtenemos las posibles divisiones binarias
    values.sort()
    divs = [(values[i]+values[i+1])/2 for i in range(len(values)-1)]

    # Elegimos la division con la entropia minima
    min_e = inf
    best_d = -1
    for d in divs:
      # Calculamos la probabilidad de aparicion de cada etiqueta dado
      # que el atributo es mayor o igual a la division.
      dic = {}
      N_i = 0
      for i, y in enumerate(Y):
        if y[0] in dic and X[i][a]>=d: 
          dic[y[0]] += 1
          N_i += 1
        elif X[i][a]>=d: 
          dic[y[0]] = 1
          N_i += 1
      # Calculamos la entropia de una de las divisiones.
      e = N_i*self.crit(*[dic[d]/N_i for d in dic])/N

      # Calculamos la probabilidad de aparicion de cada etiqueta dado
      # que el atributo es menor a la division.
      dic = {}
      N_i = 0
      for i, y in enumerate(Y):
        if y[0] in dic and X[i][a]<d: 
          dic[y[0]] += 1
          N_i += 1
        elif X[i][a]<d: 
          dic[y[0]] = 1
          N_i += 1
      # Calculamos la entropia de una de las divisiones.
      e += N_i*self.crit(*[dic[d]/N_i for d in dic])/N

      if e < min_e:
        min_e = e 
        best_d = d
    
    # Retornamos la entropia actual menos la de las divisiones
    return r - min_e, best_d

  def train(self, splits = -1, criterio="Entropy"):
    if criterio == "Entropy": self.crit = self.entropy
    elif criterio == "Gini": self.crit = self.gini

    root = Node(None, self.X, self.Y, self.atr_types, self.mayoria(self.Y))
    queue = [root]
    self.tree = root

    # Usaremos un BFS en vez de DFS.
    while len(queue) > 0:
      node = queue.pop(0)

      # Si no hay mas ejemplos, tomamos el default del padre.
      if len(node.X) == 0: node.value = node.parent.default
      # Si todos los ejemplos tienen la misma etiqueta, entonces sesa etiqueta
      # sera el valor del nodo.
      elif all(node.Y[0] == y for y in node.Y): node.value = node.Y[0][0]
      # Si los ejemplos no tienen mas atributos, tomamos la moda de las etiquetas.
      elif len(node.X[0]) == 0 or splits == 0: node.value = self.mayoria(node.Y)
      # Si no, se realizara una division.
      else:
        node.leaf = False
        splits -= 1

        # Obtenemos el mejor atributo calculando la ganancia de informacion
        # de cada uno de ellos.
        best = -1
        best_g = -1
        div = -1
        for a in range(len(node.X[0])):
          values = self.get_values(node.X, a)
          if node.atr_types[a] == "Catg": 
            g = self.gain_catg(a, values, node.X, node.Y)
            if g > best_g:
              best_g = g 
              best = a
          else: 
            g, div = self.gain_cont(a, values, node.X, node.Y)
            if g > best_g:
              best_g = g 
              best = a
              best_d = div
      
        # Verificamos si el mejor atributo es categorico o continuo.
        if node.atr_types[best] == "Catg":
          # Particionamos los ejemplos segun cada valor del mejor atributo.
          for v in self.get_values(node.X, best):
            X_i, Y_i = [], []
            for i in range(len(node.X)):
              if node.X[i][best] == v:
                x = node.X[i].copy()
                x.pop(best)
                X_i.append(x)
                Y_i.append(node.Y[i]) 

            # Creamos un nuevo nodo hijo con un bloque de la particion de los ejemplos.
            atr_types_i = node.atr_types.copy()
            atr_types_i.pop(best)
            child = Node(node, X_i, Y_i, atr_types_i, self.mayoria(Y_i))
            node.childs.append(child)
            node.cond.append((best, v))
            queue.append(child)
        else:
          # Particionamos los ejemplos en menor y mayor o igual que la divison obtenida.
          X_M, X_m, Y_M, Y_m = [], [], [], []
          for i in range(len(node.X)):
            x = node.X[i].copy()
            if node.X[i][best] < best_d:
              X_m.append(x)
              Y_m.append(node.Y[i]) 
            else:
              X_M.append(x)
              Y_M.append(node.Y[i]) 

          # Con esa particion creamos dos nuevos nodos.
          atr_types_i = node.atr_types.copy()
          child_m = Node(node, X_m, Y_m, atr_types_i, self.mayoria(Y_m))
          child_M = Node(node, X_M, Y_M, atr_types_i, self.mayoria(Y_M))
          node.childs.append(child_m)
          node.childs.append(child_M)
          node.cond.append((best, "<", best_d))
          node.cond.append((best, ">=", best_d))
          queue.append(child_m)
          queue.append(child_M)

  def predict(self, x):
    """ Predecimos la etiqueta de un patron recorriendo el arbol."""

    # Partimos de la raiz.
    node_i = self.tree
    x_i = x.copy()
    while True:
      # Si el nodo actual es una hoja, retornamos su valor.
      if node_i.leaf: return node_i.value
      # En caso contrario, verificamos cual condicion del nodo cumple el patron
      # y lo enviamos al hijo correspondiente.
      for i, c in enumerate(node_i.cond):
        if len(c) == 2:
          if x_i[c[0]] == c[1]:
            node_i = node_i.childs[i]
            x_i.pop(c[0])
            break
        elif (c[1] == "<" and x_i[c[0]] < c[2]) or \
          (c[1] == ">=" and x_i[c[0]] >= c[2]):
          node_i = node_i.childs[i]
          break

  def print_tree(self, node_i = None, level = 0, atr = None):
    """ Retornamos una representacion del arbol. """
    if node_i == None: node_i = self.tree
    if atr == None: atr_i = self.atr_name.copy()
    else: atr_i = atr.copy()
    
    if node_i.leaf: 
      text = " -> " + str(node_i.value)
    else:
      best = node_i.cond[0][0]
      if len(node_i.cond[0]) == 2: text = "\n" + level*"|  " + "_ " + atr_i.pop(best)
      else: text = "\n" + level*"|  " + "_ " + atr_i[best]
      for i, c in enumerate(node_i.cond):
        text += "\n" + (level+1)*"|  " + " * " + str(c[1])
        if len(c) == 3: text += str(c[2])
        text += self.print_tree(node_i.childs[i], level+1, atr_i)
      text += "\n" + (level)*"|  " + "-"
    return text


if __name__ == "__main__":
  X = [
    [5,"Esp"], [9,"Eur"], [0,"Eur"], [3,"Esp"], [8,"Eur"], 
    [24,"Esp"], [45,"Esp"], [11,"Eur"], [30,"Eur"], [25,"Eur"], 
    [58,"Esp"], [60,"Esp"], [65,"Eur"], [78,"Esp"], [52,"Eur"],
    [40,"Esp"], [33,"Eur"]
  ]
  Y = [[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[2],[2],[1],[2],[1],[1],[0]]

  IA = DecisionTree(X, Y, 
    ["Cont", "Catg"],
    ["UNIDADES", "DESTINO"],
    None
    )

  IA.train_C(4, "Gini")
  print(IA.print_tree())


            



