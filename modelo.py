#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:24:05 2018

@author: eduardo
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import distributions as dists
import math


class Modelo:
    def __init__(self, nrealizations = 500, prob = 0.1, \
                 Graph = nx.grid_2d_graph, paramGraph = {'m':5, 'n':5, 'periodic':False}):
        """ Crea un Modelo para simular propogación de virus en un grafo según \
            el modelo de Anderson et al. 
        
        Args:
            nrealization (int, optional): Number of simulations
            prob (float): Porcentaje de nodos en estado 'susceptible' \
                          cuando se inicializa el grafo
            Graph (func): Graph constructor nx.Graph. 
            paramGraph (dictionary, optional): Optional parameters for nx.Graph 
        """
        # Simulation time each realization       
        self.time = 0        
    
        #: Number of simulations (one is not enough...)
        self.nrealizations = nrealizations    
         
        # Do the Graph
        self.G = Graph (**paramGraph)
        
        #  Solo válido para grid_2d_graph 
        self.__dim_x = paramGraph['m']
        self.__dim_y = paramGraph['n']
        self._periodic = paramGraph['periodic']
      
        # Init nodes graph (Anderson et al. paper)
        nodes = list(self.G.nodes)
        self.prob_susceptible = prob
        
        self.G.add_nodes_from(nodes, estado = 'free')      
        self.set_nodes_attr_prob (prob = self.prob_susceptible, \
                                  fro = {'estado':'free'}, \
                                  to = {'estado':'susceptible'} )
        
    def node_es_contagiado (self, node, prob = 0.5, atr = {'estado' : 'infective'}):
        """ modifica el atributo de un nodo en funcion de los atributos de los nodos de su lista de adj
            Args:
                node (node_name): eg: (0,0)
                prob (float): prob to change its node_attributte due its adjc list
            Ret: Final probablity used to modify the node attribute
            Raises:    
        """
        # probabilidad del nodo de no modoficar el valor de su atributo
        sano = 1
        
        key = list(atr.keys())[0]   # llave del atributo
        
        # actualiza la probabilidad en función del estado de los nodes en la lista de adj
        for i in list(self.G.adj[node]):
            if self.G.nodes[i][key] == atr[key] :
                sano = sano * (1 - prob) 
                
        # modifica aleatoraimente el valor del atributo        
        if np.random.uniform() > sano:
            self.G.nodes[node][key] = atr[key] 
            
        return sano
    
    def node_contagia (self, node, prob = 0.5, \
                       atr_from = {'estado' : 'susceptible'},
                       atr_to = {'estado' : 'infected'}):
        """ modifica el atributo de los nodos de su lista de adj si estos tienen \
            un determinado atributo
            Args:
                node (node_name): eg: (0,0)
                prob (float): probability to change node_attributte for nodes in its adjc list
            Ret: Final probablity used to modify the node attribute
            Raises:    
        """
        key_from = list(atr_from.keys())[0]
        key_to   = list(atr_to.keys())[0]   

        # actualiza la probabilidad en función del estado de los nodes de su lista de adj
        for i in list(self.G.adj[node]):
            if self.G.nodes[i][key_from] == atr_from[key_from]:
                if np.random.uniform() < prob:
                    self.G.nodes[i][key_to] = atr_to[key_to] 
            
    
    
    def do_list_nodes_attr(self, atr = {'estado' : 'infective'}) :
        """ Devuelve una lista con los nodos del garfo que tienen un valor
            determinado del atributo
            Args:
                atr (dictionary): Atributo : valor
            Ret:
                Lista
        """
        key = list(atr.keys())[0]
        
        fil = [x for x in self.G.nodes if self.G.nodes[x][key] == atr[key] ]
        return fil 
    

    def set_nodes_attr_prob(self, prob = 0.7, fro = {'estado' : 'infective'},\
                            to = {'estado' : 'recovered'} ) :
        """ Modifica el atributo de los nodos del grafo con una determinada probabilidad
            Arg:
                pro (float): Probabilidad con la que se modifica el atributo del nodo
                fro (dictionary): Atributo : valor de los nodos antes de modificar
                to (dictionary): Atributo : valor de los nodos despues de modificar
        """
        key = list(to.keys())[0]
        # crea una lista solo con los nodes que tienen el atributo de "from"
        filtered = self.do_list_nodes_attr(atr = fro)
        
        for x in filtered:
             if np.random.uniform() < prob:
                 self.G.nodes[x][key] = to[key] 


    def nodes_grid_jump (self, node):
        """Implementa el "salto" descrito en el paper de Rhodes and Anderson.
           Un nodo fuente puede saltar a un nodo destino o permanecer en su lugar original. \
           Las probabilidades de saltar a cualquier modo o permancer son iguales. 
           Cuando un nodo salta modifica el estado del nodo fuente y el del destino.
            Arga:
                node (tupla): name of the "jumping" sourde node
            Ret: 
                node_source, node_target (tupla)
        """
        i, j = node        #node coordinates
        
        # no adjaccency nodes where jumping is allowed
        sites = [(i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)]
        
        # "jumping" nodes inside the grid limits
        sites = [x for x in sites if x[0] > -1 \
              and x[0] < self.__dim_x and x[1]>-1 and x[1] < self.__dim_y]
        
        # add to the allowed "jumping" sites the adjanced nodes 
        sites = sites +  list(self.G.adj[node])
        
        # only allowed jumping sites are in 'estado' :  'free'
        sites =  [x for x in sites if self.G.nodes[x]['estado'] == 'free']
        
        # add the original node (remainning)
        sites.append(node)
        
        # total number of sites to jump and remain
        nsites = len(sites)
        
        # jumping with equal probabilities
        place = np.asscalar(np.random.choice(nsites, size=1))
        node_site = sites[place]
        
        # change el estado de los node destino y fuente
        if node != node_site:
            self.G.nodes[node_site]['estado'] = self.G.nodes[node]['estado']
            self.G.nodes[node]['estado'] = 'free'
                
        return node, node_site

    def nodes_grid_longjump (self, node):
        """Implementa el "salto" descrito en el paper de Rhodes and Anderson.
           Un nodo fuente puede saltar a un nodo destino o permanecer en su lugar original. \
           Las probabilidades de saltar a cualquier modo o permancer son iguales. 
           Cuando un nodo salta modifica el estado del nodo fuente y el del destino.
            Arga:
                node (tupla): name of the "jumping" sourde node
            Ret: 
                node_source, node_target (tupla)
        """
        i, j = node        #node coordinates
        flag = 0

        while flag == 0:
            x = np.random.randint(self.__dim_x)
            y = np.random.randint(self.__dim_y)

            if self.G.nodes[(x,y)]['estado'] == 'free':
                self.G.nodes[(x,y)]['estado'] = self.G.nodes[node]['estado']
                self.G.nodes[node]['estado'] = 'free'
                flag = 1
                node_site = (x,y)
                
        return node, node_site

##### Ejemplo de main #####################################3
  
    
def simulation (max_time=600, C=0.2, L=50, p=0.5, pr=0.1, ps=0.05, periodic=False, longjump = False, normLL = False):
    # Create a Model Rhodes and Anderson
    a = Modelo(prob=C, Graph = nx.grid_2d_graph, paramGraph = {'m':L, 'n':L, 'periodic':periodic})

    # set seed node to infective
    node_seed = (L // 2, L // 2)
    a.G.nodes[(node_seed)]['estado'] = 'infective'
    
    # set infected list
    infected = [node_seed]
    
    # output results
    if normLL:
        evolution_infected = [len(infected)/(L*L)]
        evolution_susceptible = [(((L*L)*C)-1)/(L*L)]
        evolution_recovered = [0]
    else:
        evolution_infected = [len(infected)/(L*L*C)]
        evolution_susceptible = [(((L*L)*C)-1)/(L*L*C)]
        evolution_recovered = [0]
    
    while a.time < max_time:    
        
        # solo se ejecuta si hay infectados
        
        if len(infected) == 0:
            break
        
        # extiende el contagio entre los susceptibles
        for node in infected:
            a.node_contagia(node, prob=p, \
                            atr_from = {'estado':'susceptible'}, \
                            atr_to = {'estado':'infective'}) 
            
        # infective -> recovered
        a.set_nodes_attr_prob(prob = pr, fro = {'estado' : 'infective'},\
                            to = {'estado' : 'recovered'} )
        
        # recovered -> susceptible
        a.set_nodes_attr_prob(prob = ps, fro = {'estado' : 'recovered'},\
                            to = {'estado' : 'susceptible'} )
        # jumping
        susceptible = a.do_list_nodes_attr(atr = {'estado':'susceptible'})
        infected =    a.do_list_nodes_attr(atr = {'estado':'infective'})
        recovered =   a.do_list_nodes_attr(atr = {'estado':'recovered'})

        all_no_free = recovered + susceptible + infected
        for node in all_no_free:
            if longjump:
                a.nodes_grid_longjump(node)
            else:
                a.nodes_grid_jump(node)

        #output statiscs
        if normLL:
            evolution_infected.append(len(infected)/(L*L))
            evolution_susceptible.append(len(susceptible)/(L*L))
            evolution_recovered.append(len(recovered)/(L*L))
        else:
            evolution_infected.append(len(infected)/(L*L*C))
            evolution_susceptible.append(len(susceptible)/(L*L*C))
            evolution_recovered.append(len(recovered)/(L*L*C))
        # incremet simulation time
        a.time += 1
        
        
    return  evolution_infected, evolution_susceptible, evolution_recovered

def collectData(nombreFichero, confidence = 0.95):

	dicc = {}

	stats = np.load(nombreFichero)
	list_data = ['s', 'i', 'r']
	i=0
	for prefix in list_data:
		dicc = collect(dicc, stats[i], prefix=prefix, confidence = confidence)
		i += 1

	return dicc

def collect(dicc, stats, prefix = 'example', confidence = 0.95):

	l = len(stats)
	time = len(stats[0])
	dicc[prefix + '_means'] = np.mean(stats, axis=0)
	dicc[prefix + '_stds'] = np.std(stats, axis=0)
	dicc[prefix + '_sems'] = dicc[prefix + '_stds']/math.sqrt(l)
	dicc[prefix + '_cvs'] = dicc[prefix + '_stds']/dicc[prefix + '_means']
	dicc[prefix + '_h'] = dicc[prefix + '_sems'] * dists.t.ppf((1 + confidence) / 2, time - 1)
	dicc[prefix + '_start'] = dicc[prefix + '_means'] - dicc[prefix + '_h']
	dicc[prefix + '_end'] = dicc[prefix + '_means'] + dicc[prefix + '_h']

	return dicc

	


	


