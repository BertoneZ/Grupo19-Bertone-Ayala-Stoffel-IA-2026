from simpleai.search import SearchProblem, astar, breadth_first, depth_first
from simpleai.search.viewers import WebViewer, BaseViewer

class RoverProblem(SearchProblem):
    def __init__(self, estado_inicial, bateria_max, zonas_sombra, muestras_igneas, muestras_sedimentarias):
        # Aquí guardamos los datos fijos del mapa que no cambian
        self.bateria_max = bateria_max
        self.zonas_sombra = zonas_sombra
        self.muestras_igneas = muestras_igneas
        self.muestras_sedimentarias = muestras_sedimentarias
        # El estado inicial lo pasamos al padre (SearchProblem)
        super().__init__(initial_state=estado_inicial)

    def actions(self, state):
        # Aquí devolverás la lista de acciones posibles desde el estado actual
        # Ejemplo: [("moverse", (1, 2)), ("recargar", None)]
        return []

    def result(self, state, action):
        # Aquí calculas cómo queda el mundo después de aplicar la acción
        # Recordá: el estado debe ser "inmutable" (usá tuplas, no listas)
        return state

    def is_goal(self, state):
        # Condición de victoria: ¿Muestras recolectadas y bodega vacía?
        return False

    def cost(self, state, action, state2):
        # El costo es el TIEMPO que tarda la acción
        return 1

    def heuristic(self, state):
        # Una estimación de cuánto falta para ganar (debe ser admisible)
        return 0

# 3. FUNCIÓN PRINCIPAL (La que llaman los Tests)
def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
    """
    Esta función debe devolver la lista de tuplas con las acciones.
    """
    # A. Definimos el estado inicial como una tupla (posicion, bateria, taladro, bodega, muestras_pendientes)
    # Ejemplo: ((0, 0), 20, None, (), (muestras_igneas, muestras_sedimentarias))
    estado_inicial = (rover_inicio, bateria_inicial, None, (), tuple(muestras_igneas), tuple(muestras_sedimentarias))

    # B. Creamos la instancia del problema
    metas_igneas = tuple(muestras_igneas)
    metas_sedimentarias = tuple(muestras_sedimentarias)
    
    problema = RoverProblem(
        estado_inicial, 
        20, # bateria max
        zonas_sombra, 
        metas_igneas, 
        metas_sedimentarias
    )

    # C. Ejecutamos el algoritmo de búsqueda (A* es el mejor para minimizar tiempo)
    # Si querés ver el grafo en el navegador, podés usar WebViewer()
    resultado = astar(problema, graph_search=True)

    # D. Transformamos el resultado en la lista de acciones que pide el enunciado
    lista_acciones = []
    if resultado:
        for accion, nuevo_estado in resultado.path():
            if accion: # La primera acción es None, la salteamos
                lista_acciones.append(accion)

    return lista_acciones

# BLOQUE DE PRUEBA MANUAL
if __name__ == "__main__":
    # Aquí podés probar tu código antes de correr los tests oficiales
    acciones = planear_rover(
        rover_inicio=(0, 0),
        bateria_inicial=20,
        zonas_sombra=[(0, 1)],
        muestras_igneas=[(1, 1)],
        muestras_sedimentarias=[(2, 2)]
    )
    print(acciones)