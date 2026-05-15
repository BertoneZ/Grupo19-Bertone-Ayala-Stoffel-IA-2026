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
        pos, bateria, taladro, bodega, igneas, sedim = state
        accs = []

        # movimientos adyacentes
        r, c = pos
        vecinos = [ (r+1, c), (r-1, c), (r, c+1), (r, c-1) ]
        for np in vecinos:
            # moverse consume 1 de batería y nunca puede dejar a 0
            if bateria - 1 > 0:
                accs.append(("moverse", np))

        # sobremarcha: 2 celdas en linea recta
        overdrive = [ (r+2, c), (r-2, c), (r, c+2), (r, c-2) ]
        for np in overdrive:
            if bateria - 4 > 0:
                accs.append(("sobremarcha", np))

        # equipar taladro (permitir equipar a termico o percusion)
        for t in ("termico", "percusion"):
            if bateria - 1 > 0 and taladro != t:
                accs.append(("equipar", t))

        # recolectar: si hay muestra en la posicion y taladro correcto y espacio en bodega
        if pos in igneas and taladro == "termico" and len(bodega) < 2 and bateria - 3 > 0:
            accs.append(("recolectar", "ignea"))
        if pos in sedim and taladro == "percusion" and len(bodega) < 2 and bateria - 3 > 0:
            accs.append(("recolectar", "sedimentaria"))

        # depositar: si hay muestras en bodega y se cumplen las condiciones
        remaining_samples = tuple(igneas) + tuple(sedim)
        if len(bodega) > 0:
            if len(bodega) == 2 or (len(bodega) == 1 and not remaining_samples):
                # allowed, pero la batería debe quedar >0 despues de la acción
                if bateria - 1 > 0:
                    accs.append(("depositar", None))

        # recargar: solo si no estamos en zona_sombra, no está llena y bateria <= 10
        if pos not in self.zonas_sombra and bateria < self.bateria_max and bateria <= 10:
            accs.append(("recargar", None))

        return accs

    def result(self, state, action):
        # Aquí calculas cómo queda el mundo después de aplicar la acción, el estado debe ser "inmutable" 
        pos, bateria, taladro, bodega, igneas, sedim = state
        nombre_accion, valor = action

        if nombre_accion == "moverse":
            return (valor, bateria - 1, taladro, bodega, igneas, sedim)

        if nombre_accion == "sobremarcha":
            return (valor, bateria - 4, taladro, bodega, igneas, sedim)

        if nombre_accion == "equipar":
            return (pos, bateria - 1, valor, bodega, igneas, sedim)

        if nombre_accion == "recargar":
            nueva_bateria = min(self.bateria_max, bateria + 10)
            return (pos, nueva_bateria, taladro, bodega, igneas, sedim)

        if nombre_accion == "depositar":
            # Vaciamos la bodega
            return (pos, bateria - 1, taladro, (), igneas, sedim)

        if nombre_accion == "recolectar":
            # valor indica el tipo: 'ignea' o 'sedimentaria'
            if valor == "ignea":
                nuevas_igneas = tuple(p for p in igneas if p != pos)
                nueva_bodega = bodega + ("ignea",)
                return (pos, bateria - 3, taladro, nueva_bodega, nuevas_igneas, sedim)
            else:
                nuevas_sedim = tuple(p for p in sedim if p != pos)
                nueva_bodega = bodega + ("sedimentaria",)
                return (pos, bateria - 3, taladro, nueva_bodega, igneas, nuevas_sedim)

        return state
    def is_goal(self, state):
        # Condición de victoria: ¿Muestras recolectadas y bodega vacía?
        pos, bateria, taladro, bodega, igneas, sedim = state
        return len(igneas) == 0 and len(sedim) == 0 and len(bodega) == 0

    def cost(self, state, action, state2):
        # El costo es el TIEMPO que tarda la acción
        nombre_accion, valor = action
        if nombre_accion == "moverse":
            return 1
        if nombre_accion == "sobremarcha":
            return 1
        if nombre_accion == "equipar":
            return 3
        if nombre_accion == "recolectar":
            return 2
        if nombre_accion == "recargar":
            return 4
        if nombre_accion == "depositar":
            # el tiempo es 1 minuto por muestra que hay en la bodega en el estado previo
            _, _, _, bodega, _, _ = state
            return len(bodega) * 1
        return 0

    def heuristic(self, state):
        pos, bateria, taladro, bodega, igneas, sedim = state
        pendientes = igneas + sedim
        cargadas = len(bodega)

        # Meta exacta
        if not pendientes and cargadas == 0:
            return 0

        # 1) Movimiento: cota inferior por muestra más lejana.
        # Cada acción de movimiento reduce Manhattan a lo sumo en 2 (sobremarcha).
        movimiento_lb = 0
        if pendientes:
            max_dist = max(abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in pendientes)
            movimiento_lb = (max_dist + 1) // 2

        # 2) Recolectar: 2 minutos por muestra pendiente (obligatorio).
        recolectar_lb = 2 * len(pendientes)

        # 3) Depositar: al final se deposita toda muestra (cargada o pendiente).
        depositar_lb = cargadas + len(pendientes)

        # 4) Equipar: cota inferior según tipos pendientes y taladro actual.
        tipos_pendientes = set()
        if igneas:
            tipos_pendientes.add("termico")
        if sedim:
            tipos_pendientes.add("percusion")

        equipar_lb = 0
        if tipos_pendientes:
            if taladro is None:
                # Si hay ambos tipos, al menos dos equipamientos; si hay uno, al menos uno.
                equipar_lb = 3 * len(tipos_pendientes)
            elif taladro not in tipos_pendientes:
                equipar_lb = 3
            elif len(tipos_pendientes) == 2:
                # Ya tengo uno de los dos: al menos un cambio futuro.
                equipar_lb = 3

        return movimiento_lb + recolectar_lb + depositar_lb + equipar_lb

# FUNCIÓN PRINCIPAL (La que llaman los Tests)
def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
    """
    Esta función debe devolver la lista de tuplas con las acciones.
    """
    # Definimos el estado inicial como una tupla (posicion, bateria, taladro, bodega, muestras_pendientes)
    # Ejemplo: ((0, 0), 20, None, (), (muestras_igneas, muestras_sedimentarias))
    estado_inicial = (rover_inicio, bateria_inicial, None, (), tuple(muestras_igneas), tuple(muestras_sedimentarias))

    # Creamos la instancia del problema
    metas_igneas = tuple(muestras_igneas)
    metas_sedimentarias = tuple(muestras_sedimentarias)
    
    problema = RoverProblem(
        estado_inicial, 
        20, # bateria max
        zonas_sombra, 
        metas_igneas, 
        metas_sedimentarias
    )

    # Ejecutamos el algoritmo de búsqueda (A* es el mejor para minimizar tiempo)
    resultado = astar(problema, graph_search=True)

    # Transformamos el resultado en la lista de acciones que pide el enunciado
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
