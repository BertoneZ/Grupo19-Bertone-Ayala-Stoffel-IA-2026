from simpleai.search import SearchProblem, astar, breadth_first, depth_first
from simpleai.search.viewers import WebViewer, BaseViewer

# entrega1.py
# Implementación de un problema de búsqueda para planear las acciones de un rover.
# - El archivo define la clase `RoverProblem` que adapta el estado del rover a la API
#   de simpleai.search/SearchProblem (actions/result/is_goal/cost/heuristic).
# - La función `planear_rover` crea el problema y ejecuta A* para obtener una lista
#   de acciones que luego devuelve.

class RoverProblem(SearchProblem):
    def __init__(self, estado_inicial, bateria_max, zonas_sombra, muestras_igneas, muestras_sedimentarias):
        # Inicializador del problema de búsqueda.
        # - `estado_inicial` es una tupla que representa la configuración del rover.
        # - `bateria_max` es la capacidad máxima de la batería (p.ej. 20).
        # - `zonas_sombra` son posiciones donde NO se puede recargar.
        # - `muestras_igneas` / `muestras_sedimentarias` son tuplas con posiciones objetivo.
        self.bateria_max = bateria_max
        self.zonas_sombra = zonas_sombra
        self.muestras_igneas = muestras_igneas
        self.muestras_sedimentarias = muestras_sedimentarias
        # Pasamos el estado inicial al padre para que gestione la búsqueda
        super().__init__(initial_state=estado_inicial)

    def actions(self, state):
        # Devuelve una lista de acciones aplicables en `state`.
        # El estado tiene la forma: (posicion, bateria, taladro, bodega, igneas, sedim)
        pos, bateria, taladro, bodega, igneas, sedim = state
        accs = []

        # 1) Movimientos básicos (una casilla ortogonal): consumen 1 de batería.
        r, c = pos
        vecinos = [ (r+1, c), (r-1, c), (r, c+1), (r, c-1) ]
        for np in vecinos:
            # Solo permitimos moverse si la acción deja batería > 0
            if bateria - 1 > 0:
                accs.append(("moverse", np))

        # 2) Sobremarcha (dos celdas en línea recta): consume 4 de batería.
        overdrive = [ (r+2, c), (r-2, c), (r, c+2), (r, c-2) ]
        for np in overdrive:
            if bateria - 4 > 0:
                accs.append(("sobremarcha", np))

        # 3) Equipar taladro: cambiar a `termico` o `percusion` (1 de batería).
        # Solo añadimos la acción si tiene sentido (hay muestras del tipo correspondiente).
        for t in ("termico", "percusion"):
            if bateria - 1 > 0 and taladro != t:
                if t == "termico" and not igneas:
                    continue
                if t == "percusion" and not sedim:
                    continue
                accs.append(("equipar", t))

        # 4) Recolectar muestra si estamos sobre una posición objetivo, tenemos el taladro
        # correcto, espacio en bodega y batería suficiente (3).
        if pos in igneas and taladro == "termico" and len(bodega) < 2 and bateria - 3 > 0:
            accs.append(("recolectar", "ignea"))
        if pos in sedim and taladro == "percusion" and len(bodega) < 2 and bateria - 3 > 0:
            accs.append(("recolectar", "sedimentaria"))

        # 5) Depositar muestras en el punto base: solo permitido cuando la bodega
        # tiene 2 muestras o quedan 0 muestras pendientes (se permite depositar 1).
        remaining_samples = tuple(igneas) + tuple(sedim)
        if len(bodega) > 0:
            if len(bodega) == 2 or (len(bodega) == 1 and not remaining_samples):
                if bateria - 1 > 0:
                    accs.append(("depositar", None))

        # 6) Recargar: solo si NO estamos en zona de sombra y la batería está por debajo
        # de un umbral razonable. El umbral se ajusta ligeramente según la presencia de sombras.
        umbral_recarga = 10 if len(self.zonas_sombra) > 10 else 8
        if pos not in self.zonas_sombra and bateria < self.bateria_max and bateria <= umbral_recarga:
            accs.append(("recargar", None))

        return accs

    def result(self, state, action):
        # Aplica una acción sobre `state` y devuelve el nuevo estado inmutable.
        pos, bateria, taladro, bodega, igneas, sedim = state
        nombre_accion, valor = action

        if nombre_accion == "moverse":
            # moverse a la celda objetivo, consumo 1
            return (valor, bateria - 1, taladro, bodega, igneas, sedim)

        if nombre_accion == "sobremarcha":
            # sobremarcha consume 4 de batería
            return (valor, bateria - 4, taladro, bodega, igneas, sedim)

        if nombre_accion == "equipar":
            # cambiar el taladro equipado
            return (pos, bateria - 1, valor, bodega, igneas, sedim)

        if nombre_accion == "recargar":
            # recarga parcial: suma 10 hasta `bateria_max`
            nueva_bateria = min(self.bateria_max, bateria + 10)
            return (pos, nueva_bateria, taladro, bodega, igneas, sedim)

        if nombre_accion == "depositar":
            # vaciamos la bodega (se pierde 1 de batería por la acción)
            return (pos, bateria - 1, taladro, (), igneas, sedim)

        if nombre_accion == "recolectar":
            # recolectar consume 3 y elimina la muestra del conjunto correspondiente
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
        # Condición de objetivo: no quedan muestras pendientes y la bodega está vacía
        pos, bateria, taladro, bodega, igneas, sedim = state
        return len(igneas) == 0 and len(sedim) == 0 and len(bodega) == 0

    def cost(self, state, action, state2):
        # El coste considera el tiempo (minutos) que dura cada acción.
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
            # depositar toma 1 minuto por muestra que había en la bodega
            _, _, _, bodega, _, _ = state
            return len(bodega) * 1
        return 0

    def heuristic(self, state):
        # Heurística admisible: estima un coste mínimo restante hasta el objetivo.
        # No debe sobreestimar el coste real; está compuesta por:
        #  - una cota de movimiento (dividiendo por 2 por la posibilidad de sobremarcha),
        #  - el coste de recolectar todas las muestras pendientes,
        #  - el coste de depositarlas,
        #  - y una cota para posibles equipamientos de taladro.
        pos, battery, taladro, bodega, igneas, sedim = state
        pendientes = igneas + sedim
        cargadas = len(bodega)

        if not pendientes and cargadas == 0:
            return 0

        movimiento_lb = 0
        if pendientes:
            # distancia máxima desde el rover hasta alguna muestra, luego ajustada
            dist_rover_muestras = [abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in pendientes]
            movimiento_lb = (max(dist_rover_muestras) + 1) // 2

        recolectar_lb = 2 * len(pendientes)
        depositar_lb = cargadas + len(pendientes)

        tipos_pendientes = set()
        if igneas: tipos_pendientes.add("termico")
        if sedim: tipos_pendientes.add("percusion")

        equipar_lb = 0
        if tipos_pendientes:
            if taladro is None:
                equipar_lb = 3 * len(tipos_pendientes)
            elif taladro not in tipos_pendientes:
                equipar_lb = 3

        return movimiento_lb + recolectar_lb + depositar_lb + equipar_lb

# FUNCIÓN PRINCIPAL (La que llaman los Tests)
def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
    """
    Esta función debe devolver la lista de tuplas con las acciones.
    """
    # Definimos el estado inicial como una tupla (posicion, bateria, taladro, bodega, muestras_pendientes)
    # Ejemplo: ((0, 0), 20, None, (), (muestras_igneas, muestras_sedimentarias))
    # Construimos el estado inicial esperado por `RoverProblem`
    estado_inicial = (rover_inicio, bateria_inicial, None, (), tuple(muestras_igneas), tuple(muestras_sedimentarias))

    # Creamos el problema con parámetros fijos (batería máxima=20 en la consigna)
    metas_igneas = tuple(muestras_igneas)
    metas_sedimentarias = tuple(muestras_sedimentarias)
    problema = RoverProblem(estado_inicial, 20, zonas_sombra, metas_igneas, metas_sedimentarias)

    # Ejecutamos búsqueda A* (graph_search=True para evitar estados repetidos)
    resultado = astar(problema, graph_search=True)

    # `resultado.path()` devuelve una secuencia (accion, estado). Convertimos a la
    # lista simple de acciones que esperan los tests (saltando la primera acción None).
    lista_acciones = []
    if resultado:
        for accion, nuevo_estado in resultado.path():
            if accion:
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
