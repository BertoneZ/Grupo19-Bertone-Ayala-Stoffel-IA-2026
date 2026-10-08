from simpleai.search import SearchProblem, astar, breadth_first, depth_first
from simpleai.search.viewers import WebViewer, BaseViewer

# - El archivo define la clase `RoverProblem` que adapta el estado del rover a la API
#   de simpleai.search/SearchProblem (actions/result/is_goal/cost/heuristic).
# - La función `planear_rover` crea el problema y ejecuta A* para obtener una lista de acciones que luego devuelve.

class RoverProblem(SearchProblem):
    def __init__(self, estado_inicial, bateria_max, zonas_sombra, muestras_igneas, muestras_sedimentarias):
        self.bateria_max = bateria_max
        self.zonas_sombra = zonas_sombra
        self.muestras_igneas = muestras_igneas
        self.muestras_sedimentarias = muestras_sedimentarias
        # Pasamos el estado inicial al padre para que gestione la búsqueda
        super().__init__(initial_state=estado_inicial)

    def actions(self, state):
    
        pos, bateria, taladro, bodega, igneas, sedim = state
        accs = []

       
        r, c = pos
        vecinos = [ (r+1, c), (r-1, c), (r, c+1), (r, c-1) ]
        for np in vecinos:
            # Solo permitimos moverse si la acción deja batería > 0
            if bateria - 1 > 0:
                accs.append(("moverse", np))

        overdrive = [ (r+2, c), (r-2, c), (r, c+2), (r, c-2) ]
        for np in overdrive:
            if bateria - 4 > 0:
                accs.append(("sobremarcha", np))

       
        for t in ("termico", "percusion"):
            if bateria - 1 > 0 and taladro != t:
                if t == "termico" and not igneas:
                    continue
                if t == "percusion" and not sedim:
                    continue
                accs.append(("equipar", t))

        if pos in igneas and taladro == "termico" and len(bodega) < 2 and bateria - 3 > 0:
            accs.append(("recolectar", "ignea"))
        if pos in sedim and taladro == "percusion" and len(bodega) < 2 and bateria - 3 > 0:
            accs.append(("recolectar", "sedimentaria"))

        remaining_samples = tuple(igneas) + tuple(sedim)
        if len(bodega) > 0:
            if len(bodega) == 2 or (len(bodega) == 1 and not remaining_samples):
                if bateria - 1 > 0:
                    accs.append(("depositar", None))

        umbral_recarga = 10 if len(self.zonas_sombra) > 10 else 8
        if pos not in self.zonas_sombra and bateria < self.bateria_max and bateria <= umbral_recarga:
            accs.append(("recargar", None))

        return accs

    def result(self, state, action):
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
            return (pos, bateria - 1, taladro, (), igneas, sedim)

        if nombre_accion == "recolectar":
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
        pos, bateria, taladro, bodega, igneas, sedim = state
        return len(igneas) == 0 and len(sedim) == 0 and len(bodega) == 0

    def cost(self, state, action, state2):
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
            _, _, _, bodega, _, _ = state
            return len(bodega) * 1
        return 0

    def heuristic(self, state):
        pos, battery, taladro, bodega, igneas, sedim = state
        pendientes = igneas + sedim
        cargadas = len(bodega)

        if not pendientes and cargadas == 0:
            return 0

        movimiento_lb = 0
        if pendientes:
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

def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
    """
    Esta función debe devolver la lista de tuplas con las acciones.
    """
    estado_inicial = (rover_inicio, bateria_inicial, None, (), tuple(muestras_igneas), tuple(muestras_sedimentarias))
    metas_igneas = tuple(muestras_igneas)
    metas_sedimentarias = tuple(muestras_sedimentarias)
    problema = RoverProblem(estado_inicial, 20, zonas_sombra, metas_igneas, metas_sedimentarias)
    resultado = astar(problema, graph_search=True)

    lista_acciones = []
    if resultado:
        for accion, nuevo_estado in resultado.path():
            if accion:
                lista_acciones.append(accion)

    return lista_acciones

if __name__ == "__main__":
    acciones = planear_rover(
        rover_inicio=(0, 0),
        bateria_inicial=20,
        zonas_sombra=[(0, 1)],
        muestras_igneas=[(1, 1)],
        muestras_sedimentarias=[(2, 2)]
    )
    print(acciones)
