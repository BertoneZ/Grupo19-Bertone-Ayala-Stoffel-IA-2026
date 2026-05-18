
from simpleai.search import SearchProblem, astar
 
# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BATERIA_MAX        = 20
CAP_BODEGA         = 2
 
COSTO_MOVER        = 1   # batería
COSTO_SOBREMARCHA  = 4   # batería
COSTO_EQUIPAR      = 1   # batería
COSTO_RECOLECTAR   = 3   # batería
RECARGA_BATERIA    = 10
 
TIEMPO_MOVER       = 1
TIEMPO_SOBREMARCHA = 1
TIEMPO_EQUIPAR     = 3
TIEMPO_RECOLECTAR  = 2
TIEMPO_RECARGAR    = 4
TIEMPO_DEPOSITAR_X_MUESTRA = 1  # por muestra en bodega
 
TALADRO_IGNEA      = "termico"
TALADRO_SEDIMENT   = "percusion"
 
PUNTO_DEPOSITO     = (0, 0)   # coordenada fija de extracción
 
# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class RoverProblem(SearchProblem):
    """
    Modela el problema del rover marciano como un espacio de estados
    para ser resuelto con A* de SimpleAI.
    """
 
    def __init__(
        self,
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias,
    ):
        self.zonas_sombra = frozenset(zonas_sombra)
 
        # Estado inicial como tupla hashable
        estado_inicial = (
            rover_inicio[0],                    # fila
            rover_inicio[1],                    # columna
            bateria_inicial,                    # batería
            None,                               # taladro (sin equipar)
            (),                                 # bodega vacía
            tuple(sorted(muestras_igneas)),     # muestras ígneas pendientes
            tuple(sorted(muestras_sedimentarias)),  # muestras sedimentarias pendientes
        )
 
        super().__init__(initial_state=estado_inicial)
 
    # ------------------------------------------------------------------
    # Desempaquetado de estado (helper interno)
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(state):
        fila, col, bateria, taladro, bodega, m_igneas, m_sediment = state
        return fila, col, bateria, taladro, bodega, m_igneas, m_sediment
 
    # ------------------------------------------------------------------
    # Acciones disponibles desde un estado
    # ------------------------------------------------------------------
    def actions(self, state):
        fila, col, bateria, taladro, bodega, m_igneas, m_sediment = self._unpack(state)
        pos = (fila, col)
        acciones = []
 
        # ── 1. MOVERSE (4 direcciones, cuesta 1 batería) ─────────────
        if bateria - COSTO_MOVER > 0:
            for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                acciones.append(("moverse", (fila + df, col + dc)))
 
        # ── 2. SOBREMARCHA (2 casillas, cuesta 4 batería) ────────────
        if bateria - COSTO_SOBREMARCHA > 0:
            for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                acciones.append(("sobremarcha", (fila + 2 * df, col + 2 * dc)))
 
        # ── 3. EQUIPAR TALADRO (cuesta 1 batería) ────────────────────
        if bateria - COSTO_EQUIPAR > 0:
            for tipo in (TALADRO_IGNEA, TALADRO_SEDIMENT):
                if taladro != tipo:
                    acciones.append(("equipar", tipo))
 
        # ── 4. RECOLECTAR (cuesta 3 batería, bodega no llena) ────────
        if bateria - COSTO_RECOLECTAR > 0 and len(bodega) < CAP_BODEGA:
            if pos in m_igneas and taladro == TALADRO_IGNEA:
                acciones.append(("recolectar", "ignea"))
            if pos in m_sediment and taladro == TALADRO_SEDIMENT:
                acciones.append(("recolectar", "sedimentaria"))
 
        # ── 5. DEPOSITAR ──────────────────────────────────────────────
        # Solo si la bodega tiene 2 muestras O es la última muestra del mapa
        if pos == PUNTO_DEPOSITO and len(bodega) > 0:
            total_restantes = len(m_igneas) + len(m_sediment)
            ultima_muestra = (total_restantes == 0 and len(bodega) > 0)
            bodega_llena   = (len(bodega) == CAP_BODEGA)
            if bodega_llena or ultima_muestra:
                acciones.append(("depositar", None))
 
        # ── 6. RECARGAR (cuesta 0 batería, suma +10) ─────────────────
        if pos not in self.zonas_sombra and bateria < BATERIA_MAX:
            acciones.append(("recargar", None))
 
        return acciones
 
    # ------------------------------------------------------------------
    # Estado resultante tras aplicar una acción
    # ------------------------------------------------------------------
    def result(self, state, action):
        fila, col, bateria, taladro, bodega, m_igneas, m_sediment = self._unpack(state)
        accion, param = action
 
        if accion == "moverse":
            nueva_fila, nueva_col = param
            return (
                nueva_fila, nueva_col,
                bateria - COSTO_MOVER,
                taladro, bodega, m_igneas, m_sediment,
            )
 
        if accion == "sobremarcha":
            nueva_fila, nueva_col = param
            return (
                nueva_fila, nueva_col,
                bateria - COSTO_SOBREMARCHA,
                taladro, bodega, m_igneas, m_sediment,
            )
 
        if accion == "equipar":
            return (
                fila, col,
                bateria - COSTO_EQUIPAR,
                param,          # nuevo taladro
                bodega, m_igneas, m_sediment,
            )
 
        if accion == "recolectar":
            nueva_bodega = bodega + (param,)
            if param == "ignea":
                nuevas_igneas = tuple(m for m in m_igneas if m != (fila, col))
                return (
                    fila, col,
                    bateria - COSTO_RECOLECTAR,
                    taladro, nueva_bodega, nuevas_igneas, m_sediment,
                )
            else:  # sedimentaria
                nuevas_sediment = tuple(m for m in m_sediment if m != (fila, col))
                return (
                    fila, col,
                    bateria - COSTO_RECOLECTAR,
                    taladro, nueva_bodega, m_igneas, nuevas_sediment,
                )
 
        if accion == "depositar":
            return (
                fila, col,
                bateria,
                taladro,
                (),             # bodega vacía
                m_igneas, m_sediment,
            )
 
        if accion == "recargar":
            nueva_bateria = min(bateria + RECARGA_BATERIA, BATERIA_MAX)
            return (
                fila, col,
                nueva_bateria,
                taladro, bodega, m_igneas, m_sediment,
            )
 
        raise ValueError(f"Acción desconocida: {accion}")
 
    # ------------------------------------------------------------------
    # Condición de meta
    # ------------------------------------------------------------------
    def is_goal(self, state):
        _, _, _, _, bodega, m_igneas, m_sediment = self._unpack(state)
        return len(bodega) == 0 and len(m_igneas) == 0 and len(m_sediment) == 0
 
    # ------------------------------------------------------------------
    # Función de costo (tiempo en minutos)
    # ------------------------------------------------------------------
    def cost(self, state, action, state2):
        accion, _ = action
        _, _, _, _, bodega, _, _ = self._unpack(state)
 
        costos = {
            "moverse":    TIEMPO_MOVER,
            "sobremarcha": TIEMPO_SOBREMARCHA,
            "equipar":    TIEMPO_EQUIPAR,
            "recolectar": TIEMPO_RECOLECTAR,
            "recargar":   TIEMPO_RECARGAR,
        }
 
        if accion == "depositar":
            return len(bodega) * TIEMPO_DEPOSITAR_X_MUESTRA
 
        return costos.get(accion, 1)
 
    # ------------------------------------------------------------------
    # Heurística admisible y consistente
    # ------------------------------------------------------------------
    def heuristic(self, state):
        """
        Estimación optimista (sin sobreestimar) del costo restante.
 
        Razonamiento por componentes:
        ─────────────────────────────
        Para cada muestra pendiente calculamos el costo mínimo de:
          1. Ir desde el rover hasta la muestra       → distancia Manhattan / 2
             (sobremarcha cubre 2 casillas por 1 min)
          2. Recolectarla                              → TIEMPO_RECOLECTAR = 2 min
          3. Ir al punto de depósito                  → dist(muestra, depósito) / 2
          4. Depositar                                 → TIEMPO_DEPOSITAR_X_MUESTRA = 1 min
 
        También consideramos:
          - Si la bodega ya tiene muestras: costo mínimo de ir a depositar y depositar.
          - Dividimos entre 2 las distancias porque con sobremarcha se recorre
            2 casillas en 1 minuto (costo 1), siendo el movimiento más barato posible.
 
        La heurística es admisible porque:
          - Usa el movimiento más barato (sobremarcha, 0.5 min/casilla).
          - Ignora restricciones de batería y cambios de taladro.
          - Cada muestra se cuenta una sola vez.
        """
        fila, col, _, _, bodega, m_igneas, m_sediment = self._unpack(state)
        pos = (fila, col)
 
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
        def tiempo_min_distancia(d):
            """Tiempo mínimo para recorrer d casillas (sobremarcha = 2 cas / min)."""
            return d / 2.0
 
        h = 0.0
        todas_las_muestras = (
            [(m, "ignea") for m in m_igneas]
            + [(m, "sedimentaria") for m in m_sediment]
        )
 
        # Costo de la bodega ya cargada: ir al depósito y depositar
        if len(bodega) > 0:
            dist_deposito = manhattan(pos, PUNTO_DEPOSITO)
            h += tiempo_min_distancia(dist_deposito) + len(bodega) * TIEMPO_DEPOSITAR_X_MUESTRA
 
        # Costo mínimo por cada muestra pendiente
        for muestra_pos, _ in todas_las_muestras:
            dist_ir       = manhattan(pos, muestra_pos)
            dist_volver   = manhattan(muestra_pos, PUNTO_DEPOSITO)
 
            h += (
                tiempo_min_distancia(dist_ir)      # ir a la muestra
                + TIEMPO_RECOLECTAR                # recolectarla
                + tiempo_min_distancia(dist_volver) # volver al depósito
                + TIEMPO_DEPOSITAR_X_MUESTRA       # depositar (al menos 1 min)
            )
 
            # Para las siguientes muestras, asumimos que el rover parte del depósito
            pos = PUNTO_DEPOSITO
 
        return h
 
 
# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def planear_rover(
    rover_inicio,
    bateria_inicial,
    zonas_sombra,
    muestras_igneas,
    muestras_sedimentarias,
):
    """
    Instancia el problema y ejecuta A* de SimpleAI.
 
    Parámetros
    ----------
    rover_inicio             : (fila, col) posición inicial del rover
    bateria_inicial          : int, batería de inicio (≤ 20)
    zonas_sombra             : iterable de (fila, col) donde NO se recarga
    muestras_igneas          : iterable de (fila, col) de muestras ígneas
    muestras_sedimentarias   : iterable de (fila, col) de muestras sedimentarias
 
    Retorna
    -------
    list of actions  : secuencia de (accion, parametro) que lleva al estado meta
                       o None si no hay solución.
    """
    problema = RoverProblem(
        rover_inicio=rover_inicio,
        bateria_inicial=bateria_inicial,
        zonas_sombra=zonas_sombra,
        muestras_igneas=muestras_igneas,
        muestras_sedimentarias=muestras_sedimentarias,
    )
 
    resultado = astar(problema, graph_search=True)
 
    if resultado is None:
        print("No se encontró solución.")
        return None
 
    # SimpleAI retorna el nodo final; extraemos la secuencia de acciones
    acciones = [accion for accion, _ in resultado.path()]
    return acciones
 
 
# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ── Configuración del escenario ──────────────────────────────────────
    rover_inicio           = (3, 3)
    bateria_inicial        = 15
    zonas_sombra           = [(2, 2), (3, 1)]
    muestras_igneas        = [(1, 1)]
    muestras_sedimentarias = [(2, 3)]
 
    print("=" * 60)
    print("  ROVER MARCIANO — Planificación con A*")
    print("=" * 60)
    print(f"  Posición inicial    : {rover_inicio}")
    print(f"  Batería inicial     : {bateria_inicial}")
    print(f"  Zonas de sombra     : {zonas_sombra}")
    print(f"  Muestras ígneas     : {muestras_igneas}")
    print(f"  Muestras sediment.  : {muestras_sedimentarias}")
    print(f"  Punto de depósito   : {PUNTO_DEPOSITO}")
    print("=" * 60)
 
    plan = planear_rover(
        rover_inicio=rover_inicio,
        bateria_inicial=bateria_inicial,
        zonas_sombra=zonas_sombra,
        muestras_igneas=muestras_igneas,
        muestras_sedimentarias=muestras_sedimentarias,
    )
 
    if plan:
        print(f"\nSolución encontrada en {len(plan)} pasos:\n")
        for i, accion in enumerate(plan, 1):
            nombre, param = accion
            detalle = f"→ {param}" if param is not None else ""
            print(f"  {i:>3}. {nombre:<12} {detalle}")
    else:
        print("\nNo se encontró un plan válido.")