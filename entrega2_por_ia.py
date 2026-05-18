"""
Entrega 2: Ares-1 — Diseño del campamento base
Solución generada íntegramente por Claude (Anthropic) como herramienta de IA generativa.

FORMULACIÓN CSP:
  Variables: una por módulo a ubicar, con nombre "{tipo}_{índice}".
    Ejemplo: "hab_0", "hab_1", "gen_0", "air_0", "lab_0", "dep_0", "dep_1".

  Dominios (celdas válidas por tipo):
    - "air_*"  → celdas del borde no craterosas             (R3 en dominio)
    - "hab_*"  → celdas interiores no craterosas            (R4 en dominio)
    - "gen_*", "lab_*", "dep_*" → cualquier celda no cráter (R2 en dominio)

  Restricciones:
    R1 sin superposición    → binaria entre todo par de variables
    R5 gen no adyacente hab → binaria entre cada (gen_i, hab_j)
    R6 gens no adyacentes   → binaria entre cada (gen_i, gen_j)
    R7 lab adyacente a dep  → n-aria (lab_i, dep_0, …, dep_k)
    R8 evacuación hab       → verificación incremental en backtracking propio

  Resolución:
    Backtracking con MCV (Minimum Remaining Values) y forward-checking para
    las restricciones binarias. R8 se verifica de forma incremental: al asignar
    cualquier módulo se comprueba que ningún hab ya asignado pierda su último
    vecino potencialmente libre.
"""

from itertools import combinations


class _Problem:
    """Contenedor mínimo compatible con el motor de backtrack de SimpleAI."""
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.var_degrees = {
            v: sum(1 for sc, _ in constraints if v in sc) for v in variables
        }


def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    """
    Resuelve el problema de diseño del campamento marciano como un CSP.
    Retorna lista de (tipo, fila, columna) o None si no existe solución.
    """
    rows, cols = camp_size
    craters_set = frozenset(craters)

    # ── Utilidades de geometría ─────────────────────────────────────────────────

    def is_border(r, c):
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def are_adjacent(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def get_neighbors(r, c):
        return frozenset(
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < rows and 0 <= c + dc < cols
        )

    neighbor_cache = {
        (r, c): get_neighbors(r, c)
        for r in range(rows) for c in range(cols)
    }

    # ── Construcción de dominios ────────────────────────────────────────────────

    all_free      = [(r, c) for r in range(rows) for c in range(cols)
                     if (r, c) not in craters_set]
    border_free   = [(r, c) for r, c in all_free if is_border(r, c)]
    interior_free = [(r, c) for r, c in all_free if not is_border(r, c)]

    # ── Variables ──────────────────────────────────────────────────────────────

    variables = []
    domains = {}

    def register(prefix, count, domain):
        for i in range(count):
            name = f"{prefix}_{i}"
            variables.append(name)
            domains[name] = list(domain)

    register("hab", habs,       interior_free)   # más restringidas primero
    register("air", airlocks,   border_free)
    register("gen", generators, all_free)
    register("lab", labs,       all_free)
    register("dep", deposits,   all_free)

    if not variables:
        return []

    hab_vars = [v for v in variables if v.startswith("hab_")]
    gen_vars = [v for v in variables if v.startswith("gen_")]
    lab_vars = [v for v in variables if v.startswith("lab_")]
    dep_vars = [v for v in variables if v.startswith("dep_")]

    # ── Funciones de restricción (compatibles con SimpleAI) ────────────────────

    def no_overlap(_, vals):
        """R1: sin superposición."""
        return vals[0] != vals[1]

    def gen_not_adj_hab(_, vals):
        """R5: generador no adyacente a habitacional."""
        return not are_adjacent(vals[0], vals[1])

    def gens_not_adjacent(_, vals):
        """R6: generadores no adyacentes entre sí."""
        return not are_adjacent(vals[0], vals[1])

    def lab_has_adjacent_dep(_, vals):
        """R7: laboratorio adyacente a al menos un depósito."""
        return any(are_adjacent(vals[0], dp) for dp in vals[1:])

    # ── Construcción del CSP ────────────────────────────────────────────────────

    constraints = []

    for v1, v2 in combinations(variables, 2):
        constraints.append(((v1, v2), no_overlap))

    for gv in gen_vars:
        for hv in hab_vars:
            constraints.append(((gv, hv), gen_not_adj_hab))

    for g1, g2 in combinations(gen_vars, 2):
        constraints.append(((g1, g2), gens_not_adjacent))

    if lab_vars and dep_vars:
        for lv in lab_vars:
            scope = (lv,) + tuple(dep_vars)
            constraints.append((scope, lab_has_adjacent_dep))

    # Índice: variable → restricciones que la involucran
    var_constraints = {v: [] for v in variables}
    for scope, fn in constraints:
        for v in scope:
            var_constraints[v].append((scope, fn))

    # ── Resolución: backtracking con MCV, forward-checking y R8 incremental ────

    def is_consistent(assignment, variable, value):
        """Verifica R1/R5/R6/R7 para la nueva asignación parcial."""
        tentative = {**assignment, variable: value}
        for scope, fn in var_constraints[variable]:
            if all(v in tentative for v in scope):
                if not fn(scope, tuple(tentative[v] for v in scope)):
                    return False
        return True

    def evacuation_ok(assignment, new_var, new_val):
        """
        R8 incremental: verifica que ningún hab ya asignado pierda su
        último vecino potencialmente libre al agregar new_var = new_val.
        """
        occupied = set(assignment.values()) | {new_val} | craters_set
        for hv in hab_vars:
            pos = assignment.get(hv) if hv != new_var else new_val
            if pos is None:
                continue
            if neighbor_cache[pos].issubset(occupied):
                return False
        return True

    def forward_check(current_domains, variable, value):
        """
        Propagación binaria: elimina valores incompatibles de las variables
        pendientes conectadas a `variable` mediante restricciones binarias.
        Retorna nuevos dominios o None si algún dominio queda vacío.
        """
        new_domains = {v: list(d) for v, d in current_domains.items()}
        new_domains[variable] = [value]

        for scope, fn in var_constraints[variable]:
            if len(scope) != 2:
                continue
            other = scope[0] if scope[1] == variable else scope[1]
            filtered = [
                v for v in new_domains[other]
                if fn(scope, (value, v) if scope[0] == variable else (v, value))
            ]
            if not filtered:
                return None
            new_domains[other] = filtered

        return new_domains

    def choose_variable(pending, current_domains):
        """MCV: variable con dominio más pequeño."""
        return min(pending, key=lambda v: len(current_domains[v]))

    def backtrack_search(assignment, current_domains):
        if len(assignment) == len(variables):
            return assignment

        pending = [v for v in variables if v not in assignment]
        var = choose_variable(pending, current_domains)

        for value in list(current_domains[var]):
            if not is_consistent(assignment, var, value):
                continue
            if not evacuation_ok(assignment, var, value):
                continue
            new_domains = forward_check(current_domains, var, value)
            if new_domains is None:
                continue
            result = backtrack_search({**assignment, var: value}, new_domains)
            if result is not None:
                return result

        return None

    solution = backtrack_search({}, domains)

    if solution is None:
        return None

    return [(var.rsplit("_", 1)[0], r, c) for var, (r, c) in solution.items()]
