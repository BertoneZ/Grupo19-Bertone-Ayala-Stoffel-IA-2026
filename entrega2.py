from simpleai.search import CspProblem, backtrack, MOST_CONSTRAINED_VARIABLE, LEAST_CONSTRAINING_VALUE

def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    total_filas = camp_size[0]
    total_columnas = camp_size[1]
    
    variables = []
    for i in range(airlocks): 
        variables.append(f"air_{i}")
    for i in range(habs): 
        variables.append(f"hab_{i}")
    for i in range(generators): 
        variables.append(f"gen_{i}")
    for i in range(labs): 
        variables.append(f"lab_{i}")
    for i in range(deposits): 
        variables.append(f"dep_{i}")
        
    celdas_validas = []

    for f in range(total_filas):
        for c in range(total_columnas):
            if (f, c) not in craters:
                celdas_validas.append((f, c))

    dominios = {}

    dominio_habitacionales = []

    for coordenada in celdas_validas:
        f = coordenada[0]
        c = coordenada[1]
        if not (f == 0 or f == total_filas - 1 or c == 0 or c == total_columnas - 1):
            dominio_habitacionales.append((f, c))

    for i in range(habs):
        dominios[f"hab_{i}"] = dominio_habitacionales

    dominio_airlocks = []

    for coordenada in celdas_validas:
        f = coordenada[0]
        c = coordenada[1]
        if f == 0 or f == total_filas - 1 or c == 0 or c == total_columnas - 1:
            dominio_airlocks.append((f, c))

    for i in range(airlocks):
        dominios[f"air_{i}"] = dominio_airlocks

    for i in range(generators): 
        dominios[f"gen_{i}"] = celdas_validas
    for i in range(labs):       
        dominios[f"lab_{i}"] = celdas_validas
    for i in range(deposits):   
        
        dominios[f"dep_{i}"] = celdas_validas

    
    def restriccion_no_superposicion(modulos, coordenadas):
        if coordenadas[0] is None or coordenadas[1] is None: 
            return True
        return coordenadas[0] != coordenadas[1]

    def son_adyacentes(posicion1, posicion2):
        f1, c1 = posicion1
        f2, c2 = posicion2
        return abs(f1 - f2) + abs(c1 - c2) == 1
        
    def restriccion_modulos_separados(modulos, coordenadas):
        if coordenadas[0] is None or coordenadas[1] is None: 
            return True
        estan_juntos = son_adyacentes(coordenadas[0], coordenadas[1])
        if estan_juntos == True:
            return False
        else:            
            return True

    def restriccion_cadena_suministro(modulos, coordenadas):
        if coordenadas[0] is None: 
            return True
        
        posicion_lab = coordenadas[0]
        posicion_depositos = []

        for coord in coordenadas[1:]:
            if coord is not None:
                posicion_depositos.append(coord)

        if posicion_depositos:
            return any(son_adyacentes(posicion_lab, pos_dep) for pos_dep in posicion_depositos)
            
        return True
    def restriccion_ruta_evacuacion(modulos, coordenadas):
        if coordenadas[0] is None: return True

        pos_hab = coordenadas[0]
        pos_otros = []

        for c in coordenadas[1:]:
            if c is not None:
                pos_otros.append(c)
        
        if len(pos_otros) == len(variables) - 1:
            f, c = pos_hab
            vecinos = [(f-1, c), (f+1, c), (f, c-1), (f, c+1)]
            salidas_libres = 0
            for v in vecinos:
                if (v in celdas_validas) and (v not in pos_otros):
                    salidas_libres += 1
            if salidas_libres > 0:
                return True
            else:
                return False
            
        return True

    restricciones = []
    
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            restricciones.append(((variables[i], variables[j]), restriccion_no_superposicion))
        
    for i in range(generators):
        for j in range(habs):
            restricciones.append(((f"gen_{i}", f"hab_{j}"), restriccion_modulos_separados))
     
    for i in range(generators):
        for j in range(i + 1, generators):
            restricciones.append(((f"gen_{i}", f"gen_{j}"), restriccion_modulos_separados))

    lista_depositos = [f"dep_{i}" for i in range(deposits)]
    for i in range(labs):
        scope = tuple([f"lab_{i}"] + lista_depositos)
        restricciones.append((scope, restriccion_cadena_suministro))

    for i in range(habs):
        hab_actual = f"hab_{i}"
        demas_modulos = [v for v in variables if v != hab_actual]
        scope = tuple([hab_actual] + demas_modulos)
        restricciones.append((scope, restriccion_ruta_evacuacion))

    problema = CspProblem(variables, dominios, restricciones)
    solucion = backtrack(problema, variable_heuristic=MOST_CONSTRAINED_VARIABLE, value_heuristic=LEAST_CONSTRAINING_VALUE)
    
    if solucion is None: return None

    resultado_final = []
    orden_tipos = ["air", "hab", "gen", "lab", "dep"]
    for tipo_buscado in orden_tipos:
        for var_name, coord in solucion.items():
            if var_name.split("_")[0] == tipo_buscado:
                resultado_final.append((tipo_buscado, coord[0], coord[1]))

    return resultado_final

