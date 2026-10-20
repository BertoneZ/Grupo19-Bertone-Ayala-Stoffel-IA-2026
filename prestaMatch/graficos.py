import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, NARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
GRIS_CLARO, GRIS_MEDIO = "#d9d8d3", "#b9b8b2"
TEXTO, TEXTO_SUAVE, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": GRILLA, "axes.labelcolor": TEXTO_SUAVE,
    "xtick.color": TEXTO_SUAVE, "ytick.color": TEXTO_SUAVE, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlecolor": TEXTO,
    "axes.spines.top": False, "axes.spines.right": False,
})


def formato_miles(valor):
    return f"{valor:,.0f}".replace(",", ".")


def grafico_pagan_vs_no_pagan(train, bancario, ruta):
    por_usuario = bancario.groupby("ID_Usuario").agg(
        Saldo=("Saldo_Promedio_Mensual", "mean"), Atraso=("Dias_Atraso_Otras_Deudas", "mean")
    )
    datos = train[["ID_Usuario", "Estado_Final"]].merge(por_usuario, left_on="ID_Usuario", right_index=True)
    promedios = datos.groupby("Estado_Final")[["Saldo", "Atraso"]].mean()

    fig, ejes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for eje, columna, titulo, texto in [
        (ejes[0], "Saldo", "Saldo promedio en el banco (USD)", formato_miles),
        (ejes[1], "Atraso", "Días de atraso en otras deudas", lambda v: f"{v:.0f} días"),
    ]:
        valores = [promedios.loc[1, columna], promedios.loc[0, columna]]
        barras = eje.barh(["Pagaron", "No pagaron"], valores, color=[AZUL, NARANJA], height=0.55)
        for barra, valor in zip(barras, valores):
            eje.text(valor * 1.02, barra.get_y() + barra.get_height() / 2, texto(valor), va="center", color=TEXTO)
        eje.set_title(titulo, loc="left")
        eje.invert_yaxis()
        eje.set_xlim(0, max(valores) * 1.3)
        eje.xaxis.set_visible(False)
        eje.spines["bottom"].set_visible(False)
        eje.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=3)
    fig.savefig(ruta, dpi=220)
    plt.close(fig)
    return promedios


def tasa_promedio_por_tramo(monto, ingreso, tasa, tramos):
    log_relacion = np.log(monto / ingreso)
    resumen = pd.Series(tasa.to_numpy()).groupby(pd.cut(log_relacion.to_numpy(), tramos), observed=False).agg(["mean", "size"])
    return (resumen["mean"].where(resumen["size"] >= 30) * 100).to_numpy()


def grafico_escala_y_ganancia(train, test, factor_monto, ganancias, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})

    eje = ejes[0]
    tramos = np.linspace(-1.5, 2.6, 18)
    centros = (tramos[:-1] + tramos[1:]) / 2
    curvas = [
        ("Train", train["Monto_Solicitado"], train, AZUL, "-"),
        ("Test como viene", test["Monto_Solicitado"], test, NARANJA, "-"),
        (f"Test x {factor_monto:.2f}".replace(".", ","), test["Monto_Solicitado"] * factor_monto, test, VERDE, (0, (4, 2))),
    ]
    for nombre, monto, datos, color, linea in curvas:
        tasas = tasa_promedio_por_tramo(monto, datos["Ingreso_Mensual"], datos["Tasa_Interes"], tramos)
        eje.plot(centros, tasas, color=color, lw=2, ls=linea, label=nombre)
    marcas = [0.25, 0.5, 1, 2, 4, 8]
    eje.set_xticks(np.log(marcas))
    eje.set_xticklabels([str(m).replace(".", ",") for m in marcas])
    eje.set_xlabel("Monto pedido / sueldo mensual")
    eje.set_ylabel("Tasa de interés (%)")
    eje.set_title("La tasa delata la escala del test", loc="left")
    eje.grid(axis="y", color=GRILLA, lw=0.6)
    eje.legend(frameon=False, fontsize=7.5, loc="upper left")

    eje = ejes[1]
    nombres = list(ganancias)
    valores = [ganancias[n] / 1000 for n in nombres]
    barras = eje.bar(nombres, valores, color=[GRIS_MEDIO, AZUL, GRIS_CLARO], width=0.6)
    for barra, valor in zip(barras, valores):
        eje.text(barra.get_x() + barra.get_width() / 2, valor + 10, f"{valor:.0f} mil", ha="center", color=TEXTO)
    eje.set_ylim(0, max(valores) * 1.15)
    eje.yaxis.set_visible(False)
    eje.spines["left"].set_visible(False)
    eje.tick_params(axis="x", length=0, labelsize=8)
    eje.set_title("Ganancia simulada (USD, escala 2M)", loc="left")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(ruta, dpi=220)
    plt.close(fig)
