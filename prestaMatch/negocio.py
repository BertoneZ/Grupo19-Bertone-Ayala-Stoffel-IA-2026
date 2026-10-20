import numpy as np
import pandas as pd

PRESUPUESTO = 2_000_000


def seleccionar_prestamos(solicitudes, prob_pago, presupuesto=PRESUPUESTO):
    prob_pago = np.asarray(prob_pago)
    monto = solicitudes["Monto_Solicitado"].to_numpy()
    tasa = solicitudes["Tasa_Interes"].to_numpy()
    assert len(prob_pago) == len(monto) and np.all((prob_pago >= 0) & (prob_pago <= 1))

    ganancia_por_dolar = prob_pago * (1 + tasa) - 1
    aprobar = np.zeros(len(monto), dtype=int)
    capital_usado = 0
    for i in np.argsort(-ganancia_por_dolar, kind="stable"):
        if ganancia_por_dolar[i] <= 0:
            break
        if capital_usado + monto[i] <= presupuesto:
            aprobar[i] = 1
            capital_usado += monto[i]

    assert monto[aprobar == 1].sum() <= presupuesto
    return aprobar


def ganancia_real(solicitudes, aprobar):
    aprobar = np.asarray(aprobar) == 1
    pago = solicitudes["Estado_Final"].to_numpy() == 1
    monto = solicitudes["Monto_Solicitado"].to_numpy()
    tasa = solicitudes["Tasa_Interes"].to_numpy()
    return (monto * tasa)[aprobar & pago].sum() - monto[aprobar & ~pago].sum()


def simular_ganancia_como_test(solicitudes, prob_pago, fraccion_presupuesto):
    datos = solicitudes.assign(Prob_Pago=np.asarray(prob_pago))
    ganancias = []
    for _, mes in datos.groupby("Fecha_Solicitud"):
        presupuesto_mes = fraccion_presupuesto * mes["Monto_Solicitado"].sum()
        aprobar = seleccionar_prestamos(mes, mes["Prob_Pago"], presupuesto_mes)
        ganancias.append(ganancia_real(mes, aprobar) * PRESUPUESTO / presupuesto_mes)
    return float(np.mean(ganancias))


def guardar_submission(test, aprobar, ruta):
    submission = pd.DataFrame({
        "ID_Solicitud": test["ID_Solicitud"].to_numpy(),
        "Decision_aprobacion": np.asarray(aprobar, dtype=int),
    })
    capital = test["Monto_Solicitado"].to_numpy()[submission["Decision_aprobacion"] == 1].sum()
    if len(submission) != len(test) or capital > PRESUPUESTO:
        raise ValueError("Submission invalida: no se guarda")
    ruta.parent.mkdir(exist_ok=True)
    submission.to_csv(ruta, index=False)
    print(f"  {ruta.name}: {submission['Decision_aprobacion'].sum()} aprobados | "
          f"capital usado USD {capital:,.0f} de {PRESUPUESTO:,.0f}")
    return submission
