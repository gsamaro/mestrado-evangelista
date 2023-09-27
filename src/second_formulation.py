from itertools import chain
from typing import Dict, List

import numpy as np
import pandas as pd
from docplex.mp.model import Model

from read_file import dataCS

from pathlib import Path


def create_variables(mdl: Model, data: dataCS) -> Model:
    mdl.z = mdl.binary_var_dict(
        (
            (i, j, t)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        ),
        lb=0,
        ub=1,
        name=f"z",
    )  # indica se o setup terminou ou nao
    mdl.w = mdl.binary_var_dict(
        (
            (i, j, t)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        ),
        lb=0,
        ub=1,
        name=f"w",
    )  # indica se há setup crossover se for 1
    mdl.x = mdl.continuous_var_dict(
        (
            (i, j, t, k)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
            for k in range(data.nperiodos)
        ),
        lb=0,
        ub=1,
        name=f"x",
    )
    mdl.l = mdl.continuous_var_dict(
        (
            (i, j, t)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        ),
        lb=0,
        name="l",
    )
    mdl.f = mdl.continuous_var_dict(
        (
            (i, j, t)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        ),
        lb=0,
        name="f",
    )
    return mdl


def define_obj_function(mdl: Model, data: dataCS) -> Model:
    mtd_func = (
        mdl.sum(
            data.sc[i] * mdl.z[i, j, t]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        )
        + sum(
            data.sc[i] * mdl.w[i, j, t]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        )
        + sum(
            data.cs[i, t, k] * mdl.x[i, j, t, k]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
            for k in range(data.nperiodos)
        )
    )
    mdl.mtd_func = mtd_func
    mdl.minimize(mtd_func)
    return mdl


def constraint_demanda_satisfeita(mdl: Model, data: dataCS) -> Model:
    for i in range(data.nitems):
        for t in range(data.nperiodos):
            if data.d[i, t] > 0:
                mdl.add_constraint(
                    mdl.sum(
                        mdl.x[i, j, k, t] for j in range(data.r) for k in range(t + 1)
                    )
                    == 1
                )
    return mdl


def constraint_capacity(mdl: Model, data: dataCS) -> Model:
    for t in range(data.nperiodos):
        for j in range(data.r):
            mdl.add_constraint(
                mdl.sum(data.st[i] * mdl.z[i, j, t] for i in range(data.nitems))
                + mdl.sum(
                    data.vt[i] * data.d[i, k] * mdl.x[i, j, t, k]
                    for i in range(data.nitems)
                    for k in range(t, data.nperiodos)
                )
                + mdl.sum(mdl.l[i, j, t] for i in range(data.nitems))
                + mdl.sum(mdl.f[i, j, t] for i in range(data.nitems))
                <= data.cap[0],
                ctname="capacity",
            )
    return mdl


def constraint_setup_and_crossover(mdl: Model, data: dataCS) -> Model:
    mdl.add_constraints(
        mdl.x[i, j, t, k] <= mdl.z[i, j, t] + mdl.w[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
        for k in range(t, data.nperiodos)
    )
    return mdl


def constraint_tempo_total_setup(mdl: Model, data: dataCS) -> Model:
    for i in range(data.nitems):
        for j in range(data.r):
            for t in range(data.nperiodos):
                if t > 0:
                    mdl.add_constraint(
                        mdl.f[i, j, t] + mdl.l[i, j, t - 1]
                        == mdl.w[i, j, t] * data.st[i]
                    )
                else:
                    mdl.add_constraint(mdl.f[i, j, t] == mdl.w[i, j, t] * data.st[i])

    return mdl


def constraint_setup_max_um_periodo(mdl: Model, data: dataCS) -> Model:
    for j in range(data.r):
        for t in range(1, data.nperiodos):
            mdl.add_constraint(
                mdl.sum(mdl.w[i, j, t - 1] for i in range(data.nitems)) <= 1
            )
    return mdl


def used_capacity(mdl, data):
    return (
        sum(
            data.st[i] * mdl.z[i, j, t]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        )
        + sum(
            data.vt[i] * data.d[i, k] * mdl.x[i, j, t, k]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
            for k in range(t, data.nperiodos)
        )
        + sum(
            mdl.l[i, j, t]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        )
        + sum(
            mdl.f[i, j, t]
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        )
    )


def total_setup_cost(mdl, data):
    return sum(
        data.sc[i] * mdl.z[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
    ) + sum(
        data.sc[i] * mdl.w[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
    )


def total_estoque_cost(mdl, data):
    return sum(
        data.cs[i, t, k] * mdl.x[i, j, t, k]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
        for k in range(data.nperiodos)
    )

    # Total de y[i]


def total_z(mdl, data):
    return sum(
        mdl.z[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
    )


def total_w(mdl, data):
    return sum(
        mdl.w[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
    )


def add_new_kpi(kpis: Dict[str, any], result, data: dataCS) -> dict:
    kpis["Instance"] = data.instance
    kpis["Best Bound"] = result.solve_details.best_bound
    kpis["Gap"] = result.solve_details.gap
    kpis["Nodes Processed"] = result.solve_details.gap
    kpis["Tempo de Solução"] = result.solve_details.time
    kpis["capacity"] = data.cap[0]
    kpis["utilization_capacity"] = (
        100 * kpis.get("used_capacity", 0) / (data.cap[0] * data.r * data.nperiodos)
    )
    kpis["nmaquinas"] = data.r
    return kpis


def closest_to_75_percent(results_per_instance: List[Dict[str, any]]) -> Dict[str, any]:
    """Dado uma lista de resultados para uma instância, retorne aquele mais próximo de 75% de utilização da capacidade."""
    return min(results_per_instance, key=lambda x: abs(x["utilization_capacity"] - 75))


def build_model(data: dataCS, capacity: float) -> (Model, dataCS):
    data.cap[0] = capacity
    mdl = Model(name="mtd")
    mdl = create_variables(mdl, data)
    mdl = define_obj_function(mdl, data)
    mdl = constraint_demanda_satisfeita(mdl, data)
    mdl = constraint_capacity(mdl, data)
    mdl = constraint_setup_and_crossover(mdl, data)
    mdl = constraint_setup_max_um_periodo(mdl, data)
    mdl.add_kpi(total_setup_cost(mdl, data), "total_setup_cost")
    mdl.add_kpi(total_estoque_cost(mdl, data), "total_estoque_cost")
    mdl.add_kpi(used_capacity(mdl, data), "used_capacity")
    mdl.add_kpi(total_w(mdl, data), "total_w")
    mdl.add_kpi(total_z(mdl, data), "total_z")
    return mdl, data


def choose_capacity(
    dataset: str, nmaquinas: int = 2, timelimit: int = 3, get_closest: bool = True
) -> Dict[str, any]:
    data = dataCS(dataset, r=nmaquinas)
    original_capacity = data.cap[0] / data.r
    instance_results = []

    for cap in np.linspace(
        original_capacity, original_capacity * 2, num=5, endpoint=True
    ):
        data = dataCS(dataset, r=nmaquinas)
        mdl, data = build_model(data, np.ceil(cap))
        mdl.parameters.timelimit = timelimit
        result = mdl.solve()

        if result == None:
            print_info(data, "infactível")
            continue

        kpis = mdl.kpis_as_dict(result, objective_key="objective_function")
        kpis = add_new_kpi(kpis, result, data)

        assert kpis["utilization_capacity"] <= 100, "Capacidade > 100%"

        instance_results.append(kpis)
        print_info(data, "concluído")
    if get_closest:
        return closest_to_75_percent(instance_results)
    else:
        return instance_results


def print_info(data: dataCS, status: str) -> None:
    print(
        f"Instance = {data.instance} Cap = {data.cap[0]} nmaquinas = {data.r} {status}"
    )


def solve_optimized_model(
    dataset: str, capacity: float, nmaquinas: int = 8, timelimit: int = 10
) -> Dict[str, any]:
    data = dataCS(dataset, r=nmaquinas)
    mdl, data = build_model(data, capacity)
    mdl.parameters.timelimit = timelimit
    result = mdl.solve()

    if result == None:
        print_info(data, "infactível")
        return None

    kpis = mdl.kpis_as_dict(result, objective_key="objective_function")
    kpis = add_new_kpi(kpis, result, data)

    # Cálculo da relaxação linear
    relaxed_model = mdl.clone()
    status = relaxed_model.solve(url=None, key=None, log_output=False)

    relaxed_objective_value = relaxed_model.objective_value
    kpis["Relaxed Objective Value"] = relaxed_objective_value

    print_info(data, "concluído")

    return kpis


def running_all_instance_choose_capacity() -> pd.DataFrame:
    # Executando e coletando os resultados
    final_results = []

    for letter in ["F"]:  ## ["F", "G"]
        for i in range(1, 6):
            for nmaq in [2, 4]:  ## [2, 4, 8]
                dataset = f"{letter}{i}.DAT"
                best_result = choose_capacity(dataset, nmaquinas=nmaq)

                if best_result:
                    final_results.append(best_result)

    if isinstance(final_results[0], list):
        df_results_optimized = pd.DataFrame(list(chain.from_iterable(final_results)))
    else:
        df_results_optimized = pd.DataFrame(final_results)
    df_results_optimized.to_excel(Path.resolve(Path.cwd() / "resultados" / "capacidadesfç2.xlsx"), index=False)
    print("Processamento de capacidades concluído.")
    return df_results_optimized


def running_all_instance_with_chosen_capacity():
    final_results = []

    pdf_capacidades = pd.read_excel(Path.resolve(Path.cwd() / "resultados" / "capacidadesfç2.xlsx"), engine="openpyxl")

    for letter in ["F", "G"]:  ## ["F", "G"]
        for i in range(1, 4):
            for nmaq in [2, 4]:
                dataset = f"{letter}{i}.DAT"

                cap = pdf_capacidades.query(
                    f"Instance == '{dataset}' and nmaquinas == {nmaq}"
                )["capacity"].values

                if len(cap) == 0:
                    print(f"Instance = {dataset} nmaquinas = {nmaq} not found")
                    continue

                best_result = solve_optimized_model(
                    dataset, capacity=cap[0], nmaquinas=nmaq, timelimit=15
                )

                if best_result:
                    final_results.append(best_result)

    df_results_optimized = pd.DataFrame(final_results)
    df_results_optimized.to_excel(Path.resolve(Path.cwd() / "resultados" / "otimizadosfç2.xlsx"), index=False)
    pass


if __name__ == "__main__":
    running_all_instance_choose_capacity()
    running_all_instance_with_chosen_capacity()