from itertools import chain
from typing import Dict, List

import numpy as np
import pandas as pd
from docplex.mp.model import Model

from read_file import dataCS

from pathlib import Path
import os
import types

try:
    from mpi4py.futures import MPIPoolExecutor
    MPI_BOOL = True
except:
    print("mpi4py not running")
    MPI_BOOL = False

INSTANCES = [f"F{i}.DAT" for i in range(1, 2)] #+ [f"G{i}.DAT" for i in range(1, 2)]
MAQUINAS = [2, 4]

CAPACIDADES_PATH = Path.resolve(Path.cwd() / "resultados" / "capacidades_f1.xlsx")
OTIMIZADOS_PATH = Path.resolve(Path.cwd() / "resultados" / "otimizados_f1.xlsx")


def create_variables(mdl: Model, data: dataCS) -> Model:
    mdl.y = mdl.binary_var_dict(
        (
            (i, j, t)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        ),
        lb=0,
        ub=1,
        name=f"y",
    )
    mdl.v = mdl.binary_var_dict(
        (
            (i, j, t)
            for i in range(data.nitems)
            for j in range(data.r)
            for t in range(data.nperiodos)
        ),
        lb=0,
        ub=1,
        name=f"v",
    )
    mdl.u = mdl.continuous_var_dict(
        ((j, t) for j in range(data.r) for t in range(data.nperiodos)), lb=0, name=f"u"
    )  # tempo extra emprestado para o setup t + 1
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
    return mdl


def define_obj_function(mdl: Model, data: dataCS) -> Model:
    mtd_func = mdl.sum(
        data.sc[i] * mdl.y[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
    ) + sum(
        data.cs[i, t, k] * mdl.x[i, j, t, k]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
        for k in range(data.nperiodos)
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
    for j in range(data.r):
        for t in range(data.nperiodos):
            if t > 0:
                mdl.add_constraint(
                    mdl.sum(data.st[i] * mdl.y[i, j, t] for i in range(data.nitems))
                    + mdl.sum(
                        data.vt[i] * data.d[i, k] * mdl.x[i, j, t, k]
                        for i in range(data.nitems)
                        for k in range(t, data.nperiodos)
                    )
                    + mdl.u[j, t]
                    <= data.cap[0] + mdl.u[j, t - 1],
                    ctname="capacity",
                )
            else:
                mdl.add_constraint(
                    mdl.sum(data.st[i] * mdl.y[i, j, t] for i in range(data.nitems))
                    + mdl.sum(
                        data.vt[i] * data.d[i, k] * mdl.x[i, j, t, k]
                        for i in range(data.nitems)
                        for k in range(t, data.nperiodos)
                    )
                    + mdl.u[j, t]
                    <= data.cap[0]
                )
    return mdl


def constraint_setup(mdl: Model, data: dataCS) -> Model:
    mdl.add_constraints(
        mdl.x[i, j, t, k] <= mdl.y[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
        for k in range(t, data.nperiodos)
    )
    return mdl


def constraint_tempo_emprestado_crossover(mdl: Model, data: dataCS) -> Model:
    # Linha 5  Uj,t-1 <= i=1∑n Vi,j,t-1 STi,t
    for j in range(data.r):
        for t in range(1, data.nperiodos):
            mdl.add_constraint(
                mdl.u[j, t - 1]
                <= mdl.sum(mdl.v[i, j, t - 1] * data.st[i] for i in range(data.nitems))
            )
    return mdl


def constraint_proibe_crossover_sem_setup(mdl: Model, data: dataCS) -> Model:
    for i in range(data.nitems):
        for j in range(data.r):
            for t in range(1, data.nperiodos):
                mdl.add_constraint(mdl.v[i, j, t - 1] <= mdl.y[i, j, t])
    return mdl


def constraint_setup_max_um_item(mdl: Model, data: dataCS) -> Model:
    mdl.add_constraints(
        mdl.sum(mdl.v[i, j, t - 1] for i in range(data.nitems)) <= 1
        for j in range(data.r)
        for t in range(1, data.nperiodos)
    )
    return mdl


def total_setup_cost(mdl, data):
    return sum(
        data.sc[i] * mdl.y[i, j, t]
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


def used_capacity(mdl, data):
    return sum(
        data.st[i] * mdl.y[i, j, t]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
    ) + sum(
        data.vt[i] * data.d[i, k] * mdl.x[i, j, t, k]
        for i in range(data.nitems)
        for j in range(data.r)
        for t in range(data.nperiodos)
        for k in range(t, data.nperiodos)
    )


def total_y(mdl, data):
    return sum(
        mdl.y[i, j, t]
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


def build_model(data: dataCS, capacity: float) -> Model:
    data.cap[0] = capacity
    mdl = Model(name="mtd")
    mdl.context.cplex_parameters.threads = 1
    mdl = create_variables(mdl, data)
    mdl = define_obj_function(mdl, data)
    mdl = constraint_demanda_satisfeita(mdl, data)
    mdl = constraint_capacity(mdl, data)
    mdl = constraint_setup(mdl, data)
    mdl = constraint_tempo_emprestado_crossover(mdl, data)
    mdl = constraint_proibe_crossover_sem_setup(mdl, data)
    mdl = constraint_setup_max_um_item(mdl, data)

    mdl.add_kpi(total_setup_cost(mdl, data), "total_setup_cost")
    mdl.add_kpi(total_estoque_cost(mdl, data), "total_estoque_cost")
    mdl.add_kpi(used_capacity(mdl, data), "used_capacity")
    mdl.add_kpi(total_y(mdl, data), "total_y")
    return mdl, data


def choose_capacity(
    dataset: str, nmaquinas: int = 2, timelimit: int = 10, get_closest: bool = True
) -> Dict[str, any]:
    data = dataCS(dataset, r=nmaquinas)
    original_capacity = data.cap[0] / data.r
    instance_results = []

    for cap in np.linspace(
        original_capacity, original_capacity * 2, num=5, endpoint=True
    ):
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
            f"Instance = {data.instance} Cap = {data.cap[0]} nmaquinas = {data.r} {status} Process {os.getppid()}"
        )


def solve_optimized_model(
    dataset: str, capacity: float, nmaquinas: int = 8, timelimit: int = 3600
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

    if not MPI_BOOL:
        for dataset in INSTANCES:
            for nmaq in MAQUINAS:
                best_result = choose_capacity(dataset, nmaquinas=nmaq)

                if best_result:
                    final_results.append(best_result)
    else:
        with MPIPoolExecutor() as executor:
            futures = executor.starmap(
                choose_capacity,
                ((dataset, nmaq) for dataset in INSTANCES for nmaq in MAQUINAS),
            )
            final_results.append(futures)
            executor.shutdown(wait=True)

    if isinstance(final_results[0], list) or isinstance(final_results[0], types.GeneratorType):
        df_results_optimized = pd.DataFrame(list(chain.from_iterable(final_results)))
    else:
        df_results_optimized = pd.DataFrame(final_results)
    df_results_optimized.to_excel(CAPACIDADES_PATH, index=False)
    print("Processamento de capacidades concluído.")
    return df_results_optimized


def running_all_instance_with_chosen_capacity():
    final_results = []

    pdf_capacidades = pd.read_excel(CAPACIDADES_PATH, engine="openpyxl")
    caps = pd.pivot_table(
        pdf_capacidades, index=["Instance", "nmaquinas"], aggfunc={"capacity": "mean"}
    ).T.to_dict()

    if not MPI_BOOL:
        for dataset in INSTANCES:
            for nmaq in MAQUINAS:
                if caps.get((dataset, nmaq), None) == None:
                    print(f"Instance = {dataset} nmaquinas = {nmaq} not found")
                    continue
                else:
                    cap = caps.get((dataset, nmaq), None)["capacity"]

                best_result = solve_optimized_model(
                    dataset, capacity=cap[0], nmaquinas=nmaq,
                )

                if best_result:
                    final_results.append(best_result)
    else:
        with MPIPoolExecutor() as executor:
            futures = executor.starmap(
                solve_optimized_model,
                (
                    (dataset, caps.get((dataset, nmaq), None)["capacity"], nmaq)
                    for dataset in INSTANCES
                    for nmaq in MAQUINAS
                ),
            )
            final_results.append(futures)
            executor.shutdown(wait=True)

    df_results_optimized = pd.DataFrame(final_results)
    df_results_optimized.to_excel(OTIMIZADOS_PATH, index=False)
    pass


if __name__ == "__main__":
    running_all_instance_choose_capacity()
    # running_all_instance_with_chosen_capacity()
