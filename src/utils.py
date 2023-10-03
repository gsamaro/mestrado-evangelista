import os
from itertools import chain
from typing import Dict, List
import types

import numpy as np
import pandas as pd


from pathlib import Path
from read_file import dataCS

try:
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py import MPI

    MPI_BOOL = True
except:
    print("mpi4py not running")
    MPI_BOOL = False

import constants


def print_info(data: dataCS, status: str) -> None:
    if MPI_BOOL:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = None
    print(
        f"Instance = {data.instance} Cap = {data.cap[0]} nmaquinas = {data.r} {status} Process {rank}"
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


def choose_capacity(
    dataset: str, build_model, nmaquinas: int = 2, get_closest: bool = True
) -> pd.DataFrame:
    data = dataCS(dataset, r=nmaquinas)
    original_capacity = data.cap[0] / data.r
    instance_results = []

    for cap in np.linspace(
        original_capacity, original_capacity * 2, num=5, endpoint=True
    ):
        print_info(data, "building")
        mdl, data = build_model(data, np.ceil(cap))
        mdl.parameters.timelimit = constants.FAST_TIMELIMIT
        result = mdl.solve()
        print_info(data, "solver finished")

        if result == None:
            print_info(data, "infactível")
            continue

        kpis = mdl.kpis_as_dict(result, objective_key="objective_function")
        kpis = add_new_kpi(kpis, result, data)

        assert kpis["utilization_capacity"] <= 100, "Capacidade > 100%"

        instance_results.append(kpis)
        print_info(data, "concluído")
    if get_closest:
        if len(instance_results) > 0:
            return pd.DataFrame([closest_to_75_percent(instance_results)])
        else:
            return pd.DataFrame([])
    else:
        return pd.DataFrame(instance_results)


def running_all_instance_choose_capacity(build_model, env_formulation) -> pd.DataFrame:
    # Executando e coletando os resultados
    final_results = []

    if not MPI_BOOL:
        for dataset in constants.INSTANCES:
            for nmaq in constants.MAQUINAS:
                best_result = choose_capacity(dataset, build_model, nmaquinas=nmaq)

                if best_result:
                    final_results.append(best_result)
    else:
        with MPIPoolExecutor() as executor:
            futures = executor.starmap(
                choose_capacity,
                (
                    (dataset, build_model, nmaq)
                    for dataset in constants.INSTANCES
                    for nmaq in constants.MAQUINAS
                ),
            )
            final_results.append(futures)
            executor.shutdown(wait=True)
    
    if len(final_results) > 0:        
        df_results_optimized = pd.concat([list(f)[0] for f in final_results], axis=0)                
        df_results_optimized.to_excel(constants.CAPACIDADES_PATH, index=False)
        print("Processamento de capacidades concluído.")
        return df_results_optimized
    else:
        print("Final results vazio.")
        return None


def solve_optimized_model(
    dataset: str, build_model, capacity: float, nmaquinas: int = 8
) -> pd.DataFrame:
    data = dataCS(dataset, r=nmaquinas)
    mdl, data = build_model(data, capacity)
    mdl.parameters.timelimit = constants.TIMELIMIT
    result = mdl.solve()

    if result == None:
        print_info(data, "infactível")
        return pd.DataFrame([])

    kpis = mdl.kpis_as_dict(result, objective_key="objective_function")
    kpis = add_new_kpi(kpis, result, data)

    # Cálculo da relaxação linear
    relaxed_model = mdl.clone()
    status = relaxed_model.solve(url=None, key=None, log_output=False)

    relaxed_objective_value = relaxed_model.objective_value
    kpis["Relaxed Objective Value"] = relaxed_objective_value

    print_info(data, "concluído")

    return pd.DataFrame([kpis])


def running_all_instance_with_chosen_capacity(
    build_model, path_to_save: str, env_formulation: str
):
    final_results = []

    complete_path_to_save = Path.resolve(Path.cwd() / "resultados" / path_to_save)

    pdf_capacidades = pd.read_excel(constants.CAPACIDADES_PATH, engine="openpyxl")
    caps = pd.pivot_table(
        pdf_capacidades, index=["Instance", "nmaquinas"], aggfunc={"capacity": "mean"}
    ).T.to_dict()

    if not MPI_BOOL:
        for dataset in constants.INSTANCES:
            for nmaq in constants.MAQUINAS:
                if caps.get((dataset, nmaq), None) == None:
                    print(f"Instance = {dataset} nmaquinas = {nmaq} not found")
                    continue
                else:
                    cap = caps.get((dataset, nmaq), None)["capacity"]

                best_result = solve_optimized_model(
                    dataset,
                    build_model,
                    capacity=cap,
                    nmaquinas=nmaq,
                )

                if best_result:
                    final_results.append(best_result)
    else:
        with MPIPoolExecutor() as executor:
            futures = executor.starmap(
                solve_optimized_model,
                (
                    (
                        dataset,
                        build_model,
                        caps.get((dataset, nmaq), None).get("capacity", 0),
                        nmaq,
                    )
                    for dataset in constants.INSTANCES
                    for nmaq in constants.MAQUINAS
                ),
            )
            final_results.append(futures)
            executor.shutdown(wait=True)
    
    df_results_optimized = pd.concat([list(f)[0] for f in final_results], axis=0)                
    df_results_optimized.to_excel(complete_path_to_save, index=False)
