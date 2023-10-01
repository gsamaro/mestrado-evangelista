from typing import Dict, List

from docplex.mp.model import Model

from read_file import dataCS


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
