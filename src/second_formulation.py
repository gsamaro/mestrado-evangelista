from docplex.mp.model import Model

from read_file import dataCS


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
