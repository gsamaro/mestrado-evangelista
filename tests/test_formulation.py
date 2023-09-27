import pytest

import src.first_formulation as first_formulation
import src.zero_formulation as zero_formulation
import src.second_formulation as second_formulation
from src.read_file import dataCS


@pytest.fixture
def data() -> dataCS:
    return dataCS("F1.dat", r=1)


def test_formulation_one_and_zero(data):
    mdl0, _ = zero_formulation.build_model(data, data.cap)
    mdl1, _ = first_formulation.build_model(data, data.cap)

    mdl0.solve()
    fob0 = mdl0.solution.objective_value

    mdl1.solve()
    fob1 = mdl1.solution.objective_value

    assert fob0 >= fob1

def test_formulation_zero_and_second(data):
    mdl0, _ = zero_formulation.build_model(data, data.cap)
    mdl2, _ = second_formulation.build_model(data, data.cap)

    mdl0.solve()
    fob0 = mdl0.solution.objective_value

    mdl2.solve()
    fob2 = mdl2.solution.objective_value

    assert fob0 >= fob2

def test_formulation_one_and_two(data):
    mdl1, _ = first_formulation.build_model(data, data.cap)
    mdl2, _ = second_formulation.build_model(data, data.cap)

    mdl1.solve()
    fob1 = mdl1.solution.objective_value


    mdl2.solve()
    fob2 = mdl2.solution.objective_value

    assert fob1 == fob2