from utils import (
    running_all_instance_choose_capacity,
    running_all_instance_with_chosen_capacity,
    read_experiments,
)
from zero_formulation import build_model as classical_formulation_build_model
from first_formulation import build_model as first_reformulation_build_model
from second_formulation import build_model as second_reformulation_build_model

if __name__ == "__main__":
    for num in [2,3,4,5,6]:
        read_experiments(f"src/experimentos/experimento{num}.yml")
        running_all_instance_choose_capacity(
            classical_formulation_build_model
        )
        running_all_instance_with_chosen_capacity(
            classical_formulation_build_model,
            path_to_save="otimizados_classical.xlsx",
            env_formulation="classical_form",
        )
        running_all_instance_with_chosen_capacity(
            first_reformulation_build_model,
            path_to_save="otimizados1.xlsx",
            env_formulation="1st_ref",
        )
        running_all_instance_with_chosen_capacity(
            second_reformulation_build_model,
            path_to_save="otimizados2.xlsx",
            env_formulation="2nd_ref",
        )
