from utils import (
    running_all_instance_choose_capacity,
    running_all_instance_with_chosen_capacity,
)
from context import ProjectContext
from zero_formulation import build_model as classical_formulation_build_model
from first_formulation import build_model as first_reformulation_build_model
from second_formulation import build_model as second_reformulation_build_model

if __name__ == "__main__":
    for num in [1, 2, 3, 4, 5, 6]:
        context = ProjectContext(f"src/experimentos/experimento{num}.yml", num)
        running_all_instance_choose_capacity(
            context,
            classical_formulation_build_model,
        )
        running_all_instance_with_chosen_capacity(
            context,
            classical_formulation_build_model,
            path_to_save=f"otimizados_0_experiment_{num}.xlsx",
            env_formulation="0_ref",
        )
        running_all_instance_with_chosen_capacity(
            context,
            first_reformulation_build_model,
            path_to_save=f"otimizados_1_experiment_{num}.xlsx",
            env_formulation="1_ref",
        )
        running_all_instance_with_chosen_capacity(
            context,
            second_reformulation_build_model,
            path_to_save=f"otimizados_2_experiment_{num}.xlsx",
            env_formulation="2_ref",
        )
