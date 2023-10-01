from utils import (
    running_all_instance_choose_capacity,
    running_all_instance_with_chosen_capacity,
)
from zero_formulation import build_model as classical_formulation_build_model
from first_formulation import build_model as first_reformulation_build_model
from second_formulation import build_model as second_reformulation_build_model

if __name__ == "__main__":
    running_all_instance_choose_capacity(
        classical_formulation_build_model, env_formulation="Classical formulation"
    )
    running_all_instance_with_chosen_capacity(
        first_reformulation_build_model,
        path_to_save="otimizados1.xlsx",
        env_formulation="1st ref",
    )
    running_all_instance_with_chosen_capacity(
        second_reformulation_build_model,
        path_to_save="otimizados2.xlsx",
        env_formulation="2nd ref",
    )
