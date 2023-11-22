from dataclasses import dataclass, field
import yaml

@dataclass
class ProjectContext:
    custo_estoque: float = field(init=False, default=1)
    multiplicador_capacidade: float = field(init=False, default=1)
    custo_setup: float = field(init=False, default=1)
    experiment_name: str
    experiment_id: str    

    def __init__(self, experiment_name: str, experiment_id: int):
        self.experiment_id = experiment_id        
        with open(experiment_name,"r") as file:
            config = yaml.safe_load(file)
            for chave,valor in config["padrao"].items():
                self.__setattr__(chave, float(valor))