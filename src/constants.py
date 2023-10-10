from pathlib import Path

INSTANCES = [f"F{i}.DAT" for i in range(1, 71)] + [f"G{i}.DAT" for i in range(1, 76)]
MAQUINAS = [2, 4, 8]
NUM_POINTS = 7
FAST_TIMELIMIT = 20
TIMELIMIT = 3600

CAPACIDADES_PATH = Path.resolve(Path.cwd() / "resultados" / "capacidades.xlsx")
RESULTADOS_INDIVIDUAIS_PATH = Path.resolve(Path.cwd() / "resultados" / "individuais")
OTIMIZADOS_INDIVIDUAIS_PATH = Path.resolve(Path.cwd() / "resultados" / "otimizados")
FINAL_PATH = Path.resolve(Path.cwd() / "resultados")

IDEAL_CAPACITY = 75
