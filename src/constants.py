from pathlib import Path

INSTANCES = [f"F{i}.DAT" for i in range(1, 71)] + [f"G{i}.DAT" for i in range(1, 76)]
MAQUINAS = [2, 4, 8]
FAST_TIMELIMIT = 10
TIMELIMIT = 3600

CAPACIDADES_PATH = Path.resolve(Path.cwd() / "resultados" / "capacidades.xlsx")