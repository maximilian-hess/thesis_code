import json
import math
import os
from glob import glob
from typing import Dict, List, Tuple, Optional

from ortools.sat.python import cp_model


Coord = Tuple[float, float]
Coords = List[Coord]
DistanceMatrix = List[List[int]]


def dump_opt_values_to_json(opt_values: Dict[int, Dict[int, int]], output_path: str) -> None:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(opt_values, f)


def read_coords_file(file_path: str) -> Coords:
	with open(file_path, "r", encoding="utf-8") as f:
		lines = [line.strip() for line in f if line.strip()]
	n = int(lines[0])
	coords: Coords = []
	for line in lines[1:]:
		x_str, y_str = line.split()
		coords.append((float(x_str), float(y_str)))
	if len(coords) != n:
		raise ValueError(f"Expected {n} coords in {file_path}, got {len(coords)}")
	return coords


def build_distance_matrix(coords: Coords) -> DistanceMatrix:
	n = len(coords)
	matrix: DistanceMatrix = [[0] * n for _ in range(n)]
	for i in range(n):
		xi, yi = coords[i]
		for j in range(i + 1, n):
			xj, yj = coords[j]
			d = int(round(math.hypot(xi - xj, yi - yj)))
			matrix[i][j] = d
			matrix[j][i] = d
	return matrix


def solve_tsp_optimal(distance_matrix: DistanceMatrix, *, time_limit_s: Optional[float] = None) -> int:
	n = len(distance_matrix)
	model = cp_model.CpModel()
	x = {}
	arcs = []

	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			var = model.NewBoolVar(f"x_{i}_{j}")
			x[i, j] = var
			arcs.append([i, j, var])

	model.AddCircuit(arcs)
	model.Minimize(sum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))

	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 8
	if time_limit_s is not None:
		solver.parameters.max_time_in_seconds = time_limit_s

	status = solver.Solve(model)
	if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
		raise RuntimeError("No TSP solution found")
	return int(solver.ObjectiveValue())


def parse_instance_name(file_name: str) -> Tuple[int, int]:
	base = os.path.splitext(file_name)[0]
	parts = base.split("_")
	if len(parts) != 3 or parts[0] != "instance":
		raise ValueError(f"Unexpected instance filename: {file_name}")
	return int(parts[1]), int(parts[2])


def solve_instances_in_folder(folder_path: str) -> Dict[int, Dict[int, int]]:
	opt_values: Dict[int, Dict[int, int]] = {}
	for file_path in sorted(glob(os.path.join(folder_path, "*.tsp"))):
		file_name = os.path.basename(file_path)
		n, i = parse_instance_name(file_name)
		coords = read_coords_file(file_path)
		distance_matrix = build_distance_matrix(coords)
		opt_values.setdefault(n, {})[i] = solve_tsp_optimal(distance_matrix)
	return opt_values


def main() -> None:
	data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
	instance_coords_folder = os.path.join(data_folder, "instance_coords")
	opt_values = solve_instances_in_folder(instance_coords_folder)
	dump_opt_values_to_json(opt_values, os.path.join(data_folder, "tsp_opt_values.json"))

	for n in sorted(opt_values.keys()):
		for i in sorted(opt_values[n].keys()):
			print(f"instance_{n}_{i}.tsp: {opt_values[n][i]}")


if __name__ == "__main__":
	main()
