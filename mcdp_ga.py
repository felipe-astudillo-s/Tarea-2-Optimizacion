"""
mcdp_ga.py
Genetic Algorithm for the Manufacturing Cell Design Problem (MCDP).

Objective: maximize Grouping Efficacy (GE)
    GE = e_in / (e_in + e_out + e_voids)
       = e_in / (total_ones + e_voids)

Solution representation: vector of length M (machines) where each element
is the cell index assigned to that machine (0 to C-1). Parts are assigned
greedily after the machine assignment is fixed.
"""

import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

class MCDPInstance:
    """Stores a parsed MCDP instance."""

    def __init__(self, matrix: np.ndarray, C: int, max_m: int):
        self.matrix = matrix.astype(int)
        # Standard MCDP incidence matrix: rows = machines, columns = parts
        self.n_machines, self.n_parts = matrix.shape
        self.C = C
        self.max_m = max_m
        self.total_ones = int(matrix.sum())

    @classmethod
    def from_block(cls, text: str) -> "MCDPInstance":
        """Parse a text block: matrix rows (CSV) + final line 'C,MaxM'."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        C, max_m = [int(x) for x in lines[-1].split(",")]
        rows = [[int(x) for x in ln.split(",")] for ln in lines[:-1]]
        return cls(np.array(rows, dtype=int), C, max_m)

    def __repr__(self) -> str:
        return (
            f"MCDPInstance(machines={self.n_machines}, parts={self.n_parts}, "
            f"C={self.C}, max_m={self.max_m}, total_ones={self.total_ones})"
        )


def parse_instance_file(filepath: str) -> list:
    """Read the instance file and return a list of MCDPInstance objects.

    Instances are separated by blank lines. Lines that do not start with a
    digit or comma are treated as comments and ignored.
    """
    with open(filepath, "r") as f:
        content = f.read()

    instances = []
    for block in content.split("\n\n"):
        data_lines = [
            ln for ln in block.splitlines()
            if ln.strip() and (ln.strip()[0].isdigit() or ln.strip()[0] == ",")
        ]
        if len(data_lines) >= 2:
            instances.append(MCDPInstance.from_block("\n".join(data_lines)))
    return instances


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

class Individual:
    """Candidate solution: machine-to-cell assignment vector."""

    def __init__(self, machine_assignment: np.ndarray):
        self.machine_assignment: np.ndarray = machine_assignment.copy()
        self.part_assignment: Optional[np.ndarray] = None
        self.fitness: Optional[float] = None

    def copy(self) -> "Individual":
        ind = Individual(self.machine_assignment)
        if self.part_assignment is not None:
            ind.part_assignment = self.part_assignment.copy()
        ind.fitness = self.fitness
        return ind

    def invalidate_fitness(self):
        self.part_assignment = None
        self.fitness = None


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """Genetic Algorithm for MCDP.

    Parameters
    ----------
    pop_size : int
        Population size.
    max_generations : int
        Maximum number of generations (stop if reached first).
    max_time_seconds : float
        Maximum execution wall-clock time in seconds (stop if reached first).
    tournament_size : int
        Number of individuals per tournament in selection.
    crossover_prob : float
        Probability of applying crossover to a pair of parents.
    mutation_prob : float
        Probability of applying mutation to an offspring.
    elite_count : int
        Number of best individuals copied unchanged to the next generation.
    crossover_type : str
        "two_point" or "one_point".
    random_seed : int or None
        Seed for the internal RNG. None means non-deterministic.
    """

    def __init__(
        self,
        pop_size: int = 50,
        max_generations: int = 500,
        max_time_seconds: float = 120.0,
        tournament_size: int = 3,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        elite_count: int = 2,
        crossover_type: str = "two_point",
        random_seed: Optional[int] = None,
    ):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_time_seconds = max_time_seconds
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        self.crossover_type = crossover_type
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _greedy_assign_parts(
        self, machine_assignment: np.ndarray, instance: MCDPInstance
    ) -> np.ndarray:
        """Assign each part to the cell containing most of its required machines.

        Matrix convention: rows = machines, columns = parts.
        For each part p (column p), we look at which machines have a 1 in
        that column and count how many of those machines are in each cell.

        Tiebreak: cell with fewest parts already assigned (load balance).
        Parts that require no machines are assigned to the least loaded cell.
        """
        C = instance.C
        part_assignment = np.empty(instance.n_parts, dtype=int)
        part_counts = np.zeros(C, dtype=int)

        for p in range(instance.n_parts):
            required = instance.matrix[:, p]  # machines needed by part p
            if required.sum() == 0:
                best = int(np.argmin(part_counts))
            else:
                scores = np.array(
                    [int(required[machine_assignment == c].sum()) for c in range(C)]
                )
                max_score = scores.max()
                candidates = np.where(scores == max_score)[0]
                best = int(candidates[np.argmin(part_counts[candidates])])
            part_assignment[p] = best
            part_counts[best] += 1

        return part_assignment

    def _compute_fitness(
        self, individual: Individual, instance: MCDPInstance
    ) -> float:
        """Compute GE and cache part_assignment on the individual."""
        if individual.fitness is not None:
            return individual.fitness

        ma = individual.machine_assignment
        pa = self._greedy_assign_parts(ma, instance)
        individual.part_assignment = pa

        e_in = 0
        e_voids = 0
        for c in range(instance.C):
            machine_idx = np.where(ma == c)[0]  # row indices (machines)
            part_idx = np.where(pa == c)[0]      # col indices (parts)
            if len(machine_idx) > 0 and len(part_idx) > 0:
                block = instance.matrix[np.ix_(machine_idx, part_idx)]
                e_in += int(block.sum())
                e_voids += int((block == 0).sum())

        denom = instance.total_ones + e_voids
        individual.fitness = e_in / denom if denom > 0 else 0.0
        return individual.fitness

    def _evaluate_population(
        self, population: list, instance: MCDPInstance
    ):
        for ind in population:
            self._compute_fitness(ind, instance)

    # ------------------------------------------------------------------
    # Repair: enforce MaxM per cell
    # ------------------------------------------------------------------

    def _repair(
        self, assignment: np.ndarray, instance: MCDPInstance
    ) -> np.ndarray:
        """Redistribute machines from overcrowded cells to cells with capacity.

        All instances satisfy C * max_m >= n_machines, so this always converges.
        The inner break is a safety guard for unexpected edge cases.
        """
        assignment = assignment.copy()
        C, max_m = instance.C, instance.max_m
        counts = np.array([(assignment == c).sum() for c in range(C)], dtype=int)

        changed = True
        while changed:
            changed = False
            for c in range(C):
                while counts[c] > max_m:
                    machines_in_c = np.where(assignment == c)[0]
                    machine_to_move = int(self._rng.choice(machines_in_c))
                    available = [k for k in range(C) if k != c and counts[k] < max_m]
                    if not available:
                        break  # constraint cannot be satisfied further; stop
                    target = int(self._rng.choice(available))
                    assignment[machine_to_move] = target
                    counts[c] -= 1
                    counts[target] += 1
                    changed = True

        return assignment

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_individual(self, instance: MCDPInstance) -> Individual:
        assignment = self._rng.integers(0, instance.C, size=instance.n_machines)
        assignment = self._repair(assignment, instance)
        return Individual(assignment)

    def _initialize_population(self, instance: MCDPInstance) -> list:
        return [self._initialize_individual(instance) for _ in range(self.pop_size)]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self, population: list) -> Individual:
        k = min(self.tournament_size, len(population))
        idx = self._rng.choice(len(population), size=k, replace=False)
        best_idx = max(idx, key=lambda i: population[i].fitness)
        return population[best_idx]

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(
        self,
        p1: Individual,
        p2: Individual,
        instance: MCDPInstance,
    ):
        """Two-point (or one-point) crossover followed by repair."""
        if self._rng.random() > self.crossover_prob:
            return p1.copy(), p2.copy()

        M = instance.n_machines
        a1, a2 = p1.machine_assignment.copy(), p2.machine_assignment.copy()

        if self.crossover_type == "two_point":
            pts = sorted(
                self._rng.choice(range(1, M), size=2, replace=False).tolist()
            )
            p, q = pts[0], pts[1]
            c1 = np.concatenate([a1[:p], a2[p:q], a1[q:]])
            c2 = np.concatenate([a2[:p], a1[p:q], a2[q:]])
        else:  # one_point
            p = int(self._rng.integers(1, M))
            c1 = np.concatenate([a1[:p], a2[p:]])
            c2 = np.concatenate([a2[:p], a1[p:]])

        c1 = self._repair(c1, instance)
        c2 = self._repair(c2, instance)
        return Individual(c1), Individual(c2)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, individual: Individual, instance: MCDPInstance) -> Individual:
        """Swap cell assignments of two machines from different cells.

        A same-cell swap is a no-op, so we try a few times to find a valid
        pair. Swapping two machines is always feasible (counts unchanged).
        """
        if self._rng.random() > self.mutation_prob:
            return individual.copy()

        assignment = individual.machine_assignment.copy()
        for _ in range(20):
            j1, j2 = self._rng.choice(instance.n_machines, size=2, replace=False)
            if assignment[j1] != assignment[j2]:
                assignment[j1], assignment[j2] = int(assignment[j2]), int(assignment[j1])
                return Individual(assignment)

        return individual.copy()

    # ------------------------------------------------------------------
    # Main GA loop
    # ------------------------------------------------------------------

    def run(self, instance: MCDPInstance) -> dict:
        """Execute the GA on the given instance.

        Returns
        -------
        dict
            best_fitness, best_machine_assignment, best_part_assignment,
            generations_run, time_elapsed, convergence_curve
        """
        self._rng = np.random.default_rng(self.random_seed)

        population = self._initialize_population(instance)
        self._evaluate_population(population, instance)

        best = max(population, key=lambda x: x.fitness).copy()
        convergence_curve = [best.fitness]

        t_start = time.time()
        gen = 0

        while gen < self.max_generations:
            if time.time() - t_start >= self.max_time_seconds:
                break

            # Elitism: preserve best individuals unchanged
            population.sort(key=lambda x: x.fitness, reverse=True)
            elites = [ind.copy() for ind in population[: self.elite_count]]

            # Build next generation
            new_pop: list = list(elites)
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                c1, c2 = self._crossover(p1, p2, instance)
                c1 = self._mutate(c1, instance)
                c2 = self._mutate(c2, instance)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            # Evaluate only newly created individuals
            for ind in new_pop[self.elite_count :]:
                self._compute_fitness(ind, instance)

            population = new_pop

            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best.fitness:
                best = current_best.copy()

            convergence_curve.append(best.fitness)
            gen += 1

        return {
            "best_fitness": best.fitness,
            "best_machine_assignment": best.machine_assignment,
            "best_part_assignment": best.part_assignment,
            "generations_run": gen,
            "time_elapsed": time.time() - t_start,
            "convergence_curve": convergence_curve,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Runs multiple independent GA executions and aggregates statistics.

    Parameters
    ----------
    ga_config : dict
        Keyword arguments for GeneticAlgorithm. If 'random_seed' is set,
        each run derives its seed as base_seed + run_index for reproducibility.
    n_runs : int
        Number of independent runs per instance.
    """

    def __init__(self, ga_config: dict, n_runs: int = 10):
        self.ga_config = ga_config
        self.n_runs = n_runs

    def run_experiment(self, instance: MCDPInstance, instance_name: str = "") -> dict:
        """Run n_runs GA executions on one instance."""
        fitness_values = []
        convergence_curves = []
        best_result: Optional[dict] = None

        for run_idx in range(self.n_runs):
            config = dict(self.ga_config)
            base_seed = config.get("random_seed", None)
            config["random_seed"] = (base_seed + run_idx) if base_seed is not None else run_idx

            ga = GeneticAlgorithm(**config)
            result = ga.run(instance)

            fitness_values.append(result["best_fitness"])
            convergence_curves.append(result["convergence_curve"])

            if best_result is None or result["best_fitness"] > best_result["best_fitness"]:
                best_result = result

            print(
                f"  [{instance_name}] Run {run_idx + 1:2d}/{self.n_runs}: "
                f"GE = {result['best_fitness']:.4f}  "
                f"({result['generations_run']} gen, {result['time_elapsed']:.1f}s)"
            )

        arr = np.array(fitness_values)
        return {
            "instance": instance_name,
            "fitness_values": fitness_values,
            "best_fitness": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "best_solution": best_result,
            "convergence_curves": convergence_curves,
        }

    def run_all(self, instances: dict) -> dict:
        """Run experiments for all instances.

        Parameters
        ----------
        instances : dict[str, MCDPInstance]
        """
        results = {}
        for name, inst in instances.items():
            print(f"\n{'='*60}")
            print(f"Instancia: {name}")
            print(f"  {inst.n_parts} partes x {inst.n_machines} máquinas | "
                  f"C={inst.C}, MaxM={inst.max_m}")
            print(f"{'='*60}")
            results[name] = self.run_experiment(inst, name)
            print(
                f"\n  -> Media GE: {results[name]['mean']:.4f} "
                f"± {results[name]['std']:.4f}  "
                f"(mejor: {results[name]['best_fitness']:.4f})"
            )
        return results


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Instancias", "instanciasMCDP.txt"
    )
    instances = parse_instance_file(filepath)

    print(f"Instancias cargadas: {len(instances)}")
    for i, inst in enumerate(instances, 1):
        print(f"  Instancia {i}: {inst}")

    print("\nPrueba rápida en Instancia 1 (50 generaciones, seed=42)...")
    ga = GeneticAlgorithm(
        pop_size=30,
        max_generations=50,
        max_time_seconds=30.0,
        random_seed=42,
    )
    result = ga.run(instances[0])
    print(f"  Mejor GE           : {result['best_fitness']:.4f}")
    print(f"  Generaciones       : {result['generations_run']}")
    print(f"  Tiempo             : {result['time_elapsed']:.2f}s")
    print(f"  Asignación máquinas: {result['best_machine_assignment']}")
    print(f"  Asignación partes  : {result['best_part_assignment']}")
