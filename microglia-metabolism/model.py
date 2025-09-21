# model.py
from __future__ import annotations
import math
from typing import Tuple, List, Dict
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agents import Microglia, HOMEOSTATIC, M1


class MicrogliaMetabolismModel(Model):
    """
    A faithful port of the NetLogo 'microglia-metabolism' model:
      - Two microglia phenotypes (homeostatic / M1)
      - Patch fields: lactate trail (transient/permanent), plaque presence, BBB integrity
      - Metabolism: oxphos/glycolysis with glucose pool; lactate trails from glycolysis
      - pH computed from permanent lactate; plaque diffusion is pH- and exercise-dependent
      - Exercise & booster regimens feed into plaque-eat-scalar, exercise-factor, booster-frequency
      - Tolerization near plaques with cooldown
    Tick = 1 hour. Stop condition in app.py defaults to 8760 ticks (1 year).
    """

    def __init__(
        self,
        width: int = 51,
        height: int = 51,
        torus: bool = True,
        # Initialization sliders (mirroring NetLogo)
        init_microglia: int = 20,
        init_plaque: int = 200,         # number of plaque patches to seed at setup
        init_vessels: int = 60,         # number of vessel walkers (paint BBB integrity)
        # Probabilities
        eat_probability: float = 0.10,  # plaque/BBB eating base probability (per step)
        fortify_probability: float = 0.10,
        lactate_probability: float = 0.05,  # chance a new trail becomes permanent (white)
        # Metabolism / environment
        added_glucose: float = 500.0,   # glucose added each tick
        exercise: str = "none",         # "none" | "moderate" | "high"
        metabolic_booster: str = "off", # "daily" | "every 2 days" | "twice per week" | "weekly" | "off"
        seed: int | None = None,
    ):
        super().__init__(seed=seed)

        # Compatibility shim for older code paths: local unique-id counter
        self._uid = 0  # << ADDED

        self.grid = MultiGrid(width, height, torus=torus)
        self.width, self.height = width, height

        # --- Globals (NetLogo 'globals') ---
        self.global_glucose = added_glucose
        self.global_pH = 7.3
        self.global_integrity = 0.0
        self.lactate_num = 0
        self.plaque_spread_prob = 0.2
        self.booster_frequency = -1
        self.plaque_eat_scalar = 1.0
        self.exercise_factor = 0.0

        # Store parameters
        self.init_microglia = init_microglia
        self.init_plaque = init_plaque
        self.init_vessels = init_vessels
        self.eat_probability = float(eat_probability)
        self.fortify_probability = float(fortify_probability)
        self.lactate_probability = float(lactate_probability)
        self.added_glucose = float(added_glucose)
        self.exercise = exercise.lower().strip()
        self.metabolic_booster = metabolic_booster.lower().strip()

        # --- Patch fields (NetLogo patches-own) ---
        shp = (width, height)
        self.lactate_flag   = np.zeros(shp, dtype=bool)
        self.lactate_dismantle = np.zeros(shp, dtype=bool)
        self.lactate_val    = np.zeros(shp, dtype=int)
        self.curr_spread    = np.zeros(shp, dtype=int)
        self.max_spread     = np.zeros(shp, dtype=int)
        self.perm_lactate   = np.zeros(shp, dtype=bool)
        self.spread_life    = np.zeros(shp, dtype=int)
        self.integrity      = np.zeros(shp, dtype=float)  # 0 none; (0,1] vessel strength
        self.plaque_val     = np.zeros(shp, dtype=int)    # 1 for plaque present

        # --- Agents container (explicit list) ---
        self.microglia: list[Microglia] = []

        # --- Setup ---
        self._initialize_vessels(self.init_vessels)
        self._seed_plaques(self.init_plaque)
        self._spawn_microglia(self.init_microglia)

        # Booster schedule & exercise scaling
        self._configure_booster(self.metabolic_booster)
        self._configure_exercise(self.exercise)

        # Initialize pH-dependent plaque spread probability
        self._update_spread_prob()

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.steps,
                "global_glucose": lambda m: m.global_glucose,
                "global_pH": lambda m: m.global_pH,
                "global_integrity": lambda m: m._mean_integrity_percent(),
                "plaques": lambda m: int(np.sum(m.plaque_val == 1)),
                "perm_lactate_patches": lambda m: int(np.sum(m.perm_lactate)),
                "homeostatic": lambda m: sum(1 for a in m.microglia if a.phenotype == HOMEOSTATIC),
                "m1": lambda m: sum(1 for a in m.microglia if a.phenotype == M1),
            }
        )

    # ----- Unique-id shim (so existing code using self.next_id() keeps working) -----
    def next_id(self) -> int:
        self._uid += 1
        return self._uid

    # ----- Setup helpers -----
    def _initialize_vessels(self, walkers: int):
        for _ in range(walkers):
            side = self.random.randrange(4)
            if side == 0:
                x, y = 1, self.random.randrange(self.height)
            elif side == 1:
                x, y = self.width - 2, self.random.randrange(self.height)
            elif side == 2:
                x, y = self.random.randrange(self.width), 1
            else:
                x, y = self.random.randrange(self.width), self.height - 2
            steps = 0
            while 0 < x < self.width - 1 and 0 < y < self.height - 1 and steps < (self.width + self.height) * 2:
                self.integrity[x, y] = max(self.integrity[x, y], 0.5 + 0.5 * self.random.random())
                neigh = self.grid.get_neighborhood((x, y), moore=True, include_center=False, radius=1)
                if neigh:
                    x, y = self.random.choice(neigh)
                steps += 1

    def _seed_plaques(self, n: int):
        all_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        for x, y in self.random.sample(all_cells, k=min(n, len(all_cells))):
            self.plaque_val[x, y] = 1

    def _spawn_microglia(self, n: int):
        for _ in range(n):
            a = Microglia(self.next_id(), self, phenotype=HOMEOSTATIC)
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            self.grid.place_agent(a, (x, y))
            self.microglia.append(a)

    def _configure_booster(self, regimen: str):
        mapping = {
            "daily": 24,
            "every 2 days": 48,
            "twice per week": 84,
            "weekly": 168,
            "off": -1,
        }
        self.booster_frequency = mapping.get(regimen, -1)

    def _configure_exercise(self, level: str):
        if level == "high":
            self.plaque_eat_scalar = 0.10
            self.exercise_factor = 0.10
        elif level == "moderate":
            self.plaque_eat_scalar = 0.25
            self.exercise_factor = 0.05
        else:
            self.plaque_eat_scalar = 1.00
            self.exercise_factor = 0.00

    # ----- Plaque diffusion probability (pH & exercise) -----
    def _update_spread_prob(self):
        if self.global_pH > 7.0:
            p = 0.2
        elif self.global_pH < 6.2:
            p = 0.9
        else:
            pH = self.global_pH
            p = (-1.048951) * (pH ** 3) + 20.92657 * (pH ** 2) - 139.60227 * pH + 311.95485
        p -= self.exercise_factor
        self.plaque_spread_prob = float(np.clip(p, 0.0, 1.0))

    # ----- Lactate trail dynamics -----
    def _spread_trail_from(self, x: int, y: int):
        r = self.curr_spread[x, y]
        xs = range(max(0, x - r), min(self.width, x + r + 1))
        ys = range(max(0, y - r), min(self.height, y + r + 1))
        for xi in xs:
            for yi in ys:
                if abs(xi - x) + abs(yi - y) <= r:
                    self.lactate_val[xi, yi] += 1
                    self.lactate_dismantle[xi, yi] = True
        self.curr_spread[x, y] = r + 1
        if self.random.random() < self.lactate_probability:
            self.perm_lactate[x, y] = True
            self.lactate_flag[x, y] = False
            self.lactate_dismantle[x, y] = False

    def _reset_trail_at(self, x: int, y: int):
        self.lactate_val[x, y] = max(0, self.lactate_val[x, y] - 1)
        if self.lactate_val[x, y] == 0:
            self.spread_life[x, y] = 10
            self.lactate_dismantle[x, y] = False
            self.curr_spread[x, y] = 0
            self.max_spread[x, y] = 3
            self.lactate_flag[x, y] = False
            self.perm_lactate[x, y] = False

    # ----- Per-step environment updates -----
    def _diffuse_plaques(self):
        to_set = []
        for x in range(self.width):
            for y in range(self.height):
                if self.plaque_val[x, y] == 1:
                    for nx, ny in self.grid.get_neighborhood((x, y), moore=False, include_center=False, radius=1):
                        if self.random.random() < self.plaque_spread_prob:
                            to_set.append((nx, ny))
        for nx, ny in to_set:
            self.plaque_val[nx, ny] = 1

    def _update_lactate_system(self):
        active_origins = np.argwhere(np.logical_and(self.lactate_flag, self.curr_spread < np.maximum(1, self.max_spread)))
        for x, y in active_origins:
            self._spread_trail_from(int(x), int(y))
        dismantle_cells = np.argwhere(self.lactate_dismantle)
        for x, y in dismantle_cells:
            self.spread_life[x, y] -= 1
        expired = np.argwhere(self.spread_life <= 0)
        for x, y in expired:
            self._reset_trail_at(int(x), int(y))

    def _update_pH_and_prob(self):
        self.lactate_num = int(np.sum(self.perm_lactate))
        self.global_pH = 7.3 - 0.02 * self.lactate_num
        self._update_spread_prob()

    def _mean_integrity_percent(self) -> float:
        valid = self.integrity[self.integrity > 0.0]
        if valid.size == 0:
            return 0.0
        return 100.0 * float(valid.mean())

    # ----- Step -----
    def step(self):
        # 1) Agents act (random order)
        agents = list(self.microglia)
        self.random.shuffle(agents)
        for a in agents:
            a.step()

        # 2) Environment updates
        self._diffuse_plaques()
        self._update_lactate_system()
        self._update_pH_and_prob()

        # 3) Glucose replenishment
        self.global_glucose += self.added_glucose

        # 4) Booster dosing: reset tolerance on schedule
        if self.booster_frequency != -1 and (self.steps % self.booster_frequency == 0) and self.steps > 0:
            for a in agents:
                a.s.immune_tolerance = False
                a.s.tolerance_timer = 0

        # 5) Update global integrity monitor
        self.global_integrity = self._mean_integrity_percent()

        # 6) Collect data
        self.datacollector.collect(self)
