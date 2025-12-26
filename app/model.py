from __future__ import annotations
from typing import Tuple, Set
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import io
from agents import Microglia, Neuron, Astrocyte


class MicrogliaNeuronModel(Model):
    def __init__(
        self,
        width: int = 33,
        height: int = 33,
        torus: bool = True,
        # microglia seeding
        init_m0_microglia: int = 5,
        init_m1_microglia: int = 5,
        init_m2_microglia: int = 5,
        # neuron seeding
        init_healthy_neuron: int = 10,
        init_damaged_neuron: int = 10,
        init_dead_neuron: int = 10,
        # astrocyte seeding (by phenotype)
        init_a0_astrocytes: int = 10,
        init_a1_astrocytes: int = 5,
        init_a2_astrocytes: int = 5,
        astro_coverage_radius: int = 2,
        # core microglia behavior
        eat_probability: float = 0.70,
        sensing_efficiency: float = 0.50,
        damage_chance: float = 0.005,
        neuron_distance: int = 5,
        damage_radius: int = 3,
        seed: int | None = None,
        # Neuron knobs
        damage_ratio_thresh: float = 0.6,
        healthy_ratio_thresh: float = 0.2,
        healthy_chance: float = 0.20,
        damage_to_death_ticks: int = 20,
        death_ratio_thresh: float = 0.9,
        death_chance: float = 0.25,
        # Microglia / Astrocyte phenotype signal thresholds (microglia)
        homeo_signal_thresh: float = 1.0,
        pro_inflam_signal_thresh: float = 0.2,
        anti_inflam_signal_thresh: float = 0.8,
        homeo_chance: float = 0.2,
        pro_inflam_chance: float = 0.2,
        anti_inflam_chance: float = 0.2,
        # Astrocyte-specific phenotype thresholds
        astro_homeo_signal_thresh: float = 0.5,
        astro_pro_inflam_signal_thresh: float = 0.15,
        astro_anti_inflam_signal_thresh: float = 0.6,
        astro_homeo_chance: float = 0.25,
        astro_pro_inflam_chance: float = 0.3,
        astro_anti_inflam_chance: float = 0.3,
        # signal decay
        pro_decay: float = 0.20,
        anti_decay: float = 0.10,
        # resolution times (separate for microglia vs astrocytes)
        m2_to_m0_resolution_ticks: int = 10,   # microglia M2 -> M0
        a2_to_a0_resolution_ticks: int = 15,   # astrocyte A2 -> A0
        # astrocyte-driven microglia recruitment (optional; used inside Astrocyte)
        recruitment_threshold: float = 3.0,
        microglia_recruitment_prob: float = 0.01,
        # ---- lipid dynamics ----
        neuron_ld_production_rate: float = 0.2,
        neuron_ld_damage_boost: float = 0.5,
        neuron_ld_death_boost: float = 1.0,
        neuron_ld_packet_size: float = 0.5,
        neuron_to_astro_ld_transfer_prob: float = 0.3,
        neuron_to_astro_ld_transfer_radius: int = 2,
        astro_ld_oxidation_rate: float = 0.05,
        astro_ld_to_anti_inflam: float = 0.05,
        microglia_ld_from_phagocytosis: float = 0.5,
        microglia_ld_pro_inflam_threshold: float = 5.0,
        microglia_ld_pro_inflam_boost: float = 0.5,
    ):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=torus)
        self.width, self.height = width, height

        # counters
        self._uid: int = 0
        self.steps: int = 0

        # params
        self.eat_probability = float(eat_probability)
        self.sensing_efficiency = float(sensing_efficiency)
        self.damage_chance = float(damage_chance)
        self.neuron_distance = int(neuron_distance)
        self.damage_radius = int(damage_radius)

        # neuron params
        self.damage_ratio_thresh = float(damage_ratio_thresh)
        self.healthy_ratio_thresh = float(healthy_ratio_thresh)
        self.healthy_chance = float(healthy_chance)
        self.damage_to_death_ticks = int(damage_to_death_ticks)
        self.death_ratio_thresh = float(death_ratio_thresh)
        self.death_chance = float(death_chance)

        # microglia phenotype signal params
        self.homeo_signal_thresh = float(homeo_signal_thresh)
        self.pro_inflam_signal_thresh = float(pro_inflam_signal_thresh)
        self.anti_inflam_signal_thresh = float(anti_inflam_signal_thresh)
        self.homeo_chance = float(homeo_chance)
        self.pro_inflam_chance = float(pro_inflam_chance)
        self.anti_inflam_chance = float(anti_inflam_chance)

        # astrocyte phenotype signal params
        self.astro_homeo_signal_thresh = float(astro_homeo_signal_thresh)
        self.astro_pro_inflam_signal_thresh = float(astro_pro_inflam_signal_thresh)
        self.astro_anti_inflam_signal_thresh = float(astro_anti_inflam_signal_thresh)
        self.astro_homeo_chance = float(astro_homeo_chance)
        self.astro_pro_inflam_chance = float(astro_pro_inflam_chance)
        self.astro_anti_inflam_chance = float(astro_anti_inflam_chance)

        # signal decay params
        self.pro_decay = float(pro_decay)
        self.anti_decay = float(anti_decay)

        # resolution times
        self.m2_to_m0_resolution_ticks = int(m2_to_m0_resolution_ticks)  # microglia
        self.a2_to_a0_resolution_ticks = int(a2_to_a0_resolution_ticks)  # astrocytes

        # astrocyte params
        self.astro_coverage_radius = int(astro_coverage_radius)
        self.recruitment_threshold = float(recruitment_threshold)
        self.microglia_recruitment_prob = float(microglia_recruitment_prob)
        
        # lipid dynamics (tunable)
        self.neuron_ld_production_rate = float(neuron_ld_production_rate)
        self.neuron_ld_damage_boost = float(neuron_ld_damage_boost)
        self.neuron_ld_death_boost = float(neuron_ld_death_boost)
        self.neuron_ld_packet_size = float(neuron_ld_packet_size)
        self.neuron_to_astro_ld_transfer_prob = float(neuron_to_astro_ld_transfer_prob)
        self.neuron_to_astro_ld_transfer_radius = int(neuron_to_astro_ld_transfer_radius)

        self.astro_ld_oxidation_rate = float(astro_ld_oxidation_rate)
        self.astro_ld_to_anti_inflam = float(astro_ld_to_anti_inflam)

        self.microglia_ld_from_phagocytosis = float(microglia_ld_from_phagocytosis)
        self.microglia_ld_pro_inflam_threshold = float(microglia_ld_pro_inflam_threshold)
        self.microglia_ld_pro_inflam_boost = float(microglia_ld_pro_inflam_boost)
        
        self.microglia_ld_oxidation_rate: float = 0.02   # fraction burned per tick
        self.microglia_ld_to_pro_inflam: float = 0.05    # pro signal per LD used (M1)
        self.microglia_ld_to_anti_inflam: float = 0.05   # anti signal per LD used (M0/M2)

        self.microglia_ld_max: float = 50.0 
        self.neuron_ld_max: float = 10.0


        # patch fields
        shp = (width, height)
        # neuron-sourced damage field for chemotaxis
        self.damage_val = np.zeros(shp, dtype=float)
        # cytokine fields
        self.pro_inflam_val = np.zeros(shp, dtype=float)
        self.anti_inflam_val = np.zeros(shp, dtype=float)

        self.residence = np.zeros(shp, dtype=bool)
        self.res_curr = np.zeros(shp, dtype=int)
        self.dismantling = np.zeros(shp, dtype=bool)
        self.dis_curr = np.zeros(shp, dtype=int)

        # agents
        self.microglia: list[Microglia] = []
        self.neurons: list[Neuron] = []
        self.astrocytes: list[Astrocyte] = []

        # -------- neurons --------
        for _ in range(init_healthy_neuron):
            n = Neuron(self.next_id(), self, damaged=False, dead=False)
            self._place_random(n)
            self.neurons.append(n)

        for _ in range(init_damaged_neuron):
            n = Neuron(self.next_id(), self, damaged=True, dead=False)
            self._place_random(n)
            self.neurons.append(n)
            x, y = n.pos
            self.residence[x, y] = True
            self.res_curr[x, y] = 1
            self.pro_inflam_val[x, y] = max(self.pro_inflam_val[x, y], 1.0)
            self.damage_val[x, y] = max(self.damage_val[x, y], 1.0)

        for _ in range(init_dead_neuron):
            n = Neuron(self.next_id(), self, damaged=False, dead=True)
            self._place_random(n)
            self.neurons.append(n)
            x, y = n.pos
            self.residence[x, y] = True
            self.res_curr[x, y] = 1
            self.pro_inflam_val[x, y] = max(self.pro_inflam_val[x, y], 1.0)
            self.damage_val[x, y] = max(self.damage_val[x, y], 1.5)

        # -------- microglia phenotypes --------
        for _ in range(init_m0_microglia):
            m = Microglia(self.next_id(), self, phenotype="M0")
            self._place_random(m)
            self.microglia.append(m)
        for _ in range(init_m1_microglia):
            m = Microglia(self.next_id(), self, phenotype="M1")
            self._place_random(m)
            self.microglia.append(m)
        for _ in range(init_m2_microglia):
            m = Microglia(self.next_id(), self, phenotype="M2")
            self._place_random(m)
            self.microglia.append(m)

        # -------- astrocytes by phenotype --------
        for _ in range(init_a0_astrocytes):
            a = Astrocyte(
                self.next_id(),
                self,
                phenotype="A0",
            )
            self._place_random(a)
            self.astrocytes.append(a)

        for _ in range(init_a1_astrocytes):
            a = Astrocyte(
                self.next_id(),
                self,
                phenotype="A1",
            )
            self._place_random(a)
            self.astrocytes.append(a)

        for _ in range(init_a2_astrocytes):
            a = Astrocyte(
                self.next_id(),
                self,
                phenotype="A2",
            )
            self._place_random(a)
            self.astrocytes.append(a)

        # -------- DataCollector --------
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.steps,
                # neurons
                "damaged_neurons": lambda m: sum(
                    1 for n in m.neurons if (n.pos is not None and n.damaged)
                ),
                "dead_neurons": lambda m: sum(
                    1 for n in m.neurons if (n.pos is not None and n.dead)
                ),
                "healthy_neurons": lambda m: sum(
                    1 for n in m.neurons if (n.pos is not None and n.healthy)
                ),
                "neurons_total": lambda m: sum(
                    1 for n in m.neurons if n.pos is not None
                ),
                # cytokine fields
                "total_pro_inflammation": lambda m: float(m.pro_inflam_val.sum()),
                "total_anti_inflammation": lambda m: float(m.anti_inflam_val.sum()),
                "mean_pro_inflammation": lambda m: float(m.pro_inflam_val.mean()),
                "mean_anti_inflammation": lambda m: float(m.anti_inflam_val.mean()),
                # microglia
                "microglia_total": lambda m: len(m.microglia),
                "microglia_M0": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.phenotype == "M0"
                ),
                "microglia_M1": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.phenotype == "M1"
                ),
                "microglia_M2": lambda m: sum(
                    1 for g in m.microglia
                    if g.pos is not None and g.phenotype == "M2"
                ),
                # astrocytes
                "astrocytes_total": lambda m: len(m.astrocytes),
                "astrocytes_A0": lambda m: sum(
                    1 for a in m.astrocytes
                    if a.pos is not None and a.phenotype == "A0"
                ),
                "astrocytes_A1": lambda m: sum(
                    1 for a in m.astrocytes
                    if a.pos is not None and a.phenotype == "A1"
                ),
                "astrocytes_A2": lambda m: sum(
                    1 for a in m.astrocytes
                    if a.pos is not None and a.phenotype == "A2"
                ),
                # lipid droplet metrics
                "total_ld_neurons": lambda m: float(sum(
                    getattr(n, "lipid_droplets", 0.0)
                    for n in m.neurons
                    if n.pos is not None
                )),
                "total_ld_microglia": lambda m: float(sum(
                    getattr(g, "lipid_droplets", 0.0)
                    for g in m.microglia
                    if g.pos is not None
                )),
                "total_ld_astrocytes": lambda m: float(sum(
                    getattr(a, "lipid_pool", 0.0)
                    for a in m.astrocytes
                    if a.pos is not None
                )),
            }
        )

    # -------- helpers --------

    def next_id(self) -> int:
        self._uid += 1
        return self._uid

    def _place_random(self, agent):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        self.grid.place_agent(agent, (x, y))

    def _torus_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.grid.torus:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return (dx * dx + dy * dy) ** 0.5

    def remove_neuron(self, neuron: Neuron):
        if neuron.pos is not None:
            self.grid.remove_agent(neuron)
            neuron.pos = None

    # -------- damage spatial spread / retraction --------

    def _diffuse_inflammation(self):
        """
        Diffuse neuron-sourced DAMAGE (chemotaxis cue) outward
        from residence centers.
        """
        coords = list(zip(*np.where(self.residence)))
        for x, y in coords:
            r = self.res_curr[x, y]
            if r <= self.damage_radius - 1:
                xs = range(x - r, x + r + 1)
                ys = range(y - r, y + r + 1)
                for xi in xs:
                    for yi in ys:
                        xi2 = xi % self.width
                        yi2 = yi % self.height
                        if max(abs(xi - x), abs(yi - y)) <= r:
                            self.damage_val[xi2, yi2] += 1.0
                self.res_curr[x, y] = r + 1

    def _dismantle_inflammation(self):
        """
        Shrink DAMAGE around patches flagged as 'dismantling'
        (triggered by M2 repair).
        """
        coords = list(zip(*np.where(self.dismantling)))
        to_stop = []
        for x, y in coords:
            r = self.dis_curr[x, y]
            xs = range(x - r, x + r + 1)
            ys = range(y - r, y + r + 1)
            for xi in xs:
                for yi in ys:
                    xi2 = xi % self.width
                    yi2 = yi % self.height
                    if max(abs(xi - x), abs(yi - y)) <= r:
                        self.damage_val[xi2, yi2] = max(
                            0.0, self.damage_val[xi2, yi2] - 1.0
                        )
            r -= 1
            if r <= 0:
                to_stop.append((x, y))
            else:
                self.dis_curr[x, y] = r
        for x, y in to_stop:
            self.dismantling[x, y] = False
            self.dis_curr[x, y] = 0

    def _decay_signals(self):
        """
        Per-tick decay of both cytokine fields.
        This is what makes pro/anti signals die out over time when not replenished.
        """
        if self.pro_decay > 0.0:
            self.pro_inflam_val *= (1.0 - self.pro_decay)
        if self.anti_decay > 0.0:
            self.anti_inflam_val *= (1.0 - self.anti_decay)

        # clamp numerical negatives
        np.maximum(self.pro_inflam_val, 0.0, out=self.pro_inflam_val)
        np.maximum(self.anti_inflam_val, 0.0, out=self.anti_inflam_val)

    # -------- main step --------

    def step(self):
        # 1) microglia act (move, interact, emit cytokines)
        for mg in self.microglia:
            mg.step()

        # 2) astrocytes reshape fields, act on neurons, optionally recruit microglia
        for a in self.astrocytes:
            a.step()

        # 3) neurons update based on local pro/anti fields
        for n in self.neurons:
            n.step()

        # 4) spatial expansion/retraction of damage every 5 ticks
        if self.steps % 5 == 0:
            self._diffuse_inflammation()
            self._dismantle_inflammation()

        # 5) decay of cytokine fields
        self._decay_signals()

        self.datacollector.collect(self)
        self.steps += 1

    def all_damaged_cleared(self) -> bool:
        return not any((n.pos is not None) and n.damaged for n in self.neurons)
