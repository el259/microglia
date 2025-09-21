# agents.py
# Microglia metabolism ABM — agents for Mesa

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import random
from mesa import Agent

HOMEOSTATIC = "homeostatic"   # NetLogo 'microglia' breed
M1           = "m1"           # NetLogo 'm1-microglia' breed


@dataclass
class MGState:
    stopped: bool = False
    ticks_on_vessel: int = 0
    immune_tolerance: bool = False
    tolerance_timer: int = 0
    tolerance_cooldown: int = 0
    has_energy: bool = True


class Microglia(Agent):
    """
    A single microglia agent. Encodes both phenotypes via `phenotype` field:
    - 'homeostatic'  ↔ NetLogo breed `microglia`
    - 'm1'           ↔ NetLogo breed `m1-microglia`
    """
    def __init__(self, unique_id, model, phenotype= HOMEOSTATIC):
        super().__init__(model)
        self.unique_id = unique_id
        self.phenotype = phenotype
        self.s = MGState()

    # ----- movement (random walk; NetLogo: rt random 50, lt random 50, fd 1) -----
    def random_grid_move(self):
        if self.s.stopped or not self.s.has_energy:
            return
        # Move to one of the Moore neighbors (8-neighborhood) or stay in place if blocked
        x, y = self.pos
        candidates = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)
        if candidates:
            self.model.grid.move_agent(self, self.random.choice(candidates))

    # ----- micro-environment queries -----
    def on_vessel(self) -> bool:
        x, y = self.pos
        return self.model.integrity[x, y] > 0.0

    def on_plaque(self) -> bool:
        x, y = self.pos
        return self.model.plaque_val[x, y] == 1

    def near_plaque(self) -> bool:
        # neighbors (Moore radius 1) contain plaque?
        for nx, ny in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1):
            if self.model.plaque_val[nx, ny] == 1:
                return True
        return False

    # ----- metabolism (oxphos vs glycolysis) -----
    def oxphos(self):
        # NetLogo: needs 42 glucose; does NOT create excess lactate
        if self.model.global_glucose >= 42:
            self.model.global_glucose -= 42
            self.s.has_energy = True
        else:
            self.s.has_energy = False
        # explicitly: oxphos sets no lactate flag at patch level
        x, y = self.pos
        # no-op for lactate flags here; trails are only from glycolysis (M1)

    def glycolysis(self):
        # NetLogo: needs 420 glucose; DOES create lactate trail if energy acquired
        if self.model.global_glucose >= 420:
            self.model.global_glucose -= 420
            self.s.has_energy = True
        else:
            self.s.has_energy = False

        if self.s.has_energy:
            x, y = self.pos
            # mark trail origin and initialize if needed (mirrors lactate? at patch-here)
            if not self.model.lactate_flag[x, y] and not self.model.perm_lactate[x, y]:
                self.model.lactate_flag[x, y] = True
                # initialize a fresh trail on this origin patch
                self.model.curr_spread[x, y] = 0
                if self.model.max_spread[x, y] == 0:
                    self.model.max_spread[x, y] = 3
                self.model.spread_life[x, y] = 10
                # ensure dismantle will act on downstream non-permanent trail patches
                self.model.lactate_dismantle[x, y] = False

    # ----- actions on BBB & plaques (per NetLogo procedures) -----
    def survey_and_fortify(self):
        # NetLogo: stop, then with prob fortify_probability add +0.01 to integrity (cap at 1)
        self.s.stopped = True
        x, y = self.pos
        if self.random.random() < self.model.fortify_probability:
            self.model.integrity[x, y] = min(1.0, self.model.integrity[x, y] + 0.01)

    def phagocytose_bbb(self):
        # NetLogo: stop, then with prob eat_probability reduce integrity by 0.01 (floored at 0)
        # If integrity <= 0, revert to homeostatic & resume
        self.s.stopped = True
        x, y = self.pos
        if self.random.random() < self.model.eat_probability:
            self.model.integrity[x, y] = max(0.0, self.model.integrity[x, y] - 0.01)

        if self.model.integrity[x, y] <= 0.0:
            self.phenotype = HOMEOSTATIC
            self.s.stopped = False
            self.model.integrity[x, y] = 0.0

    def phagocytose_plaque(self):
        # NetLogo: stopped, success if (random-float 1 * plaque-eat-scalar < eat-probability)
        # That is success probability = min(1, eat_probability / plaque_eat_scalar)
        self.s.stopped = True
        success_prob = min(1.0, self.model.eat_probability / max(1e-9, self.model.plaque_eat_scalar))
        if self.random.random() < success_prob:
            x, y = self.pos
            self.model.plaque_val[x, y] = 0
            # When plaque is cleared, return to homeostatic
            self.phenotype = HOMEOSTATIC
            self.s.stopped = False

    # ----- tolerance logic -----
    def update_tolerance_timers(self):
        # Close to plaque → accumulate tolerance; if far, decrement tolerance_timer
        near = self.near_plaque()
        if near:
            # near plaques increments tolerance timer (both phenotypes)
            self.s.tolerance_timer += 1
        else:
            if self.s.tolerance_timer > 0:
                self.s.tolerance_timer -= 1

        # If tolerant, and now away from plaques, run cooldown and then reset
        if self.s.immune_tolerance:
            if not near:
                self.s.tolerance_cooldown = max(0, self.s.tolerance_cooldown - 1)
                if self.s.tolerance_cooldown == 0:
                    self.s.immune_tolerance = False
        else:
            # Become tolerant if timer >= 24 h; set cooldown to 96 h
            if self.s.tolerance_timer >= 24:
                self.s.immune_tolerance = True
                self.s.tolerance_cooldown = 96

    # ----- one step -----
    def step(self):
        # 1) phenotype metabolism
        if self.phenotype == HOMEOSTATIC:
            self.oxphos()
        else:
            self.glycolysis()

        # 2) move (if not stopped and has energy)
        self.random_grid_move()
        if not self.s.stopped and self.s.has_energy:
            self.s.ticks_on_vessel = 0  # reset timer when moving

        # 3) tolerance dynamics
        self.update_tolerance_timers()

        # 4) actions by phenotype
        if not self.s.has_energy:
            # no energy, skip actions this tick
            return

        x, y = self.pos

        if self.phenotype == HOMEOSTATIC:
            # homeostatic: fortify vessels; may switch to M1 if near plaques (unless tolerant)
            if self.on_vessel():
                self.survey_and_fortify()
                if self.s.stopped:
                    self.s.ticks_on_vessel += 1
            if self.near_plaque() and not self.s.immune_tolerance:
                self.phenotype = M1
            if self.on_plaque():
                # while homeostatic on plaque, try to eat; contact increases tolerance_timer already
                self.phagocytose_plaque()

            # If stayed stopped on vessel for 40 ticks, resume
            if self.s.ticks_on_vessel >= 40:
                self.s.ticks_on_vessel = 0
                self.s.stopped = False

        else:
            # M1: degrade BBB & eat plaques, produce lactate trails
            if self.on_plaque():
                self.phagocytose_plaque()

            if self.on_vessel():
                self.phagocytose_bbb()
                if self.s.stopped:
                    self.s.ticks_on_vessel += 1
            # resume after 40 ticks stopped on vessel
            if self.s.ticks_on_vessel >= 40:
                self.s.ticks_on_vessel = 0
                self.s.stopped = False
