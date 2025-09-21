# agents.py
from __future__ import annotations
from typing import Tuple, Set
from mesa import Agent


class Neuron(Agent):
    """Neuron agent: healthy or damaged; static location; synaptic links to other neurons."""
    def __init__(self, unique_id, model, damaged: bool = False):
        # Avoid super().__init__; set fields directly.
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.damaged: bool = damaged
        self.links: Set["Neuron"] = set()

    def become_damaged(self):
        if not self.damaged and self.pos is not None:
            self.damaged = True
            x, y = self.pos
            self.model.residence[x, y] = True
            # NetLogo-like: mark residence and (if not already) start spread radius at 1 and seed inflam
            self.model.res_curr[x, y] = max(self.model.res_curr[x, y], 1)
            self.model.inflam_val[x, y] = max(self.model.inflam_val[x, y], 1)

    def step(self):
        if self.pos is None or self.damaged:
            return
        if any(n.damaged for n in self.links):
            if self.model.random.random() < self.model.damage_chance:
                self.become_damaged()


class Microglia(Agent):
    """Microglia with chemotaxis, surveillance, and phagocytosis; temperature-scaled motion."""
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.wait_ticks: int = 0

    def pick_chemotaxis_target(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        neigh = self.model.grid.get_neighborhood(pos, moore=True, include_center=True, radius=1)
        vals = []
        for cell in neigh:
            x, y = cell
            vals.append((cell, self.model.inflam_val[x, y]))
        max_val = max(v for _, v in vals)
        if all(v == max_val for _, v in vals):
            choices = [c for c in neigh if c != pos]
            return self.model.random.choice(choices) if choices else pos
        candidates = [c for c, v in vals if v == max_val]
        return self.model.random.choice(candidates)

    def undirected_target(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        neigh = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=1)
        return self.model.random.choice(neigh)

    def move_once(self):
        x, y = self.pos
        if self.model.random.random() < self.model.sensing_efficiency:
            tx, ty = self.pick_chemotaxis_target((x, y))
        else:
            tx, ty = self.undirected_target((x, y))
        self.model.grid.move_agent(self, (tx, ty))

    def interact_here(self) -> bool:
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        neurons = [a for a in cellmates if isinstance(a, Neuron)]
        if not neurons:
            return False
        n: Neuron = neurons[0]
        if n.damaged:
            if self.model.random.random() < self.model.eat_probability:
                px, py = n.pos
                self.model.remove_neuron(n)
                if self.model.residence[px, py]:
                    self.model.residence[px, py] = False
                    self.model.dis_curr[px, py] = self.model.res_curr[px, py]
                    self.model.res_curr[px, py] = 0
                    self.model.dismantling[px, py] = True
            self.wait_ticks += 5
            return True
        else:
            self.wait_ticks += 2
            return True

    def step(self):
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        # NetLogo resonance: one move per tick + a probabilistic extra move based on temperature
        # Expected moves ~= 1 + 0.05*(T-37) without unbounded multi-moves per tick
        speed = max(0.0, 1.0 + 0.05 * (self.model.temperature - 37.0))
        moves = 1
        extra = speed - 1.0
        if extra > 0 and self.model.random.random() < extra:
            moves += 1

        for _ in range(moves):
            self.move_once()
            if self.interact_here():
                break
