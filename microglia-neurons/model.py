# model.py
from __future__ import annotations
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agents import Microglia, Neuron


class MicrogliaNeuronModel(Model):
    def __init__(
        self,
        width: int = 33,
        height: int = 33,
        torus: bool = True,
        init_microglia: int = 5,
        init_h_neuron: int = 10,
        init_d_neuron: int = 10,
        eat_probability: float = 0.70,
        sensing_efficiency: float = 0.50,
        damage_chance: float = 0.005,
        neuron_distance: int = 5,
        temperature: float = 37.0,
        inflam_radius: int = 3,
        seed: int | None = None,
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
        self.temperature = float(temperature)
        self.inflam_radius = int(inflam_radius)

        # patch fields
        shp = (width, height)
        self.inflam_val = np.zeros(shp, dtype=int)
        self.residence = np.zeros(shp, dtype=bool)
        self.res_curr = np.zeros(shp, dtype=int)
        self.dismantling = np.zeros(shp, dtype=bool)
        self.dis_curr = np.zeros(shp, dtype=int)

        # agents
        self.microglia: list[Microglia] = []
        self.neurons: list[Neuron] = []

        # neurons
        for _ in range(init_h_neuron):
            n = Neuron(self.next_id(), self, damaged=False)
            self._place_random(n)
            self.neurons.append(n)
        for _ in range(init_d_neuron):
            n = Neuron(self.next_id(), self, damaged=True)
            self._place_random(n)
            self.neurons.append(n)
            x, y = n.pos
            # NetLogo resonance: seed residence + initial ring and initial inflam
            self.residence[x, y] = True
            self.res_curr[x, y] = max(self.res_curr[x, y], 1)
            self.inflam_val[x, y] = max(self.inflam_val[x, y], 1)

        # wire network: ONE random link per neuron within distance (NetLogo-like)
        self._wire_neuron_network()

        # microglia
        for _ in range(init_microglia):
            m = Microglia(self.next_id(), self)
            self._place_random(m)
            self.microglia.append(m)

        # data
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.steps,
                "damaged_neurons": lambda m: sum(1 for n in m.neurons if n.pos is not None and n.damaged),
                "healthy_neurons": lambda m: sum(1 for n in m.neurons if n.pos is not None and not n.damaged),
                "total_inflammation": lambda m: int(m.inflam_val.sum()),
                "mean_inflammation": lambda m: float(m.inflam_val.mean()),
                "microglia": lambda m: len(m.microglia),
            }
        )

    def next_id(self) -> int:
        self._uid += 1
        return self._uid

    def _place_random(self, agent):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        self.grid.place_agent(agent, (x, y))

    # Toroidal distance helper (MultiGrid has neighborhood APIs but no get_distance)
    def _torus_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = abs(a[0] - b[0]); dy = abs(a[1] - b[1])
        if self.grid.torus:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return (dx * dx + dy * dy) ** 0.5

    def _wire_neuron_network(self):
        # NetLogo-like: for each neuron, choose at most ONE neighbor within radius and link
        pos_to_neuron = {n.pos: n for n in self.neurons if n.pos is not None}
        for n in self.neurons:
            if n.pos is None:
                continue
            hood = self.grid.get_neighborhood(n.pos, moore=True, include_center=False, radius=self.neuron_distance)
            cand = [pos_to_neuron[p] for p in hood if p in pos_to_neuron and pos_to_neuron[p] is not n]
            if cand:
                m = self.random.choice(cand)
                n.links.add(m); m.links.add(n)

    def remove_neuron(self, neuron: Neuron):
        if neuron.pos is not None:
            self.grid.remove_agent(neuron)
            neuron.pos = None

    def _diffuse_inflammation(self):
        coords = list(zip(*np.where(self.residence)))
        for x, y in coords:
            r = self.res_curr[x, y]
            if r <= self.inflam_radius - 1:
                xs = range(x - r, x + r + 1)
                ys = range(y - r, y + r + 1)
                for xi in xs:
                    for yi in ys:
                        xi2 = xi % self.width
                        yi2 = yi % self.height
                        if max(abs(xi - x), abs(yi - y)) <= r:
                            self.inflam_val[xi2, yi2] += 1
                self.res_curr[x, y] = r + 1

    def _dismantle_inflammation(self):
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
                        self.inflam_val[xi2, yi2] = max(0, self.inflam_val[xi2, yi2] - 1)
            r -= 1
            if r <= 0:
                to_stop.append((x, y))
            else:
                self.dis_curr[x, y] = r
        for x, y in to_stop:
            self.dismantling[x, y] = False
            self.dis_curr[x, y] = 0

    def step(self):
        self.random.shuffle(self.microglia)
        for mg in self.microglia:
            mg.step()
        for n in self.neurons:
            n.step()
        if self.steps % 5 == 0:
            self._diffuse_inflammation()
            self._dismantle_inflammation()
        self.datacollector.collect(self)
        self.steps += 1

    def all_damaged_cleared(self) -> bool:
        return all((not n.damaged) or (n.pos is None) for n in self.neurons)
