from __future__ import annotations
from typing import Tuple, Set
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import io


class Neuron(Agent):
    """Neuron agent: healthy ↔ damaged → dead. Dead emits signals and waits for phagocytosis."""

    def __init__(
        self,
        unique_id,
        model,
        damaged: bool = False,
        dead: bool = False,
        healthy: bool = True,
        reach: int = 2,
        ticks_damaged: int = 0,
    ):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.damaged: bool = damaged
        self.dead: bool = dead
        self.healthy: bool = healthy and not (damaged or dead)
        self.reach: int = reach
        self.ticks_damaged: int = ticks_damaged
        self.lipid_droplets: float = 0.0

        if self.damaged:
            self.healthy = False
        if self.dead:
            self.healthy = False
            self.damaged = False

    def become_damaged(self):
        if self.dead or self.damaged or self.pos is None:
            return
        self.healthy = False
        self.damaged = True
        self.ticks_damaged = 0
        x, y = self.pos
        self.model.residence[x, y] = True
        self.model.res_curr[x, y] = max(self.model.res_curr[x, y], 1)
        # neuron-sourced pro-inflam & damage (chemotaxis source)
        self.model.pro_inflam_val[x, y] = max(self.model.pro_inflam_val[x, y], 1.0)
        self.model.damage_val[x, y] = max(self.model.damage_val[x, y], 1.0)
        self.reach = 1
        self.lipid_droplets = min(
            self.lipid_droplets + self.model.neuron_ld_damage_boost,
            self.model.neuron_ld_max,
        )
        
    def become_dead(self):
        if self.dead or self.pos is None:
            return
        self.dead = True
        self.damaged = False
        self.healthy = False
        self.ticks_damaged = 0
        x, y = self.pos
        self.model.residence[x, y] = True
        self.model.res_curr[x, y] = max(self.model.res_curr[x, y], 1)
        # stronger pro-inflam & damage source
        self.model.pro_inflam_val[x, y] = max(self.model.pro_inflam_val[x, y], 1.0)
        self.model.damage_val[x, y] = max(self.model.damage_val[x, y], 1.5)
        self.reach = 0
        self.lipid_droplets = min(
            self.lipid_droplets + self.model.neuron_ld_death_boost,
            self.model.neuron_ld_max,
        )

    def become_healthy(self):
        if self.dead or self.healthy:
            return
        self.damaged = False
        self.healthy = True
        self.ticks_damaged = 0
        self.reach = 2
        self.lipid_droplets *= 0.2 
        
    def _maybe_export_lipids_to_astrocytes(self):
        if self.pos is None:
            return
        if self.lipid_droplets <= 0.0:
            return

        # find astrocytes in a radius
        x, y = self.pos
        neigh = self.model.grid.get_neighborhood(
            (x, y),
            moore=True,
            include_center=True,
            radius=self.model.neuron_to_astro_ld_transfer_radius,
        )
        cells = self.model.grid.get_cell_list_contents(neigh)
        astrocytes = [a for a in cells if isinstance(a, Astrocyte)]
        if not astrocytes:
            return

        # with some probability, export one "packet" of LD to a random astrocyte
        if self.model.random.random() < self.model.neuron_to_astro_ld_transfer_prob:
            target = self.model.random.choice(astrocytes)
            packet = min(
                self.lipid_droplets,
                self.model.neuron_ld_packet_size,
            )
            if packet > 0:
                self.lipid_droplets -= packet
                target.lipid_pool += packet


    def step(self):
        if self.pos is None:
            return
        if self.dead:
            return

        # --- params ---
        damage_ratio_thresh = self.model.damage_ratio_thresh
        healthy_ratio_thresh = self.model.healthy_ratio_thresh
        p_damage = self.model.damage_chance
        p_healthy = self.model.healthy_chance
        damage_to_death_ticks = self.model.damage_to_death_ticks
        death_ratio_thresh = self.model.death_ratio_thresh
        p_death = self.model.death_chance
        eps = 1e-9

        x, y = self.pos
        neigh_pos = self.model.grid.get_neighborhood(
            (x, y), moore=True, include_center=False, radius=1
        )
        if not neigh_pos:
            return

        pro_vals = [self.model.pro_inflam_val[nx, ny] for (nx, ny) in neigh_pos]
        anti_vals = [self.model.anti_inflam_val[nx, ny] for (nx, ny) in neigh_pos]
        mean_pro = sum(pro_vals) / len(pro_vals)
        mean_anti = sum(anti_vals) / len(anti_vals)
        mean_ratio = mean_pro / (mean_anti + eps)

        # --- Healthy -> Damaged ---
        if self.healthy and (mean_ratio >= damage_ratio_thresh):
            if self.model.random.random() < p_damage:
                self.become_damaged()
            return

        # --- Damaged branch ---
        if self.damaged:
            self.ticks_damaged += 1
            
            self.lipid_droplets = min(
                self.lipid_droplets + self.model.neuron_ld_production_rate,
                self.model.neuron_ld_max,
            )
            
            self._maybe_export_lipids_to_astrocytes()

            # Damaged -> Dead
            if self.ticks_damaged >= damage_to_death_ticks:
                if mean_ratio >= death_ratio_thresh:
                    if self.model.random.random() < p_death:
                        self.become_dead()
                        return

            # Damaged -> Healthy
            if mean_ratio <= healthy_ratio_thresh:
                if self.model.random.random() < p_healthy:
                    self.become_healthy()
            return


class Astrocyte(Agent):
    def __init__(self, unique_id, model, phenotype: str = "A0"):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.phenotype = phenotype  # "A0", "A1", "A2"
        self.wait_ticks = 0
        self.resolution_ticks = 0  # how long in low-signal environment
        self.coverage_radius = self.model.astro_coverage_radius  # helps establish neighborhoods
        # example: lipid store, if you want later
        self.lipid_pool = 1.0
        
    def _oxidize_lipids(self):
        if self.pos is None:
            return
        if self.lipid_pool <= 0.0:
            return

        x, y = self.pos
        pro = float(self.model.pro_inflam_val[x, y])
        anti = float(self.model.anti_inflam_val[x, y])
        total = pro + anti
        
        if total < self.model.astro_homeo_signal_thresh:
            return
        
        # amount oxidized this tick
        frac = max(0.0, min(1.0, self.model.astro_ld_oxidation_rate))
        used = self.lipid_pool * frac
        if used <= 0.0:
            return

        self.lipid_pool -= used
        self.model.anti_inflam_val[x, y] += used * self.model.astro_ld_to_anti_inflam


    def _neighborhood_positions(self):
        x, y = self.pos
        neigh = self.model.grid.get_neighborhood(
            (x, y), moore=True, include_center=True, radius=self.coverage_radius
        )
        return neigh

    def _nearby_microglia_counts(self):
        """Count nearby M1/M2 microglia in the astrocyte domain."""
        cells = self.model.grid.get_cell_list_contents(self._neighborhood_positions())
        num_M1 = 0
        num_M2 = 0
        for a in cells:
            # Microglia is defined later; this is resolved at runtime.
            if isinstance(a, Microglia):
                if a.phenotype == "M1":
                    num_M1 += 1
                elif a.phenotype == "M2":
                    num_M2 += 1
        return num_M1, num_M2

    def _update_phenotype(self):
        if self.pos is None:
            return

        x, y = self.pos
        pro = float(self.model.pro_inflam_val[x, y])
        anti = float(self.model.anti_inflam_val[x, y])
        total = pro + anti
        eps = 1e-9

        # Astrocyte-specific thresholds / probabilities
        homeo_thresh = self.model.astro_homeo_signal_thresh
        pro_thresh = self.model.astro_pro_inflam_signal_thresh
        anti_thresh = self.model.astro_anti_inflam_signal_thresh
        p_homeo = self.model.astro_homeo_chance
        p_pro = self.model.astro_pro_inflam_chance
        p_anti = self.model.astro_anti_inflam_chance

        # Microglia-driven induction bias
        num_M1, num_M2 = self._nearby_microglia_counts()
        r = self.model.random.random()

        if num_M1 > 0 and r < 0.3:
            # microglial A1-inducing cytokine environment
            self.phenotype = "A1"
        elif num_M2 > 0 and r < 0.3:
            # microglial A2-inducing environment
            self.phenotype = "A2"
        else:
            # fallback to local pro/anti ratio
            if total < homeo_thresh:
                # drift back to A0 over time
                if self.model.random.random() < p_homeo:
                    self.phenotype = "A0"
            else:
                ratio = pro / (anti + eps)
                r2 = self.model.random.random()

                # Strongly pro environment → A1 neurotoxic
                if ratio >= pro_thresh:
                    if r2 < p_pro:
                        self.phenotype = "A1"
                # Strongly anti / resolving environment → A2 protective
                elif ratio <= anti_thresh:
                    if r2 < p_anti:
                        self.phenotype = "A2"
                else:
                    if r2 < p_homeo:
                        self.phenotype = "A0"

        # resolution: A2 -> A0 after long low-signal
        if self.phenotype == "A2":
            if total < homeo_thresh:
                self.resolution_ticks += 1
                if self.resolution_ticks >= self.model.a2_to_a0_resolution_ticks:
                    self.phenotype = "A0"
                    self.resolution_ticks = 0
            else:
                self.resolution_ticks = 0
        else:
            self.resolution_ticks = 0

    # Astrocytes can broadcast inflammation more broad than microglia
    def _modulate_fields(self):
        for nx, ny in self._neighborhood_positions():
            if self.phenotype == "A1":
                # slightly stronger than before so it registers, but still < microglia
                self.model.pro_inflam_val[nx, ny] += 0.15
            elif self.phenotype == "A2":
                self.model.anti_inflam_val[nx, ny] += 0.15
                # damp the pro-inflam values in area
                self.model.pro_inflam_val[nx, ny] = max(
                    0.0, self.model.pro_inflam_val[nx, ny] - 0.07
                )
            else:
                # A0 homeostatic: mild pro cleanup (e.g., glutamate/ROS control)
                self.model.pro_inflam_val[nx, ny] *= 0.99

    def _neuron_in_radius(self):
        cells = self.model.grid.get_cell_list_contents(self._neighborhood_positions())
        return [a for a in cells if isinstance(a, Neuron)]

    # encodes:
    # - glutamate clearance and lipid support (A0/A2) as increased probability of becoming healthy
    # - neurotoxicity (A1) as increased chance of becoming damaged when already stressed
    def _act_on_neurons(self):
        neurons = self._neuron_in_radius()
        if not neurons:
            return

        for neuron in neurons:
            if self.phenotype == "A2":
                # strong local neuroprotection
                if neuron.damaged and not neuron.dead:
                    # base probability + small boost from lipid pool
                    base_p = 0.2
                    lipid_boost = min(self.lipid_pool * 0.01, 0.3)
                    p_heal = min(1.0, base_p + lipid_boost)
                    if self.model.random.random() < p_heal:
                        neuron.become_healthy()
                        
                # A2 also slightly dismantles local damage/excitotoxicity
                for nx, ny in self._neighborhood_positions():
                    self.model.damage_val[nx, ny] = max(
                        0.0, self.model.damage_val[nx, ny] - 0.1
                    )

            elif self.phenotype == "A1":
                # A1 can push healthy neurons into damage, or push damaged neurons to death
                if neuron.healthy:
                    if self.model.random.random() < 0.01:
                        neuron.become_damaged()
                elif neuron.damaged and not neuron.dead:
                    if self.model.random.random() < 0.05:
                        neuron.become_dead()
            else:
                # A0: small, background protection
                if neuron.damaged and not neuron.dead:
                    if self.model.random.random() < 0.02:
                        neuron.become_healthy()

    def _maybe_recruit_microglia(self):
        if self.phenotype != "A1" or self.pos is None:
            return

        x, y = self.pos
        local_damage = float(self.model.damage_val[x, y])
        local_pro = float(self.model.pro_inflam_val[x, y])
        if (local_damage + local_pro) < self.model.recruitment_threshold:
            return

        # probability to recruit one new microglia agent (M1) which then gets sent to source
        if self.model.random.random() < self.model.microglia_recruitment_prob:
            m = Microglia(self.model.next_id(), self.model, phenotype="M1")
            self.model.grid.place_agent(m, (x, y))
            self.model.microglia.append(m)

    def step(self):
        """Main astrocyte behavior per tick."""
        if self.pos is None:
            return

        self._update_phenotype()
        self._oxidize_lipids()
        self._modulate_fields()
        self._act_on_neurons()
        self._maybe_recruit_microglia()


class Microglia(Agent):
    """
    Microglia with three phenotypes:
      M0 = homeostatic (surveillance)
      M1 = pro-inflammatory (attack)
      M2 = anti-inflammatory (repair)
    """

    def __init__(self, unique_id, model, phenotype: str):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.phenotype = phenotype  # "M0", "M1", "M2"
        self.wait_ticks: int = 0
        self.resolution_ticks: int = 0  # how long M2 has sat in resolved environment
        self.lipid_droplets: float = 0.0 # LD burden in Microglia

    def _update_phenotype(self):
        """
        Switch M0/M1/M2 based on a single ratio:

            ratio = pro / (anti + eps),

        plus:
        - a minimal total-signal threshold to allow homeostasis,
        - M2 -> M0 if low-signal persists.
        """
        if self.pos is None:
            return

        x, y = self.pos
        pro = float(self.model.pro_inflam_val[x, y])
        anti = float(self.model.anti_inflam_val[x, y])
        total = pro + anti
        eps = 1e-9

        homeo_thresh = self.model.homeo_signal_thresh
        pro_thresh = self.model.pro_inflam_signal_thresh
        anti_thresh = self.model.anti_inflam_signal_thresh
        p_homeo = self.model.homeo_chance
        p_pro = self.model.pro_inflam_chance
        p_anti = self.model.anti_inflam_chance

        # Very low signal: allow drift back to M0
        if total < homeo_thresh:
            if self.model.random.random() < p_homeo:
                self.phenotype = "M0"
            # resolution logic handled below
        else:
            ratio = pro / (anti + eps)
            r = self.model.random.random()

            if ratio >= pro_thresh:
                if r < p_pro:
                    self.phenotype = "M1"
            elif ratio <= anti_thresh:
                if r < p_anti:
                    self.phenotype = "M2"
            else:
                if r < p_homeo:
                    self.phenotype = "M0"

        # --- Resolution-based M2 -> M0 ---
        if self.phenotype == "M2":
            if total < homeo_thresh:
                self.resolution_ticks += 1
                if self.resolution_ticks >= self.model.m2_to_m0_resolution_ticks:
                    self.phenotype = "M0"
                    self.resolution_ticks = 0
            else:
                self.resolution_ticks = 0
        else:
            self.resolution_ticks = 0

    def _random_step(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        if neighbors:
            new_pos = self.model.random.choice(neighbors)
            self.model.grid.move_agent(self, new_pos)

    def _chemotaxis_step(self, field_name: str):
        """
        Gradient-following that always moves to a neighbor.

        Used for chemotaxis up neuron/lesion-derived damage field.
        """
        field = getattr(self.model, field_name)
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        if not neighbors:
            return

        vals = [(n, self._field_value(field, n)) for n in neighbors]
        max_val = max(v for _, v in vals)
        candidates = [p for p, v in vals if v == max_val]
        new_pos = self.model.random.choice(candidates)
        self.model.grid.move_agent(self, new_pos)

    @staticmethod
    def _field_value(field, pos):
        try:
            return field[pos]
        except TypeError:
            # if field is a dict-like
            return field.get(pos, 0.0)

    # phenotype-specific movement
    def _move_M0(self):
        """Homeostatic: simple random surveillance."""
        self._random_step()

    def _move_M1(self):
        """Pro-inflammatory: chemotaxis up neuron-derived damage gradient."""
        if self.model.random.random() < self.model.sensing_efficiency:
            self._chemotaxis_step("damage_val")
        else:
            self._random_step()

    def _move_M2(self):
        """Anti-inflammatory: same chemotaxis target (damage), different behavior at target."""
        if self.model.random.random() < self.model.sensing_efficiency:
            self._chemotaxis_step("damage_val")
        else:
            self._random_step()

    # -------- interactions with neurons --------
    def _neurons_here(self):
        """All neuron agents at this patch."""
        cell_agents = self.model.grid.get_cell_list_contents([self.pos])
        return [a for a in cell_agents if isinstance(a, Neuron)]

    def _interact_M0(self) -> bool:
        """
        M0: surveillance + quiet clearance of dead neurons.

        - If a dead neuron is present and phagocytosed -> wait 5 ticks.
        - If *any* neuron is present but not eaten     -> "surveillance" -> wait 5 ticks.
        """
        neurons = self._neurons_here()
        if not neurons:
            return False

        # Prefer to clear dead neurons if present
        for neuron in neurons:
            if neuron.dead and neuron.pos is not None:
                if self.model.random.random() < self.model.eat_probability:
                    self.model.remove_neuron(neuron)
                    x, y = self.pos
                    self.model.anti_inflam_val[x, y] += 0.05
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True

        # No dead neuron, but there is at least one neuron -> surveillance
        self.wait_ticks = 5
        return True

    def _interact_M1(self) -> bool:
        """
        M1: aggressive phagocytosis of damaged/dead neurons.

        - On successful phagocytosis -> wait 5 ticks.
        """
        neurons = self._neurons_here()
        for neuron in neurons:
            if (neuron.damaged or neuron.dead) and neuron.pos is not None:
                p = min(1.0, self.model.eat_probability * 1.2)
                if self.model.random.random() < p:
                    ld_from_neuron = getattr(neuron, "lipid_droplets", 0.0)
                    self.lipid_droplets += ld_from_neuron + self.model.microglia_ld_from_phagocytosis
                    self.lipid_droplets = min(self.lipid_droplets, self.model.microglia_ld_max)

                    self.model.remove_neuron(neuron)
                    x, y = self.pos
                    self.model.pro_inflam_val[x, y] += 1.0
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
        return False

    def _interact_M2(self) -> bool:
        """
        M2: repair/cleanup.
          - clears dead neurons
          - helps damaged neurons recover

        - Any successful phagocytosis or healing -> wait 5 ticks.
        """
        neurons = self._neurons_here()
        for neuron in neurons:
            if neuron.dead and neuron.pos is not None:
                if self.model.random.random() < self.model.eat_probability:
                    # capture neuron LD + baseline LD from phagocytosis
                    ld_from_neuron = getattr(neuron, "lipid_droplets", 0.0)
                    self.lipid_droplets += ld_from_neuron + self.model.microglia_ld_from_phagocytosis
                    self.lipid_droplets = min(self.lipid_droplets, self.model.microglia_ld_max)
                    
                    self.model.remove_neuron(neuron)
                    x, y = self.pos
                    self.model.anti_inflam_val[x, y] += 1.0
                    # dismantle damage wave from this site
                    self.model.dismantling[x, y] = True
                    self.model.dis_curr[x, y] = self.model.damage_radius
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
            elif neuron.damaged and (not neuron.dead) and neuron.pos is not None:
                if self.model.random.random() < self.model.sensing_efficiency:
                    neuron.become_healthy()
                    x, y = self.pos
                    self.model.anti_inflam_val[x, y] += 1.0
                    self.model.pro_inflam_val[x, y] = max(
                        0.0, self.model.pro_inflam_val[x, y] - 1.0
                    )
                    self.wait_ticks = 5
                else:
                    self.wait_ticks = 5
                return True
        return False


    # -------- cytokine emission along path --------
    def _emit_cytokines_at(self, pos):
        """
        Emit cytokines at (and around) the given position.
        Called after *each move*, so microglia leave a trail along the damage gradient.
        """
        if pos is None:
            return

        # local neighborhood
        neigh = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)

        if self.phenotype == "M1":
            ld_factor = 1.0
            if self.lipid_droplets >= self.model.microglia_ld_pro_inflam_threshold:
                ld_factor += self.model.microglia_ld_pro_inflam_boost
            # pro-inflam trail
            for nx, ny in neigh:
                self.model.pro_inflam_val[nx, ny] += 0.2 * ld_factor  # small per-move increment
        elif self.phenotype == "M2":
            # anti-inflam trail
            for nx, ny in neigh:
                self.model.anti_inflam_val[nx, ny] += 0.2

    def step(self):
        # If "busy" (engulfing / surveilling), just count down
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return
        if self.pos is None:
            return

        # 1) Update phenotype based on cytokine ratio + resolution
        self._update_phenotype()

        # 2) Movement with phenotype-specific speed
        base_speed = max(0.0, 1.0 + 0.05 * (72 - 37.0))  # your original temperature logic
        if self.phenotype == "M1":
            speed_factor = 1.5
        else:
            speed_factor = 1.0
        speed = base_speed * speed_factor
        moves = 1
        extra = speed - 1.0
        if extra > 0 and self.model.random.random() < extra:
            moves += 1

        acted = False
        for _ in range(moves):
            if self.phenotype == "M1":
                self._move_M1()
                acted = self._interact_M1()
            elif self.phenotype == "M2":
                self._move_M2()
                acted = self._interact_M2()
            else:  # M0
                self._move_M0()
                acted = self._interact_M0()

            # emit cytokines along the path
            self._emit_cytokines_at(self.pos)

            if acted:
                break
