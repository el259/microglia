# app.py
from __future__ import annotations
import argparse
import os
import io
import time
import random
import numpy as np
import pandas as pd
import altair as alt
import solara
import pyperclip
from solara.lab import use_task
import ipywidgets as widgets
import base64

# CLI-only pyplot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
from datetime import datetime

##

from agents import Microglia, Neuron, Astrocyte
from model import MicrogliaNeuronModel


# ----------------- Core simulation helpers -----------------
def run_sim(
    steps=500,
    width=33,
    height=33,
    m0=3,
    m1=8,
    m2=2,
    h_neurons=10,
    d_neurons=10,
    eat=0.70,
    sense=0.50,
    damage=0.005,
    ndist=5,
    radius=3,
    seed=None,

    # astrocyte knobs
    a0_astro=10,
    a1_astro=5,
    a2_astro=5,
    astro_radius=2,
    a2_res=15,
    recruit_thresh=3.0,
    recruit_prob=0.01,

    # lipid knobs (already in your UI)
    neuron_ld_production_rate=0.2,
    neuron_ld_packet_size=0.5,
    neuron_to_astro_ld_transfer_prob=0.3,
    astro_ld_oxidation_rate=0.3,
    microglia_ld_pro_inflam_threshold=5.0,

    # ---------------- NEW: Neuron transition knobs ----------------
    damage_ratio_thresh=0.6,
    healthy_ratio_thresh=0.2,
    healthy_chance=0.20,
    damage_to_death_ticks=20,
    death_ratio_thresh=0.9,
    death_chance=0.25,

    # ---------------- NEW: Microglia phenotype signal knobs ----------------
    homeo_signal_thresh=1.0,
    pro_inflam_signal_thresh=0.2,
    anti_inflam_signal_thresh=0.8,
    homeo_chance=0.2,
    pro_inflam_chance=0.2,
    anti_inflam_chance=0.2,
    m2_to_m0_resolution_ticks=10,

    # ---------------- NEW: Astrocyte phenotype signal knobs ----------------
    astro_homeo_signal_thresh=0.5,
    astro_pro_inflam_signal_thresh=0.15,
    astro_anti_inflam_signal_thresh=0.6,
    astro_homeo_chance=0.25,
    astro_pro_inflam_chance=0.3,
    astro_anti_inflam_chance=0.3,
    a2_to_a0_resolution_ticks=15,  # you already effectively expose as a2_res; this is the model param name

    # ---------------- NEW: Field decay knobs ----------------
    pro_decay=0.20,
    anti_decay=0.10,

    # ---------------- NEW: Extra lipid dynamics knobs (present in model) ----------------
    neuron_ld_damage_boost=0.5,
    neuron_ld_death_boost=1.0,
    neuron_to_astro_ld_transfer_radius=2,
    astro_ld_to_anti_inflam=0.05,
    microglia_ld_from_phagocytosis=0.5,
    microglia_ld_pro_inflam_boost=0.5,
):
    model = MicrogliaNeuronModel(
        width=width,
        height=height,
        init_m0_microglia=m0,
        init_m1_microglia=m1,
        init_m2_microglia=m2,
        init_healthy_neuron=h_neurons,
        init_damaged_neuron=d_neurons,
        init_dead_neuron=0,
        eat_probability=eat,
        sensing_efficiency=sense,
        damage_chance=damage,
        neuron_distance=ndist,
        damage_radius=radius,
        seed=seed,

        # astrocytes
        init_a0_astrocytes=a0_astro,
        init_a1_astrocytes=a1_astro,
        init_a2_astrocytes=a2_astro,
        astro_coverage_radius=astro_radius,
        a2_to_a0_resolution_ticks=a2_to_a0_resolution_ticks,  # model name

        recruitment_threshold=recruit_thresh,
        microglia_recruitment_prob=recruit_prob,

        # lipid knobs you already had
        neuron_ld_production_rate=neuron_ld_production_rate,
        neuron_ld_packet_size=neuron_ld_packet_size,
        neuron_to_astro_ld_transfer_prob=neuron_to_astro_ld_transfer_prob,
        astro_ld_oxidation_rate=astro_ld_oxidation_rate,
        microglia_ld_pro_inflam_threshold=microglia_ld_pro_inflam_threshold,

        # NEW: neuron transition
        damage_ratio_thresh=damage_ratio_thresh,
        healthy_ratio_thresh=healthy_ratio_thresh,
        healthy_chance=healthy_chance,
        damage_to_death_ticks=damage_to_death_ticks,
        death_ratio_thresh=death_ratio_thresh,
        death_chance=death_chance,

        # NEW: microglia phenotype signals + resolution
        homeo_signal_thresh=homeo_signal_thresh,
        pro_inflam_signal_thresh=pro_inflam_signal_thresh,
        anti_inflam_signal_thresh=anti_inflam_signal_thresh,
        homeo_chance=homeo_chance,
        pro_inflam_chance=pro_inflam_chance,
        anti_inflam_chance=anti_inflam_chance,
        m2_to_m0_resolution_ticks=m2_to_m0_resolution_ticks,

        # NEW: astrocyte phenotype signals
        astro_homeo_signal_thresh=astro_homeo_signal_thresh,
        astro_pro_inflam_signal_thresh=astro_pro_inflam_signal_thresh,
        astro_anti_inflam_signal_thresh=astro_anti_inflam_signal_thresh,
        astro_homeo_chance=astro_homeo_chance,
        astro_pro_inflam_chance=astro_pro_inflam_chance,
        astro_anti_inflam_chance=astro_anti_inflam_chance,

        # NEW: field decay
        pro_decay=pro_decay,
        anti_decay=anti_decay,

        # NEW: extra lipid dynamics
        neuron_ld_damage_boost=neuron_ld_damage_boost,
        neuron_ld_death_boost=neuron_ld_death_boost,
        neuron_to_astro_ld_transfer_radius=neuron_to_astro_ld_transfer_radius,
        astro_ld_to_anti_inflam=astro_ld_to_anti_inflam,
        microglia_ld_from_phagocytosis=microglia_ld_from_phagocytosis,
        microglia_ld_pro_inflam_boost=microglia_ld_pro_inflam_boost,
    )

    for _ in range(steps):
        if not any((n.pos is not None) and n.damaged for n in model.neurons):
            break
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    return model, df

def main():
    ap = argparse.ArgumentParser(description="Microglia–Neuron ABM")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--width", type=int, default=33)
    ap.add_argument("--height", type=int, default=33)
    ap.add_argument("--m0", type=int, default=5)
    ap.add_argument("--m1", type=int, default=5)
    ap.add_argument("--m2", type=int, default=5)
    ap.add_argument("--h_neurons", type=int, default=10)
    ap.add_argument("--d_neurons", type=int, default=10)
    ap.add_argument("--eat", type=float, default=0.70)
    ap.add_argument("--sense", type=float, default=0.50)
    ap.add_argument("--damage", type=float, default=0.005)
    ap.add_argument("--ndist", type=int, default=5)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    # astrocyte flags (CLI)
    ap.add_argument("--a0_astro", type=int, default=10)
    ap.add_argument("--a1_astro", type=int, default=5)
    ap.add_argument("--a2_astro", type=int, default=5)
    ap.add_argument("--astro_radius", type=int, default=2)
    ap.add_argument("--a2_res", type=int, default=15)
    ap.add_argument("--recruit_thresh", type=float, default=3.0)
    ap.add_argument("--recruit_prob", type=float, default=0.01)
    # lipid-related flags
    ap.add_argument("--neuron_ld_release_prob_healthy", type=float, default=0.01)
    ap.add_argument("--neuron_ld_release_prob_damaged", type=float, default=0.10)
    ap.add_argument("--ld_packet_size", type=float, default=1.0)
    ap.add_argument("--max_ld_neuron", type=float, default=10.0)
    ap.add_argument("--max_ld_astro", type=float, default=50.0)
    ap.add_argument("--microglia_ld_inflam_threshold", type=float, default=20.0)
    ap.add_argument("--ld_to_m1_prob", type=float, default=0.20)


    args, _ = ap.parse_known_args()

    model, df_raw = run_sim(**vars(args))

    # Normalize columns similar to the UI
    df = df_raw.copy()
    if "total_inflammation" not in df.columns:
        if "total_pro_inflammation" in df.columns and "total_anti_inflammation" in df.columns:
            df["total_inflammation"] = (
                df["total_pro_inflammation"] + df["total_anti_inflammation"]
            )
    if "mean_inflammation" not in df.columns:
        if "mean_pro_inflammation" in df.columns and "mean_anti_inflammation" in df.columns:
            df["mean_inflammation"] = (
                df["mean_pro_inflammation"] + df["mean_anti_inflammation"]
            )

    print("\nFinal snapshot:")
    print(df.tail(1).to_string(index=False))
    print(f"Stopped at tick {int(df['step'].iloc[-1])}")

    # NOTE: now 4 subplots to include astrocytes
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))

    # Neurons
    axs[0].plot(df["step"], df["healthy_neurons"], label="healthy")
    axs[0].plot(df["step"], df["damaged_neurons"], label="damaged")
    if "dead_neurons" in df.columns:
        axs[0].plot(df["step"], df["dead_neurons"], label="dead")
    if "neurons_total" in df.columns:
        axs[0].plot(df["step"], df["neurons_total"], label="total")
    axs[0].set_title("Neurons")
    axs[0].legend()

    # Inflammation (pro / anti / total)
    if "total_pro_inflammation" in df.columns:
        axs[1].plot(df["step"], df["total_pro_inflammation"], label="pro")
    if "total_anti_inflammation" in df.columns:
        axs[1].plot(df["step"], df["total_anti_inflammation"], label="anti")
    if "total_inflammation" in df.columns:
        axs[1].plot(df["step"], df["total_inflammation"], label="total")
    axs[1].set_title("Inflammation")
    axs[1].legend()

    # Microglia
    if "microglia_total" in df.columns:
        axs[2].plot(df["step"], df["microglia_total"], label="total")
    if "microglia_M0" in df.columns:
        axs[2].plot(df["step"], df["microglia_M0"], label="M0")
    if "microglia_M1" in df.columns:
        axs[2].plot(df["step"], df["microglia_M1"], label="M1")
    if "microglia_M2" in df.columns:
        axs[2].plot(df["step"], df["microglia_M2"], label="M2")
    axs[2].set_title("Microglia")
    axs[2].legend()

    # Astrocytes (if present)
    if "astrocytes_total" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_total"], label="total")
    if "astrocytes_A0" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_A0"], label="A0")
    if "astrocytes_A1" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_A1"], label="A1")
    if "astrocytes_A2" in df.columns:
        axs[3].plot(df["step"], df["astrocytes_A2"], label="A2")
    axs[3].set_title("Astrocytes")
    axs[3].legend()

    plt.tight_layout()
    if os.environ.get("SOLARA_SERVER"):
        plt.savefig("summary.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)



# ----------------- Solara UI -----------------
@solara.component
def Page():
    solara.Title("Microglia–Neuron ABM — Live")

    # ----------- Controls / parameters ----------->
    steps, set_steps = solara.use_state(5000)  # tick cap
    m0, set_m0 = solara.use_state(5)
    m1, set_m1 = solara.use_state(5)
    m2, set_m2 = solara.use_state(5)
    h_neurons, set_hn = solara.use_state(10)
    d_neurons, set_dn = solara.use_state(10)
    eat, set_eat = solara.use_state(0.70)
    sense, set_sense = solara.use_state(0.50)
    damage, set_damage = solara.use_state(0.005)
    ndist, set_ndist = solara.use_state(5)
    radius, set_radius = solara.use_state(3)
    seed, set_seed = solara.use_state(32)

    # astrocyte knobs
    a0_astro, set_a0_astro = solara.use_state(10)
    a1_astro, set_a1_astro = solara.use_state(5)
    a2_astro, set_a2_astro = solara.use_state(5)
    astro_radius, set_astro_radius = solara.use_state(2)
    a2_res, set_a2_res = solara.use_state(15)
    recruit_thresh, set_recruit_thresh = solara.use_state(3.0)
    recruit_prob, set_recruit_prob = solara.use_state(0.01)

    # lipid knobs
    neuron_ld_production_rate, set_neuron_ld_production_rate = solara.use_state(0.2)
    neuron_ld_packet_size, set_neuron_ld_packet_size = solara.use_state(0.5)
    neuron_to_astro_ld_transfer_prob, set_neuron_to_astro_ld_transfer_prob = solara.use_state(0.3)
    astro_ld_oxidation_rate, set_astro_ld_oxidation_rate = solara.use_state(0.3)
    microglia_ld_pro_inflam_threshold, set_microglia_ld_pro_inflam_threshold = solara.use_state(5.0)
    
    # ---------------- NEW: neuron transition knobs ----------------
    damage_ratio_thresh, set_damage_ratio_thresh = solara.use_state(0.6)
    healthy_ratio_thresh, set_healthy_ratio_thresh = solara.use_state(0.2)
    healthy_chance, set_healthy_chance = solara.use_state(0.20)
    damage_to_death_ticks, set_damage_to_death_ticks = solara.use_state(20)
    death_ratio_thresh, set_death_ratio_thresh = solara.use_state(0.9)
    death_chance, set_death_chance = solara.use_state(0.25)

    # ---------------- NEW: microglia phenotype signal knobs ----------------
    homeo_signal_thresh, set_homeo_signal_thresh = solara.use_state(1.0)
    pro_inflam_signal_thresh, set_pro_inflam_signal_thresh = solara.use_state(0.2)
    anti_inflam_signal_thresh, set_anti_inflam_signal_thresh = solara.use_state(0.8)
    homeo_chance, set_homeo_chance = solara.use_state(0.2)
    pro_inflam_chance, set_pro_inflam_chance = solara.use_state(0.2)
    anti_inflam_chance, set_anti_inflam_chance = solara.use_state(0.2)
    m2_to_m0_resolution_ticks, set_m2_to_m0_resolution_ticks = solara.use_state(10)

    # ---------------- NEW: astrocyte phenotype signal knobs ----------------
    astro_homeo_signal_thresh, set_astro_homeo_signal_thresh = solara.use_state(0.5)
    astro_pro_inflam_signal_thresh, set_astro_pro_inflam_signal_thresh = solara.use_state(0.15)
    astro_anti_inflam_signal_thresh, set_astro_anti_inflam_signal_thresh = solara.use_state(0.6)
    astro_homeo_chance, set_astro_homeo_chance = solara.use_state(0.25)
    astro_pro_inflam_chance, set_astro_pro_inflam_chance = solara.use_state(0.3)
    astro_anti_inflam_chance, set_astro_anti_inflam_chance = solara.use_state(0.3)

    # ---------------- NEW: field decay ----------------
    pro_decay, set_pro_decay = solara.use_state(0.20)
    anti_decay, set_anti_decay = solara.use_state(0.10)

    # ---------------- NEW: extra lipid dynamics ----------------
    neuron_ld_damage_boost, set_neuron_ld_damage_boost = solara.use_state(0.5)
    neuron_ld_death_boost, set_neuron_ld_death_boost = solara.use_state(1.0)
    neuron_to_astro_ld_transfer_radius, set_neuron_to_astro_ld_transfer_radius = solara.use_state(2)
    astro_ld_to_anti_inflam, set_astro_ld_to_anti_inflam = solara.use_state(0.05)
    microglia_ld_from_phagocytosis, set_microglia_ld_from_phagocytosis = solara.use_state(0.5)
    microglia_ld_pro_inflam_boost, set_microglia_ld_pro_inflam_boost = solara.use_state(0.5)

    stop_mode, set_stop_mode = solara.use_state("Damaged cleared")

    # perf knobs
    fps_limit, set_fps_limit = solara.use_state(8)              # max frames/sec
    steps_per_frame, set_steps_per_frame = solara.use_state(5)  # sim ticks per render

    # ----------- Simulation state ----------->
    model_ref = solara.use_reactive(None)   # holds MicrogliaNeuronModel
    step_count, set_step_count = solara.use_state(0)

    running, set_running = solara.use_state(False)
    stop_ref = solara.use_reactive(False)
    loop_active_ref = solara.use_reactive(False)

    frame_png, set_frame_png = solara.use_state(None)

    show_metrics = solara.use_reactive(True)  # on/off metrics column


    # csv stuff
    state_setters = {
        "steps": set_steps,
        "m0": set_m0,
        "m1": set_m1,
        "m2": set_m2,
        "h_neurons": set_hn,
        "d_neurons": set_dn,
        "eat": set_eat,
        "sense": set_sense,
        "damage": set_damage,
        "ndist": set_ndist,
        "radius": set_radius,
    }
    
    state_setters.update({
        "damage_ratio_thresh": set_damage_ratio_thresh,
        "healthy_ratio_thresh": set_healthy_ratio_thresh,
        "healthy_chance": set_healthy_chance,
        "damage_to_death_ticks": set_damage_to_death_ticks,
        "death_ratio_thresh": set_death_ratio_thresh,
        "death_chance": set_death_chance,

        "homeo_signal_thresh": set_homeo_signal_thresh,
        "pro_inflam_signal_thresh": set_pro_inflam_signal_thresh,
        "anti_inflam_signal_thresh": set_anti_inflam_signal_thresh,
        "homeo_chance": set_homeo_chance,
        "pro_inflam_chance": set_pro_inflam_chance,
        "anti_inflam_chance": set_anti_inflam_chance,
        "m2_to_m0_resolution_ticks": set_m2_to_m0_resolution_ticks,

        "astro_homeo_signal_thresh": set_astro_homeo_signal_thresh,
        "astro_pro_inflam_signal_thresh": set_astro_pro_inflam_signal_thresh,
        "astro_anti_inflam_signal_thresh": set_astro_anti_inflam_signal_thresh,
        "astro_homeo_chance": set_astro_homeo_chance,
        "astro_pro_inflam_chance": set_astro_pro_inflam_chance,
        "astro_anti_inflam_chance": set_astro_anti_inflam_chance,

        "pro_decay": set_pro_decay,
        "anti_decay": set_anti_decay,

        "neuron_ld_damage_boost": set_neuron_ld_damage_boost,
        "neuron_ld_death_boost": set_neuron_ld_death_boost,
        "neuron_to_astro_ld_transfer_radius": set_neuron_to_astro_ld_transfer_radius,
        "astro_ld_to_anti_inflam": set_astro_ld_to_anti_inflam,
        "microglia_ld_from_phagocytosis": set_microglia_ld_from_phagocytosis,
        "microglia_ld_pro_inflam_boost": set_microglia_ld_pro_inflam_boost,
    })    
        # ---------- SAVE helpers ----------
    save_status, set_save_status = solara.use_state("")

    def _current_param_dict() -> dict:
        # what you currently expose in the UI (ICs + knobs)
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "steps_cap": int(steps),
            "stop_mode": str(stop_mode),
            "seed": int(seed),

            # ICs
            "m0": int(m0), "m1": int(m1), "m2": int(m2),
            "h_neurons": int(h_neurons), "d_neurons": int(d_neurons),

            # core knobs
            "eat": float(eat),
            "sense": float(sense),
            "damage": float(damage),
            "ndist": int(ndist),
            "radius": int(radius),

            # astro ICs/knobs
            "a0_astro": int(a0_astro),
            "a1_astro": int(a1_astro),
            "a2_astro": int(a2_astro),
            "astro_radius": int(astro_radius),
            "a2_res": int(a2_res),
            "recruit_thresh": float(recruit_thresh),
            "recruit_prob": float(recruit_prob),

            # lipid knobs
            "neuron_ld_production_rate": float(neuron_ld_production_rate),
            "neuron_ld_packet_size": float(neuron_ld_packet_size),
            "neuron_to_astro_ld_transfer_prob": float(neuron_to_astro_ld_transfer_prob),
            "astro_ld_oxidation_rate": float(astro_ld_oxidation_rate),
            "microglia_ld_pro_inflam_threshold": float(microglia_ld_pro_inflam_threshold),
        }

    def _ensure_save_dir() -> str:
        outdir = os.path.join(os.getcwd(), "snapshots")
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def _make_run_id() -> str:
        # stable-ish naming you can grep later
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{t}_seed{seed}_step{step_count}"

    def save_inflammation_and_csv():
        try:
            # ensure metrics_df reflects the latest model state before saving
            m = model_ref.value
            if m is not None:
                refresh_metrics_from_model(m)

            outdir = _ensure_save_dir()
            run_id = _make_run_id()

            # 1) save CSV (metrics_df)
            df = metrics_df.copy()
            if df is None or df.empty:
                set_save_status("Nothing to save yet (metrics_df is empty).")
                return

            csv_path = os.path.join(outdir, f"{run_id}_metrics.csv")
            df.to_csv(csv_path, index=False)

            # 2) save params as JSON (easy reproducibility)
            params_path = os.path.join(outdir, f"{run_id}_params.json")
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(_current_param_dict(), f, indent=2)

            # 3) save inflammation PNG using matplotlib
            png_path = os.path.join(outdir, f"{run_id}_inflammation.png")

            fig = plt.figure(figsize=(6.5, 3.5), dpi=160)
            ax = fig.add_subplot(1, 1, 1)

            # Ensure derived "total_inflammation" exists
            plot_df = df.copy()
            if "total_inflammation" not in plot_df.columns:
                if "total_pro_inflammation" in plot_df.columns and "total_anti_inflammation" in plot_df.columns:
                    plot_df["total_inflammation"] = (
                        plot_df["total_pro_inflammation"] + plot_df["total_anti_inflammation"]
                    )

            # Plot what exists
            if "total_pro_inflammation" in plot_df.columns:
                ax.plot(plot_df["step"], plot_df["total_pro_inflammation"], label="pro")
            if "total_anti_inflammation" in plot_df.columns:
                ax.plot(plot_df["step"], plot_df["total_anti_inflammation"], label="anti")
            if "total_inflammation" in plot_df.columns:
                ax.plot(plot_df["step"], plot_df["total_inflammation"], label="total")

            ax.set_title(f"Inflammation — step {int(plot_df['step'].iloc[-1])}")
            ax.set_xlabel("step")
            ax.set_ylabel("inflammation")
            ax.legend()
            fig.tight_layout()
            fig.savefig(png_path)
            plt.close(fig)

            set_save_status(f"Saved:\n- {png_path}\n- {csv_path}\n- {params_path}")
        except Exception as e:
            set_save_status(f"Save failed: {type(e).__name__}: {e}")

    # load csv helper
    def load_csv_params(file_info, state_setters):
        int_params = {
            "steps","m0","m1","m2","h_neurons","d_neurons","ndist","radius",
            "damage_to_death_ticks","m2_to_m0_resolution_ticks","neuron_to_astro_ld_transfer_radius",
        }

        df = pd.read_csv(file_info["file_obj"])

        for _, row in df.iterrows():
            name = row["param"]
            value = row["value"]

            if name in state_setters:
                if name in int_params:
                    value = int(value)
                else:
                    value = float(value)
                state_setters[name](value)
            else:
                print(f"Warning: '{name}' is not recognized")

        print("CSV parameters loaded")

  
    # metrics df that feeds charts
    metrics_df, set_metrics_df = solara.use_state(
        pd.DataFrame(columns=[
            "step",
            "healthy_neurons",
            "damaged_neurons",
            "dead_neurons",
            "neurons_total",
            "total_pro_inflammation",
            "total_anti_inflammation",
            "total_inflammation",
            "mean_inflammation",
            "microglia_total",
            "microglia_M0",
            "microglia_M1",
            "microglia_M2",
            # NEW: astrocytes
            "astrocytes_total",
            "astrocytes_A0",
            "astrocytes_A1",
            "astrocytes_A2",
            "total_ld_neurons",
            "total_ld_microglia",
            "total_ld_astrocytes",
        ])
    )

    # ----------- persistent matplotlib fig / ax / image ----------->
    fig_grid, ax_grid, img_artist = solara.use_memo(
        lambda: init_grid_figure(),
        dependencies=[]
    )

    # ----------- model setup / reset ----------->
    def build_model():
        # fresh model at step 0
        model, _ = run_sim(
            steps=0,
            width=33,
            height=33,
            m0=m0,
            m1=m1,
            m2=m2,
            h_neurons=h_neurons,
            d_neurons=d_neurons,
            eat=eat,
            sense=sense,
            damage=damage,
            ndist=ndist,
            radius=radius,
            seed=seed,

            a0_astro=a0_astro,
            a1_astro=a1_astro,
            a2_astro=a2_astro,
            astro_radius=astro_radius,
            a2_res=a2_res,
            recruit_thresh=recruit_thresh,
            recruit_prob=recruit_prob,

            neuron_ld_production_rate=neuron_ld_production_rate,
            neuron_ld_packet_size=neuron_ld_packet_size,
            neuron_to_astro_ld_transfer_prob=neuron_to_astro_ld_transfer_prob,
            astro_ld_oxidation_rate=astro_ld_oxidation_rate,
            microglia_ld_pro_inflam_threshold=microglia_ld_pro_inflam_threshold,

            # NEW
            damage_ratio_thresh=damage_ratio_thresh,
            healthy_ratio_thresh=healthy_ratio_thresh,
            healthy_chance=healthy_chance,
            damage_to_death_ticks=damage_to_death_ticks,
            death_ratio_thresh=death_ratio_thresh,
            death_chance=death_chance,

            homeo_signal_thresh=homeo_signal_thresh,
            pro_inflam_signal_thresh=pro_inflam_signal_thresh,
            anti_inflam_signal_thresh=anti_inflam_signal_thresh,
            homeo_chance=homeo_chance,
            pro_inflam_chance=pro_inflam_chance,
            anti_inflam_chance=anti_inflam_chance,
            m2_to_m0_resolution_ticks=m2_to_m0_resolution_ticks,

            astro_homeo_signal_thresh=astro_homeo_signal_thresh,
            astro_pro_inflam_signal_thresh=astro_pro_inflam_signal_thresh,
            astro_anti_inflam_signal_thresh=astro_anti_inflam_signal_thresh,
            astro_homeo_chance=astro_homeo_chance,
            astro_pro_inflam_chance=astro_pro_inflam_chance,
            astro_anti_inflam_chance=astro_anti_inflam_chance,

            pro_decay=pro_decay,
            anti_decay=anti_decay,

            neuron_ld_damage_boost=neuron_ld_damage_boost,
            neuron_ld_death_boost=neuron_ld_death_boost,
            neuron_to_astro_ld_transfer_radius=neuron_to_astro_ld_transfer_radius,
            astro_ld_to_anti_inflam=astro_ld_to_anti_inflam,
            microglia_ld_from_phagocytosis=microglia_ld_from_phagocytosis,
            microglia_ld_pro_inflam_boost=microglia_ld_pro_inflam_boost,
        )

        model_ref.value = model
        set_step_count(0)

        # draw first frame
        render_frame(model)

        # seed charts with an initial snapshot row, even before first .step()
        seed_metrics_from_model(model)

    # run once on mount
    solara.use_effect(build_model, [])

    # ----------- stopping conditions ----------->
    def should_stop(m: MicrogliaNeuronModel) -> bool:
        if stop_mode == "Damaged cleared":
            if not any((n.pos is not None) and n.damaged for n in m.neurons):
                return True
        elif stop_mode == "Inflammation is zero":
            if int((m.pro_inflam_val + m.anti_inflam_val).sum()) == 0:
                return True
        elif stop_mode == "No neurons left":
            if all(n.pos is None for n in m.neurons):
                return True
        elif stop_mode == "No microglia left":
            if len(m.microglia) == 0:
                return True
        if m.steps >= steps:
            return True
        return False

    # ----------- metrics update helpers ----------->
    def compute_full_metrics_df(m: MicrogliaNeuronModel) -> pd.DataFrame:
        """
        Pull the current DataCollector dataframe and return a fresh DataFrame
        aligned with the plotting expectations.
        """
        if m is None:
            return pd.DataFrame(columns=[
                "step",
                "healthy_neurons",
                "damaged_neurons",
                "dead_neurons",
                "neurons_total",
                "total_pro_inflammation",
                "total_anti_inflammation",
                "total_inflammation",
                "mean_inflammation",
                "microglia_total",
                "microglia_M0",
                "microglia_M1",
                "microglia_M2",
                "astrocytes_total",
                "astrocytes_A0",
                "astrocytes_A1",
                "astrocytes_A2",
                "total_ld_neurons",
                "total_ld_microglia",
                "total_ld_astrocytes",
            ])

        df_full = m.datacollector.get_model_vars_dataframe()

        if df_full.shape[0] == 0:
            # fabricate one row from current model state so charts aren't blank
            astro_list = getattr(m, "astrocytes", [])
            fabricated = pd.DataFrame([{
                "step": m.steps,
                "healthy_neurons": len([n for n in m.neurons if (n.pos is not None and n.healthy)]),
                "damaged_neurons": len([n for n in m.neurons if (n.pos is not None and n.damaged)]),
                "dead_neurons": len([n for n in m.neurons if (n.pos is not None and n.dead)]),
                "neurons_total": len([n for n in m.neurons if n.pos is not None]),
                "total_pro_inflammation": float(np.sum(m.pro_inflam_val)),
                "total_anti_inflammation": float(np.sum(m.anti_inflam_val)),
                "total_inflammation": float(np.sum(m.pro_inflam_val + m.anti_inflam_val)),
                "mean_inflammation": float(np.mean(m.pro_inflam_val + m.anti_inflam_val)),
                "microglia_total": len([mg for mg in m.microglia if mg.pos is not None]),
                "microglia_M0": len([mg for mg in m.microglia if mg.pos is not None and mg.phenotype == "M0"]),
                "microglia_M1": len([mg for mg in m.microglia if mg.pos is not None and mg.phenotype == "M1"]),
                "microglia_M2": len([mg for mg in m.microglia if mg.pos is not None and mg.phenotype == "M2"]),
                # astrocytes
                "astrocytes_total": len([a for a in astro_list if a.pos is not None]),
                "astrocytes_A0": len([a for a in astro_list if a.pos is not None and getattr(a, "phenotype", None) == "A0"]),
                "astrocytes_A1": len([a for a in astro_list if a.pos is not None and getattr(a, "phenotype", None) == "A1"]),
                "astrocytes_A2": len([a for a in astro_list if a.pos is not None and getattr(a, "phenotype", None) == "A2"]),
                # lipids
                "total_ld_neurons": float(
                    np.mean([getattr(n, "lipid_droplets", 0.0) for n in m.neurons if n.pos is not None]) 
                    if len([n for n in m.neurons if n.pos is not None]) > 0 else 0.0
                ),
                "total_ld_microglia": float(
                    np.mean([getattr(mg, "lipid_droplets", 0.0) for mg in m.microglia if mg.pos is not None]) 
                    if len([mg for mg in m.microglia if mg.pos is not None]) > 0 else 0.0
                ),
                "total_ld_astrocytes": float(
                    np.mean([getattr(a, "lipid_pool", 0.0) for a in astro_list if a.pos is not None]) 
                    if len([a for a in astro_list if a.pos is not None]) > 0 else 0.0
                ),

            }])
            return fabricated.reset_index(drop=True)

        df_full = df_full.copy()

        # ensure basic neuron columns exist
        for col in ["healthy_neurons", "damaged_neurons", "dead_neurons"]:
            if col not in df_full.columns:
                df_full[col] = 0
                
        # ensure lipid columns exist (fallback to 0 if not collected)
        for col in ["total_ld_neurons", "total_ld_microglia", "total_ld_astrocytes"]:
            if col not in df_full.columns:
                df_full[col] = 0.0

        if "neurons_total" not in df_full.columns:
            # fallback: derive from components if available
            if all(c in df_full.columns for c in ["healthy_neurons", "damaged_neurons", "dead_neurons"]):
                df_full["neurons_total"] = (
                    df_full["healthy_neurons"]
                    + df_full["damaged_neurons"]
                    + df_full["dead_neurons"]
                )
            else:
                df_full["neurons_total"] = 0

        # total pro/anti inflammation
        if "total_pro_inflammation" not in df_full.columns and "total_pro_inflammation" in df_full.columns:
            pass  # no-op, kept for symmetry
        if "total_anti_inflammation" not in df_full.columns and "total_anti_inflammation" in df_full.columns:
            pass

        # total/mean inflammation = pro + anti
        if "total_inflammation" not in df_full.columns:
            if "total_pro_inflammation" in df_full.columns and "total_anti_inflammation" in df_full.columns:
                df_full["total_inflammation"] = (
                    df_full["total_pro_inflammation"] + df_full["total_anti_inflammation"]
                )
        if "mean_inflammation" not in df_full.columns:
            if "mean_pro_inflammation" in df_full.columns and "mean_anti_inflammation" in df_full.columns:
                df_full["mean_inflammation"] = (
                    df_full["mean_pro_inflammation"] + df_full["mean_anti_inflammation"]
                )

        # microglia_total convenience
        if "microglia_total" not in df_full.columns:
            if "microglia_total" in df_full.columns:
                pass
            elif "microglia" in df_full.columns:
                df_full["microglia_total"] = df_full["microglia"]
            else:
                df_full["microglia_total"] = 0

        # ensure microglia phenotype columns exist
        for col in ["microglia_M0", "microglia_M1", "microglia_M2"]:
            if col not in df_full.columns:
                df_full[col] = 0

        # ensure astrocyte columns exist
        for col in ["astrocytes_total", "astrocytes_A0", "astrocytes_A1", "astrocytes_A2"]:
            if col not in df_full.columns:
                df_full[col] = 0

        return df_full.reset_index(drop=True)

    def seed_metrics_from_model(m: MicrogliaNeuronModel):
        df_now = compute_full_metrics_df(m)
        set_metrics_df(df_now.copy())

    def refresh_metrics_from_model(m: MicrogliaNeuronModel):
        df_now = compute_full_metrics_df(m)
        set_metrics_df(df_now.copy())

    # ----------- figure/PNG update ----------->
    def render_frame(m: MicrogliaNeuronModel):
        """
        Update the persistent matplotlib Figure and push PNG bytes to frame_png.
        This is what drives the visual grid.
        """
        if m is None:
            return

        # heatmap: pro-inflammation only (you can change to pro+anti if you want)
        heatmap = m.pro_inflam_val.T
        img_artist.set_data(heatmap)

        h_map, w_map = heatmap.shape
        img_artist.set_extent((0, w_map, 0, h_map))
        ax_grid.set_xlim(0, w_map)
        ax_grid.set_ylim(0, h_map)
        ax_grid.set_aspect("equal")

        # clear previous agent markers
        for artist in list(ax_grid.lines):
            artist.remove()

        # draw neurons
        for n in m.neurons:
            if n.pos is None:
                continue
            x, y = n.pos
            # color by state: healthy, damaged, dead
            if n.dead:
                color = "black"
            elif n.damaged:
                color = "red"
            else:
                color = "limegreen"
            ax_grid.plot(
                x, y,
                marker="s", ms=5, mec="black",
                mfc=color,
            )

        # draw microglia by phenotype
        for mg in m.microglia:
            if mg.pos is None:
                continue
            x, y = mg.pos
            if mg.phenotype == "M0":
                color = "blue"
            elif mg.phenotype == "M1":
                color = "orange"
            else:  # M2
                color = "purple"
            ax_grid.plot(
                x, y,
                marker="o", ms=5, mec="black", mfc=color,
            )

        # NEW: draw astrocytes by phenotype
        astro_list = getattr(m, "astrocytes", [])
        for a in astro_list:
            if a.pos is None:
                continue
            x, y = a.pos
            pheno = getattr(a, "phenotype", None)
            if pheno == "A0":
                color = "cyan"
            elif pheno == "A1":
                color = "magenta"
            elif pheno == "A2":
                color = "yellow"
            else:
                color = "white"  # unknown/other
            ax_grid.plot(
                x, y,
                marker="^", ms=5, mec="black", mfc=color,
            )

        ax_grid.set_title(f"Step {m.steps}")

        # rasterize -> PNG bytes
        buf = io.BytesIO()
        canvas = FigureCanvas(fig_grid)
        canvas.draw()
        fig_grid.savefig(buf, format="png", dpi=80)
        set_frame_png(buf.getvalue())

    # ----------- helpers for seed ----------->
    def randomize_seed():
        new_seed = random.randint(0, 2**8 - 1)
        set_seed(new_seed)
        on_reset()

    def copy_seed():
        pyperclip.copy(str(seed))

    # ----------- helper for resizing figure dynamically ----------->
    def resize_grid_figure():
        if show_metrics.value:
            fig_grid.set_size_inches(4, 4)  # normal size when metrics are visible
        else:
            fig_grid.set_size_inches(8, 8)  # bigger when metrics hidden
        fig_grid.tight_layout()

    # ----------- main sim loop ----------->
    @use_task
    def runner():
        if loop_active_ref.value:
            return
        loop_active_ref.value = True

        while True:
            if stop_ref.value:
                break

            m = model_ref.value
            if m is None:
                break

            if should_stop(m):
                break

            # advance the sim multiple ticks before we redraw
            for _ in range(steps_per_frame):
                if stop_ref.value or should_stop(m):
                    break
                m.step()

            # update UI-visible state
            set_step_count(m.steps)
            render_frame(m)
            refresh_metrics_from_model(m)

            # throttle FPS
            if fps_limit > 0:
                time.sleep(1.0 / float(fps_limit))

            if not running:
                break

        loop_active_ref.value = False
        stop_ref.value = False
        if running:
            set_running(False)

    # ----------- button handlers ----------->
    def on_reset():
        set_running(False)
        stop_ref.value = True
        loop_active_ref.value = False
        build_model()

    def on_start():
        if loop_active_ref.value:
            return
        stop_ref.value = False
        set_running(True)
        runner()

    def on_stop():
        set_running(False)
        stop_ref.value = True

    def step_once():
        m = model_ref.value
        if m is None:
            return
        if should_stop(m):
            return
        # single step for manual advance
        m.step()
        set_step_count(m.steps)
        render_frame(m)
        refresh_metrics_from_model(m)
    
    def chart_lipids(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()

        cols = ["step", "total_ld_neurons", "total_ld_microglia", "total_ld_astrocytes"]
        cols = [c for c in cols if c in df.columns]
        if len(cols) <= 1:  # only "step" present
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()

        tidy = pd.melt(
            df[cols],
            id_vars="step",
            var_name="type",
            value_name="value",
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("value:Q", title="Mean lipid droplets"),
                color=alt.Color("type:N", title="Cell type"),
            )
            .properties(title="Lipid droplets (environment totals)", width=350, height=180)
        )

    # ----------- Altair charts ----------->
    def chart_neurons(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        cols = ["step", "healthy_neurons", "damaged_neurons", "dead_neurons", "neurons_total"]
        cols = [c for c in cols if c in df.columns]
        tidy = pd.melt(
            df[cols],
            id_vars="step", var_name="type", value_name="value"
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("value:Q", title="Count"),
                color=alt.Color("type:N", title="Neuron state"),
            )
            .properties(title="Neurons", width=350, height=180)
        )

    def chart_inflammation(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        cols = [
            "step",
            "total_pro_inflammation",
            "total_anti_inflammation",
            "total_inflammation",
        ]
        cols = [c for c in cols if c in df.columns]
        tidy = pd.melt(
            df[cols],
            id_vars="step", var_name="type", value_name="value"
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("value:Q", title="Inflammation (total)"),
                color=alt.Color("type:N", title="Field"),
            )
            .properties(title="Inflammation (pro vs anti vs total)", width=350, height=180)
        )

    def chart_microglia(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        cols = ["step", "microglia_total", "microglia_M0", "microglia_M1", "microglia_M2"]
        cols = [c for c in cols if c in df.columns]
        tidy = pd.melt(
            df[cols],
            id_vars="step",
            var_name="type",
            value_name="value",
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("value:Q", title="# Microglia"),
                color=alt.Color("type:N", title="Type"),
            )
            .properties(title="Microglia (total + phenotypes)", width=350, height=180)
        )

    # Astrocyte chart
    def chart_astrocytes(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        cols = ["step", "astrocytes_total", "astrocytes_A0", "astrocytes_A1", "astrocytes_A2"]
        cols = [c for c in cols if c in df.columns]
        if len(cols) <= 1:  # only "step" present
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        tidy = pd.melt(
            df[cols],
            id_vars="step",
            var_name="type",
            value_name="value",
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("value:Q", title="# Astrocytes"),
                color=alt.Color("type:N", title="Type"),
            )
            .properties(title="Astrocytes (total + phenotypes)", width=350, height=180)
        )

    # ----------- UI Layout ----------->
    with solara.Columns([1, 1]):
        # LEFT: controls + grid
        with solara.Column():
            with solara.Card("Parameters"):
                value_style = {
                    "width": "6ch",
                    "fontFamily": "monospace",
                    "textAlign": "center",
                    "paddingRight": "8px",
                }

                def row_with_value(val, widget_callable):
                    with solara.Row():
                        solara.Text(f"{val}", style=value_style)
                        widget_callable()


                row_with_value(steps, lambda: solara.SliderInt("steps (max)", value=steps, min=50, max=5000, on_value=set_steps))
                row_with_value(m0,   lambda: solara.SliderInt("M0 microglia (homeostatic)", value=m0, min=0, max=50, on_value=set_m0))
                row_with_value(m1,   lambda: solara.SliderInt("M1 microglia (pro-inflam)", value=m1, min=0, max=50, on_value=set_m1))
                row_with_value(m2,   lambda: solara.SliderInt("M2 microglia (anti-inflam)", value=m2, min=0, max=50, on_value=set_m2))
                row_with_value(h_neurons, lambda: solara.SliderInt("healthy neurons", value=h_neurons, min=0, max=200, on_value=set_hn))
                row_with_value(d_neurons, lambda: solara.SliderInt("damaged neurons", value=d_neurons, min=0, max=200, on_value=set_dn))
                row_with_value(eat, lambda: solara.SliderFloat("eat prob", value=eat, min=0.0, max=1.0, on_value=set_eat))
                row_with_value(sense, lambda: solara.SliderFloat("sense", value=sense, min=0.0, max=1.0, on_value=set_sense))
                row_with_value(damage, lambda: solara.SliderFloat("damage chance", value=damage, min=0.0, max=0.1, step=0.001, on_value=set_damage))
                row_with_value(ndist, lambda: solara.SliderInt("neuron link dist", value=ndist, min=1, max=10, on_value=set_ndist))
                row_with_value(radius, lambda: solara.SliderInt("inflam radius", value=radius, min=1, max=10, on_value=set_radius))
                row_with_value(seed, lambda: solara.InputInt("set seed", value=seed, on_value=set_seed))
                
                        

                # astrocyte parameter sliders
                row_with_value(a0_astro, lambda: solara.SliderInt("A0 astrocytes", value=a0_astro, min=0, max=200, on_value=set_a0_astro))
                row_with_value(a1_astro, lambda: solara.SliderInt("A1 astrocytes", value=a1_astro, min=0, max=200, on_value=set_a1_astro))
                row_with_value(a2_astro, lambda: solara.SliderInt("A2 astrocytes", value=a2_astro, min=0, max=200, on_value=set_a2_astro))
                row_with_value(astro_radius, lambda: solara.SliderInt("astro coverage radius", value=astro_radius, min=1, max=10, on_value=set_astro_radius))
                row_with_value(a2_res, lambda: solara.SliderInt("A2 → A0 resolution ticks", value=a2_res, min=1, max=100, on_value=set_a2_res))
                row_with_value(recruit_thresh, lambda: solara.SliderFloat("A1 recruit threshold", value=recruit_thresh, min=0.0, max=10.0, step=0.1, on_value=set_recruit_thresh))
                row_with_value(recruit_prob, lambda: solara.SliderFloat("A1 → M1 recruit prob", value=recruit_prob, min=0.0, max=0.5, step=0.01, on_value=set_recruit_prob))
                
                # lipid parameter slides
                row_with_value(neuron_ld_production_rate, lambda: solara.SliderFloat(
                    "neuron LD production (damaged)", value=neuron_ld_production_rate,
                    min=0.0, max=1.0, step=0.01, on_value=set_neuron_ld_production_rate
                ))
                row_with_value(neuron_ld_packet_size, lambda: solara.SliderFloat(
                    "LD packet size neuron→astro", value=neuron_ld_packet_size,
                    min=0.0, max=5.0, step=0.1, on_value=set_neuron_ld_packet_size
                ))
                row_with_value(neuron_to_astro_ld_transfer_prob, lambda: solara.SliderFloat(
                    "neuron→astro LD transfer prob", value=neuron_to_astro_ld_transfer_prob,
                    min=0.0, max=1.0, step=0.01, on_value=set_neuron_to_astro_ld_transfer_prob
                ))
                row_with_value(astro_ld_oxidation_rate, lambda: solara.SliderFloat(
                    "astro LD oxidation rate", value=astro_ld_oxidation_rate,
                    min=0.0, max=1.0, step=0.01, on_value=set_astro_ld_oxidation_rate
                ))
                row_with_value(microglia_ld_pro_inflam_threshold, lambda: solara.SliderFloat(
                    "microglia LD inflam threshold", value=microglia_ld_pro_inflam_threshold,
                    min=0.0, max=50.0, step=0.5, on_value=set_microglia_ld_pro_inflam_threshold
                ))
                # ---------------- NEW: neuron transition sliders ----------------
                row_with_value(damage_ratio_thresh, lambda: solara.SliderFloat(
                    "damage ratio thresh (pro/(anti+eps))", value=damage_ratio_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_damage_ratio_thresh
                ))
                row_with_value(healthy_ratio_thresh, lambda: solara.SliderFloat(
                    "healthy ratio thresh", value=healthy_ratio_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_healthy_ratio_thresh
                ))
                row_with_value(healthy_chance, lambda: solara.SliderFloat(
                    "healthy chance", value=healthy_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_healthy_chance
                ))
                row_with_value(damage_to_death_ticks, lambda: solara.SliderInt(
                    "damage→death ticks", value=damage_to_death_ticks,
                    min=1, max=200, on_value=set_damage_to_death_ticks
                ))
                row_with_value(death_ratio_thresh, lambda: solara.SliderFloat(
                    "death ratio thresh", value=death_ratio_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_death_ratio_thresh
                ))
                row_with_value(death_chance, lambda: solara.SliderFloat(
                    "death chance", value=death_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_death_chance
                ))

                # ---------------- NEW: microglia phenotype signal sliders ----------------
                row_with_value(homeo_signal_thresh, lambda: solara.SliderFloat(
                    "microglia homeo signal thresh", value=homeo_signal_thresh,
                    min=0.0, max=10.0, step=0.1, on_value=set_homeo_signal_thresh
                ))
                row_with_value(pro_inflam_signal_thresh, lambda: solara.SliderFloat(
                    "microglia pro ratio thresh", value=pro_inflam_signal_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_pro_inflam_signal_thresh
                ))
                row_with_value(anti_inflam_signal_thresh, lambda: solara.SliderFloat(
                    "microglia anti ratio thresh", value=anti_inflam_signal_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_anti_inflam_signal_thresh
                ))
                row_with_value(homeo_chance, lambda: solara.SliderFloat(
                    "microglia homeo chance", value=homeo_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_homeo_chance
                ))
                row_with_value(pro_inflam_chance, lambda: solara.SliderFloat(
                    "microglia pro chance", value=pro_inflam_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_pro_inflam_chance
                ))
                row_with_value(anti_inflam_chance, lambda: solara.SliderFloat(
                    "microglia anti chance", value=anti_inflam_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_anti_inflam_chance
                ))
                row_with_value(m2_to_m0_resolution_ticks, lambda: solara.SliderInt(
                    "M2→M0 resolution ticks", value=m2_to_m0_resolution_ticks,
                    min=1, max=200, on_value=set_m2_to_m0_resolution_ticks
                ))

                # ---------------- NEW: astrocyte phenotype signal sliders ----------------
                row_with_value(astro_homeo_signal_thresh, lambda: solara.SliderFloat(
                    "astro homeo signal thresh", value=astro_homeo_signal_thresh,
                    min=0.0, max=10.0, step=0.1, on_value=set_astro_homeo_signal_thresh
                ))
                row_with_value(astro_pro_inflam_signal_thresh, lambda: solara.SliderFloat(
                    "astro pro ratio thresh", value=astro_pro_inflam_signal_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_astro_pro_inflam_signal_thresh
                ))
                row_with_value(astro_anti_inflam_signal_thresh, lambda: solara.SliderFloat(
                    "astro anti ratio thresh", value=astro_anti_inflam_signal_thresh,
                    min=0.0, max=5.0, step=0.05, on_value=set_astro_anti_inflam_signal_thresh
                ))
                row_with_value(astro_homeo_chance, lambda: solara.SliderFloat(
                    "astro homeo chance", value=astro_homeo_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_astro_homeo_chance
                ))
                row_with_value(astro_pro_inflam_chance, lambda: solara.SliderFloat(
                    "astro pro chance", value=astro_pro_inflam_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_astro_pro_inflam_chance
                ))
                row_with_value(astro_anti_inflam_chance, lambda: solara.SliderFloat(
                    "astro anti chance", value=astro_anti_inflam_chance,
                    min=0.0, max=1.0, step=0.01, on_value=set_astro_anti_inflam_chance
                ))

                # ---------------- NEW: field decay sliders ----------------
                row_with_value(pro_decay, lambda: solara.SliderFloat(
                    "pro field decay", value=pro_decay,
                    min=0.0, max=1.0, step=0.01, on_value=set_pro_decay
                ))
                row_with_value(anti_decay, lambda: solara.SliderFloat(
                    "anti field decay", value=anti_decay,
                    min=0.0, max=1.0, step=0.01, on_value=set_anti_decay
                ))

                # ---------------- NEW: extra lipid dynamics sliders ----------------
                row_with_value(neuron_ld_damage_boost, lambda: solara.SliderFloat(
                    "neuron LD damage boost", value=neuron_ld_damage_boost,
                    min=0.0, max=5.0, step=0.05, on_value=set_neuron_ld_damage_boost
                ))
                row_with_value(neuron_ld_death_boost, lambda: solara.SliderFloat(
                    "neuron LD death boost", value=neuron_ld_death_boost,
                    min=0.0, max=10.0, step=0.1, on_value=set_neuron_ld_death_boost
                ))
                row_with_value(neuron_to_astro_ld_transfer_radius, lambda: solara.SliderInt(
                    "neuron→astro LD transfer radius", value=neuron_to_astro_ld_transfer_radius,
                    min=0, max=10, on_value=set_neuron_to_astro_ld_transfer_radius
                ))
                row_with_value(astro_ld_to_anti_inflam, lambda: solara.SliderFloat(
                    "astro LD→anti conversion", value=astro_ld_to_anti_inflam,
                    min=0.0, max=1.0, step=0.01, on_value=set_astro_ld_to_anti_inflam
                ))
                row_with_value(microglia_ld_from_phagocytosis, lambda: solara.SliderFloat(
                    "microglia LD from phagocytosis", value=microglia_ld_from_phagocytosis,
                    min=0.0, max=5.0, step=0.05, on_value=set_microglia_ld_from_phagocytosis
                ))
                row_with_value(microglia_ld_pro_inflam_boost, lambda: solara.SliderFloat(
                    "microglia LD→pro boost factor", value=microglia_ld_pro_inflam_boost,
                    min=0.0, max=5.0, step=0.05, on_value=set_microglia_ld_pro_inflam_boost
                ))

                row_with_value(seed, lambda: solara.InputInt("set seed", value=seed, on_value=set_seed))

                solara.Select(
                    label="Stop when",
                    value=stop_mode,
                    values=[
                        "Damaged cleared",
                        "Tick cap only",
                        "Inflammation is zero",
                        "No neurons left",
                        "No microglia left",
                    ],
                    on_value=set_stop_mode,
                )

                row_with_value(fps_limit, lambda: solara.SliderInt("FPS limit", value=fps_limit, min=1, max=30, on_value=set_fps_limit))
                row_with_value(steps_per_frame, lambda: solara.SliderInt("Steps per frame", value=steps_per_frame, min=1, max=50, on_value=set_steps_per_frame))

                with solara.Row(style={"padding-bottom": "20px"}):
                    solara.Button("Reset", on_click=on_reset, color="secondary")
                    solara.Button("Start", on_click=on_start, color="primary", disabled=loop_active_ref.value)
                    solara.Button("Stop", on_click=on_stop, color="warning", disabled=not loop_active_ref.value)
                
                with solara.Row(style={"padding-bottom": "10px"}):
                    solara.Button("Save inflammation + CSV", on_click=save_inflammation_and_csv, color="success")

                with solara.Row(style={"padding-bottom": "20px"}):
                    solara.Button("Step once", on_click=step_once)
                    solara.Button(
                        "Hide metrics" if show_metrics.value else "Show metrics",
                        on_click=lambda: (
                            show_metrics.set(not show_metrics.value),
                            resize_grid_figure(),
                            render_frame(model_ref.value)
                        ),
                        color="secondary"
                    )

                with solara.Row(style={"padding-bottom": "20px"}):
                    solara.Button("Copy Seed", on_click=copy_seed)
                    solara.Button("Randomize Seed", on_click=randomize_seed)

                with solara.Row():
                    solara.FileDrop("Load CSV", on_file=lambda f: load_csv_params(f, state_setters), lazy=True)




                solara.Text(f"Current step: {step_count}")
                
                if save_status:
                    solara.Markdown(f"```\n{save_status}\n```")

            if frame_png:
                solara.Image(frame_png, format="png", width="100%")

        # RIGHT: charts
        if show_metrics.value:
            with solara.Column():
                with solara.Card("Metrics (live)"):
                    solara.FigureAltair(chart_neurons(metrics_df))
                    solara.FigureAltair(chart_inflammation(metrics_df))
                    solara.FigureAltair(chart_microglia(metrics_df))
                    # NEW astrocyte chart
                    solara.FigureAltair(chart_astrocytes(metrics_df))
                    solara.FigureAltair(chart_lipids(metrics_df))

# ----------- helper for persistent figure ----------->
def init_grid_figure():
    """
    Create and return a persistent matplotlib Figure, Axes, and base image artist.
    We'll mutate these instead of creating a new figure per frame.
    """
    fig = Figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    init_data = np.zeros((10, 10))
    img_artist = ax.imshow(init_data.T, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Step 0")
    fig.tight_layout()

    return fig, ax, img_artist


# Only auto-run CLI when executed directly, not when Solara imports us
# if __name__ == "__main__" and not os.environ.get("SOLARA_SERVER"):
#     main()

Page()