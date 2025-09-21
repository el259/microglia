# app.py
from __future__ import annotations
import argparse
import os
import io
import time
import numpy as np
import pandas as pd
import altair as alt
import solara
from solara.lab import use_task

# Keep pyplot only for CLI plotting; we won't use it in the live viewer.
import matplotlib.pyplot as plt  # CLI path
from matplotlib.figure import Figure               # OO API for live frames
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from model import MicrogliaNeuronModel


# ----------------- Core simulation helpers -----------------
def run_sim(steps=500, width=33, height=33, microglia=5, h_neurons=10, d_neurons=10,
            eat=0.70, sense=0.50, damage=0.005, ndist=5, temp=37.0, radius=3, seed=None):
    model = MicrogliaNeuronModel(
        width=width,
        height=height,
        init_microglia=microglia,
        init_h_neuron=h_neurons,
        init_d_neuron=d_neurons,
        eat_probability=eat,
        sensing_efficiency=sense,
        damage_chance=damage,
        neuron_distance=ndist,
        temperature=temp,
        inflam_radius=radius,
        seed=seed,
    )
    for _ in range(steps):
        # NetLogo-like: stop if domain condition hits, else cap by steps
        if model.all_damaged_cleared():
            break
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    return model, df


def main():
    ap = argparse.ArgumentParser(description="Microglia–Neuron ABM")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--width", type=int, default=33)
    ap.add_argument("--height", type=int, default=33)
    ap.add_argument("--microglia", type=int, default=5)
    ap.add_argument("--h_neurons", type=int, default=10)
    ap.add_argument("--d_neurons", type=int, default=10)
    ap.add_argument("--eat", type=float, default=0.70)
    ap.add_argument("--sense", type=float, default=0.50)
    ap.add_argument("--damage", type=float, default=0.005)
    ap.add_argument("--ndist", type=int, default=5)
    ap.add_argument("--temp", type=float, default=37.0)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    args, _ = ap.parse_known_args()

    _, df = run_sim(**vars(args))

    print("\nFinal snapshot:")
    print(df.tail(1).to_string(index=False))
    print(f"Stopped at tick {int(df['step'].iloc[-1])}")

    # One-off CLI plot with pyplot; CLOSE afterwards to avoid figure buildup.
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(df["step"], df["healthy_neurons"], label="healthy")
    axs[0].plot(df["step"], df["damaged_neurons"], label="damaged")
    axs[0].set_title("Neurons"); axs[0].legend()

    axs[1].plot(df["step"], df["total_inflammation"], label="total")
    axs[1].plot(df["step"], df["mean_inflammation"], label="mean")
    axs[1].set_title("Inflammation"); axs[1].legend()

    axs[2].plot(df["step"], df["microglia"], label="microglia")
    axs[2].set_title("Agents"); axs[2].legend()

    plt.tight_layout()
    if os.environ.get("SOLARA_SERVER"):
        plt.savefig("summary.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ----------------- Solara UI (with LIVE view + metrics + NetLogo-like stopping) -----------------
@solara.component
def Page():
    solara.Title("Microglia–Neuron ABM — Live")

    # Controls (reactive state)
    steps, set_steps = solara.use_state(5000)  # tick cap (NetLogo 'ticks >= max-ticks [ stop ]')
    microglia, set_microglia = solara.use_state(5)
    h_neurons, set_hn = solara.use_state(10)
    d_neurons, set_dn = solara.use_state(10)
    eat, set_eat = solara.use_state(0.70)
    sense, set_sense = solara.use_state(0.50)
    damage, set_damage = solara.use_state(0.005)
    ndist, set_ndist = solara.use_state(5)
    temp, set_temp = solara.use_state(37.0)
    radius, set_radius = solara.use_state(3)

    # Stop-mode selector: mirrors typical NetLogo 'stop' patterns
    stop_mode, set_stop_mode = solara.use_state("Damaged cleared")  # default domain stop

    # Simulation state
    model_ref = solara.use_reactive(None)   # holds MicrogliaNeuronModel
    step_count, set_step_count = solara.use_state(0)
    running, set_running = solara.use_state(False)
    frame_png, set_frame_png = solara.use_state(None)

    # Metrics (Altair) state – we keep a tidy dataframe that updates each tick
    metrics_df, set_metrics_df = solara.use_state(pd.DataFrame())

    def build_model():
        model, _ = run_sim(steps=0, width=33, height=33,
                           microglia=microglia, h_neurons=h_neurons, d_neurons=d_neurons,
                           eat=eat, sense=sense, damage=damage, ndist=ndist, temp=temp, radius=radius)
        model_ref.value = model
        set_step_count(0)
        # reset metrics to empty
        set_metrics_df(pd.DataFrame(columns=[
            "step", "healthy_neurons", "damaged_neurons",
            "total_inflammation", "mean_inflammation", "microglia"
        ]))
        render_frame()
        update_metrics_chart_data()

    # ---- NetLogo-like stopping conditions ----
    def should_stop() -> bool:
        m = model_ref.value
        if m is None:
            return True
        # Domain conditions first (like 'if <condition> [ stop ]' at top of 'go')
        if stop_mode == "Damaged cleared":
            if m.all_damaged_cleared():
                return True
        elif stop_mode == "Inflammation is zero":
            if int(m.inflam_val.sum()) == 0:
                return True
        elif stop_mode == "No neurons left":
            if all(n.pos is None for n in m.neurons):
                return True
        elif stop_mode == "No microglia left":
            if len(m.microglia) == 0:
                return True
        # Tick cap
        if m.steps >= steps:
            return True
        return False

    def render_frame():
        """Use OO Matplotlib to avoid pyplot figure accumulation."""
        m = model_ref.value
        if m is None:
            return

        fig = Figure(figsize=(5, 5))
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)

        # heatmap
        ax.imshow(m.inflam_val.T, origin="lower", interpolation="nearest")

        # overlay agents
        for n in m.neurons:
            if n.pos is None:
                continue
            x, y = n.pos
            ax.plot(x, y, marker="s", ms=5, mec="black",
                    mfc=("red" if n.damaged else "limegreen"))
        for mg in m.microglia:
            if mg.pos is None:
                continue
            x, y = mg.pos
            ax.plot(x, y, marker="o", ms=5, mec="black", mfc="orange")

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Step {m.steps}")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        # Free figure resources
        fig.clear()
        del fig

        set_frame_png(buf.getvalue())

    def update_metrics_chart_data():
        """Pull the current DataCollector dataframe and store in state for Altair."""
        m = model_ref.value
        if m is None:
            return
        # Mesa DataCollector provides this directly. :contentReference[oaicite:2]{index=2}
        df = m.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        set_metrics_df(df)

    @use_task
    def runner():
        # background loop; started by use_effect when 'running' becomes True
        while running and model_ref.value is not None and not should_stop():
            model_ref.value.step()
            set_step_count(model_ref.value.steps)
            # Update both the frame and metrics
            render_frame()
            # For responsiveness, we can update metrics every tick; if heavy, do every N ticks.
            update_metrics_chart_data()
            time.sleep(0.05)  # ~20 FPS-ish
        # auto-stop toggle so Start button re-enables & loop won't restart
        if running and should_stop():
            set_running(False)

    # Build model once, after first render
    solara.use_effect(build_model, [])  # 'setup' after mount. :contentReference[oaicite:3]{index=3}

    # Start/stop the runner based on state, but never from render directly
    def start_runner_if_needed():
        if running and model_ref.value is not None and not should_stop():
            runner()  # schedule/ensure the task is running
    solara.use_effect(start_runner_if_needed, [running, model_ref.value, steps, stop_mode, step_count])  # :contentReference[oaicite:4]{index=4}

    # ------- Build Altair charts from metrics_df (live) -------
    # Altair in Solara: FigureAltair renders Vega-Lite charts reactively. :contentReference[oaicite:5]{index=5}
    def chart_neurons(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        tidy = pd.melt(
            df[["step", "healthy_neurons", "damaged_neurons"]],
            id_vars="step", var_name="type", value_name="value"
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(x="step:Q", y="value:Q", color="type:N")
            .properties(title="Neurons", width=350, height=180)
        )

    def chart_inflammation(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "value": [], "type": []})).mark_line()
        tidy = pd.melt(
            df[["step", "total_inflammation", "mean_inflammation"]],
            id_vars="step", var_name="type", value_name="value"
        )
        return (
            alt.Chart(tidy)
            .mark_line()
            .encode(x="step:Q", y="value:Q", color="type:N")
            .properties(title="Inflammation", width=350, height=180)
        )

    def chart_microglia(df: pd.DataFrame):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step": [], "microglia": []})).mark_line()
        return (
            alt.Chart(df)
            .mark_line()
            .encode(x="step:Q", y="microglia:Q")
            .properties(title="Microglia", width=350, height=180)
        )

    # --------------------------- UI ---------------------------
    with solara.Columns([1, 1]):  # left = grid, right = metrics
        with solara.Column():
            with solara.Card("Parameters"):
                solara.SliderInt("steps (max)", value=steps, min=50, max=5000, on_value=set_steps)
                solara.SliderInt("microglia", value=microglia, min=1, max=50, on_value=set_microglia)
                solara.SliderInt("healthy neurons", value=h_neurons, min=0, max=200, on_value=set_hn)
                solara.SliderInt("damaged neurons", value=d_neurons, min=0, max=200, on_value=set_dn)
                solara.SliderFloat("eat prob", value=eat, min=0.0, max=1.0, on_value=set_eat)
                solara.SliderFloat("sense", value=sense, min=0.0, max=1.0, on_value=set_sense)
                solara.SliderFloat("damage chance", value=damage, min=0.0, max=0.1, step=0.001, on_value=set_damage)
                solara.SliderInt("neuron link dist", value=ndist, min=1, max=10, on_value=set_ndist)
                solara.SliderFloat("temperature (C)", value=temp, min=30.0, max=42.0, step=0.1, on_value=set_temp)
                solara.SliderInt("inflam radius", value=radius, min=1, max=10, on_value=set_radius)

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

                with solara.Row():
                    def on_reset():
                        set_running(False)  # ensure loop halts
                        build_model()

                    solara.Button("Reset", on_click=on_reset, color="secondary")
                    solara.Button("Start", on_click=lambda: set_running(True), color="primary", disabled=running)
                    solara.Button("Stop", on_click=lambda: set_running(False), color="warning", disabled=not running)

                    def step_once():
                        if model_ref.value is None:
                            return
                        if should_stop():
                            return
                        model_ref.value.step()
                        set_step_count(model_ref.value.steps)
                        render_frame()
                        update_metrics_chart_data()

                    solara.Button("Step once", on_click=step_once)

                solara.Text(f"Current step: {step_count}")

            if frame_png:
                solara.Image(frame_png, format="png", width="100%")  # ipywidgets wants CSS strings for width. :contentReference[oaicite:6]{index=6}

        with solara.Column():
            with solara.Card("Metrics (live)"):
                # Render three Altair charts bound to metrics_df
                solara.FigureAltair(chart_neurons(metrics_df))      # neurons (healthy/damaged)
                solara.FigureAltair(chart_inflammation(metrics_df)) # total & mean inflammation
                solara.FigureAltair(chart_microglia(metrics_df))    # microglia count

    # Never call runner() from render; effects start/stop it.


# Only auto-run CLI when executed directly, not when Solara imports us
if __name__ == "__main__" and not os.environ.get("SOLARA_SERVER"):
    main()
