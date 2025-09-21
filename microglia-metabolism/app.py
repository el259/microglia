# app.py
from __future__ import annotations
import argparse, os, io, time
import numpy as np
import pandas as pd
import altair as alt
import solara
from solara.lab import use_task

# Keep pyplot only for CLI plots; live view uses OO Matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from model import MicrogliaMetabolismModel


# ----------------- CLI runner (batch) -----------------
def run_sim(
    steps: int,
    width: int,
    height: int,
    microglia: int,
    plaques: int,
    vessels: int,
    eat: float,
    fortify: float,
    lactate: float,
    glucose: float,
    exercise: str,
    booster: str,
    seed: int | None,
):
    m = MicrogliaMetabolismModel(
        width=width,
        height=height,
        init_microglia=microglia,
        init_plaque=plaques,
        init_vessels=vessels,
        eat_probability=eat,
        fortify_probability=fortify,
        lactate_probability=lactate,
        added_glucose=glucose,
        exercise=exercise,
        metabolic_booster=booster,
        seed=seed,
    )
    for _ in range(steps):
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    return m, df


def main():
    p = argparse.ArgumentParser(description="Microglia metabolism ABM (Mesa) — NetLogo port")
    p.add_argument("--steps", type=int, default=8760, help="ticks (1 tick = 1 hour)")
    p.add_argument("--width", type=int, default=51)
    p.add_argument("--height", type=int, default=51)
    p.add_argument("--microglia", type=int, default=20)
    p.add_argument("--plaques", type=int, default=200)
    p.add_argument("--vessels", type=int, default=60)
    p.add_argument("--eat", type=float, default=0.10)
    p.add_argument("--fortify", type=float, default=0.10)
    p.add_argument("--lactate", type=float, default=0.05)
    p.add_argument("--glucose", type=float, default=500.0)
    p.add_argument("--exercise", type=str, default="none", choices=["none", "moderate", "high"])
    p.add_argument("--booster", type=str, default="off", choices=["off", "daily", "every 2 days", "twice per week", "weekly"])
    p.add_argument("--seed", type=int, default=None)
    args, _ = p.parse_known_args()

    model, df = run_sim(**vars(args))
    print("\nFinal stats:")
    print(df.tail(1).to_string(index=False))
    print("\nTotals:",
          f"plaques={int(np.sum(model.plaque_val==1))},",
          f"perm_lactate_patches={int(np.sum(model.perm_lactate))},",
          f"mean_integrity%={model.global_integrity:.2f},",
          f"pH={model.global_pH:.2f}")

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(df["step"], df["homeostatic"], label="homeostatic")
    axs[0].plot(df["step"], df["m1"], label="m1")
    axs[0].set_title("Microglia phenotypes"); axs[0].set_xlabel("tick (hour)"); axs[0].legend()

    axs[1].plot(df["step"], df["plaques"], label="plaques")
    axs[1].plot(df["step"], df["perm_lactate_patches"], label="perm-lactate")
    axs[1].set_title("Plaques & permanent lactate"); axs[1].set_xlabel("tick"); axs[1].legend()

    axs[2].plot(df["step"], df["global_pH"], label="pH")
    axs[2].plot(df["step"], df["global_integrity"], label="BBB integrity (%)")
    axs[2].plot(df["step"], df["global_glucose"], label="glucose")
    axs[2].set_title("Environment"); axs[2].set_xlabel("tick"); axs[2].legend()

    plt.tight_layout(); plt.show(); plt.close(fig)


# ----------------- Solara UI (live) -----------------
@solara.component
def Page():
    solara.Title("Microglia Metabolism ABM — Live (Solara)")

    # Controls (reactive)
    steps, set_steps = solara.use_state(8760)   # 1 year @ 1h/tick
    width, set_width = solara.use_state(51)
    height, set_height = solara.use_state(51)
    microglia, set_microglia = solara.use_state(20)
    plaques, set_plaques = solara.use_state(200)
    vessels, set_vessels = solara.use_state(60)

    eat, set_eat = solara.use_state(0.10)
    fortify, set_fortify = solara.use_state(0.10)
    lactate, set_lactate = solara.use_state(0.05)
    glucose, set_glucose = solara.use_state(500.0)
    exercise, set_exercise = solara.use_state("none")
    booster, set_booster = solara.use_state("off")

    # stop condition
    stop_mode, set_stop_mode = solara.use_state("Tick cap or domain stop")

    # Simulation state
    model_ref = solara.use_reactive(None)
    step_count, set_step_count = solara.use_state(0)
    running, set_running = solara.use_state(False)
    frame_png, set_frame_png = solara.use_state(None)

    metrics_df, set_metrics_df = solara.use_state(pd.DataFrame())

    # Build/reset model
    def build_model():
        m = MicrogliaMetabolismModel(
            width=width, height=height,
            init_microglia=microglia, init_plaque=plaques, init_vessels=vessels,
            eat_probability=eat, fortify_probability=fortify, lactate_probability=lactate,
            added_glucose=glucose, exercise=exercise, metabolic_booster=booster
        )
        model_ref.value = m
        set_step_count(0)
        set_metrics_df(pd.DataFrame(columns=[
            "step","homeostatic","m1","plaques","perm_lactate_patches",
            "global_pH","global_integrity","global_glucose"
        ]))
        render_frame()
        update_metrics()

    # Stop logic
    def should_stop() -> bool:
        m = model_ref.value
        if m is None: return True
        if stop_mode == "Tick cap or domain stop":
            pass
        elif stop_mode == "No plaques":
            if int(np.sum(m.plaque_val == 1)) == 0:
                return True
        elif stop_mode == "pH >= 7.3":
            if m.global_pH >= 7.3:
                return True
        if m.steps >= steps:
            return True
        return False

    # Render one frame (OO Matplotlib → no pyplot buildup)
    def render_frame():
        m = model_ref.value
        if m is None: return
        fig = Figure(figsize=(5,5)); FigureCanvas(fig); ax = fig.add_subplot(1,1,1)

        # heatmap: lactate intensity
        ax.imshow(m.lactate_val.T, origin="lower", interpolation="nearest")

        # overlay plaques (red squares)
        py, px = np.where(m.plaque_val.T == 1)  # note transpose
        if px.size:
            ax.plot(px, py, "s", ms=3)

        # overlay microglia (homeostatic=green, m1=orange)
        for a in m.microglia:
            if a.pos is None: continue
            x,y = a.pos
            color = "orange" if getattr(a, "phenotype", "homeostatic").lower() == "m1" else "limegreen"
            ax.plot(x, y, "o", ms=4, mec="black", mfc=color)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Step {m.steps}")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); fig.clear(); del fig
        set_frame_png(buf.getvalue())

    # DataCollector → live metrics
    def update_metrics():
        m = model_ref.value
        if m is None: return
        df = m.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        set_metrics_df(df)

    # Background loop
    @use_task
    def runner():
        while running and model_ref.value is not None and not should_stop():
            model_ref.value.step()
            set_step_count(model_ref.value.steps)
            render_frame()
            update_metrics()
            time.sleep(0.05)
        if running and should_stop():
            set_running(False)

    # Effects: build once; start/stop task
    solara.use_effect(build_model, [])
    def maybe_start():
        if running and model_ref.value is not None and not should_stop():
            runner()
    solara.use_effect(maybe_start, [running, model_ref.value, steps, stop_mode, step_count])

    # Charts (Altair)
    def chart_neurons(df):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step":[],"value":[],"type":[]})).mark_line()
        tidy = pd.melt(df[["step","homeostatic","m1"]], id_vars="step",
                       var_name="type", value_name="value")
        return (alt.Chart(tidy).mark_line()
                .encode(x="step:Q", y="value:Q", color="type:N")
                .properties(title="Microglia phenotypes", width=350, height=180))

    def chart_plaque_lactate(df):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step":[],"value":[],"type":[]})).mark_line()
        tidy = pd.melt(df[["step","plaques","perm_lactate_patches"]], id_vars="step",
                       var_name="type", value_name="value")
        return (alt.Chart(tidy).mark_line()
                .encode(x="step:Q", y="value:Q", color="type:N")
                .properties(title="Plaques & permanent lactate", width=350, height=180))

    def chart_env(df):
        if df.empty:
            return alt.Chart(pd.DataFrame({"step":[],"value":[],"type":[]})).mark_line()
        tidy = pd.melt(df[["step","global_pH","global_integrity","global_glucose"]],
                       id_vars="step", var_name="type", value_name="value")
        return (alt.Chart(tidy).mark_line()
                .encode(x="step:Q", y="value:Q", color="type:N")
                .properties(title="Environment", width=350, height=180))

    # UI layout
    with solara.Columns([1,1]):
        with solara.Column():
            with solara.Card("Parameters"):
                solara.SliderInt("steps (max; hours)", value=steps, min=24, max=20000, on_value=set_steps)
                solara.SliderInt("width", value=width, min=21, max=101, on_value=set_width)
                solara.SliderInt("height", value=height, min=21, max=101, on_value=set_height)
                solara.SliderInt("microglia", value=microglia, min=1, max=200, on_value=set_microglia)
                solara.SliderInt("plaques", value=plaques, min=0, max=5000, on_value=set_plaques)
                solara.SliderInt("vessels", value=vessels, min=0, max=500, on_value=set_vessels)
                solara.SliderFloat("eat prob", value=eat, min=0.0, max=1.0, on_value=set_eat)
                solara.SliderFloat("fortify prob", value=fortify, min=0.0, max=1.0, on_value=set_fortify)
                solara.SliderFloat("lactate perm prob", value=lactate, min=0.0, max=1.0, on_value=set_lactate)
                solara.SliderFloat("added glucose / tick", value=glucose, min=0.0, max=5000.0, on_value=set_glucose)
                solara.Select(label="exercise", value=exercise, values=["none","moderate","high"], on_value=set_exercise)
                solara.Select(label="booster", value=booster, values=["off","daily","every 2 days","twice per week","weekly"], on_value=set_booster)
                solara.Select(label="Stop when", value=stop_mode,
                              values=["Tick cap or domain stop","No plaques","pH >= 7.3"], on_value=set_stop_mode)

                with solara.Row():
                    def on_reset():
                        set_running(False); build_model()
                    solara.Button("Reset", on_click=on_reset, color="secondary")
                    solara.Button("Start", on_click=lambda: set_running(True), color="primary", disabled=running)
                    solara.Button("Stop", on_click=lambda: set_running(False), color="warning", disabled=not running)

                    def step_once():
                        if model_ref.value is None or should_stop(): return
                        model_ref.value.step()
                        set_step_count(model_ref.value.steps)
                        render_frame(); update_metrics()
                    solara.Button("Step once", on_click=step_once)

                solara.Text(f"Current step: {step_count}")

            if frame_png:
                solara.Image(frame_png, format="png", width="100%")

        with solara.Column():
            with solara.Card("Metrics (live)"):
                solara.FigureAltair(chart_neurons(metrics_df))
                solara.FigureAltair(chart_plaque_lactate(metrics_df))
                solara.FigureAltair(chart_env(metrics_df))


# IMPORTANT: prevent CLI main() from running when Solara imports this file
if __name__ == "__main__" and not os.environ.get("SOLARA_SERVER"):
    main()
