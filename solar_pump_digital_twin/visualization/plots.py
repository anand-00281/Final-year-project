"""
visualization/plots.py

All plotting for a single scenario's output DataFrame. Extends V1's
panel set with DC voltage/power, hydraulic power, and efficiency (V1
had no visibility into any of these -- Limitations B-F).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scenario(df, title: str = "", savepath: str = None):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)

    # (0,0) Irradiance + fault severity background
    ax = axes[0, 0]
    ax.plot(df.index, df["Irradiance_W_m2_ideal"], color="orange", label="Irradiance (ideal)")
    ax.set_ylabel("Irradiance (W/m^2)")
    ax_t = ax.twinx()
    ax_t.fill_between(df.index, 0, df["fault_severity"], color="red", alpha=0.25, label="fault severity")
    ax_t.set_ylabel("Fault severity (0-1)")
    ax_t.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_t.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.set_title("Irradiance & fault severity")

    # (0,1) weather_transient shading over irradiance
    ax = axes[0, 1]
    ax.plot(df.index, df["Irradiance_W_m2_ideal"], color="orange")
    ax.fill_between(df.index, 0, df["Irradiance_W_m2_ideal"].max(), where=df["weather_transient"], color="gray", alpha=0.2, label="weather_transient")
    ax.set_title("Weather-transient windows")
    ax.legend(fontsize=8)

    # (1,0) DC Voltage
    axes[1, 0].plot(df.index, df["DC_Voltage_V_ideal"], color="tab:blue", label="ideal")
    axes[1, 0].plot(df.index, df["DC_Voltage_V"], color="tab:blue", alpha=0.3, lw=0.5, label="measured")
    axes[1, 0].set_ylabel("DC Voltage (V)")
    axes[1, 0].set_title("DC bus voltage (was FIXED in V1 -- now self-consistent)")
    axes[1, 0].legend(fontsize=8)

    # (1,1) DC Current
    axes[1, 1].plot(df.index, df["DC_Current_A_ideal"], color="tab:green", label="ideal")
    axes[1, 1].plot(df.index, df["DC_Current_A"], color="tab:green", alpha=0.3, lw=0.5, label="measured")
    axes[1, 1].set_ylabel("DC Current (A)")
    axes[1, 1].set_title("Motor current")
    axes[1, 1].legend(fontsize=8)

    # (2,0) RPM
    axes[2, 0].plot(df.index, df["Motor_RPM_ideal"], color="tab:blue", label="ideal")
    axes[2, 0].plot(df.index, df["Motor_RPM"], color="tab:blue", alpha=0.3, lw=0.5, label="measured")
    axes[2, 0].set_ylabel("Speed (RPM)")
    axes[2, 0].set_title("Motor speed")
    axes[2, 0].legend(fontsize=8)

    # (2,1) Flow
    axes[2, 1].plot(df.index, df["Flow_Rate_LPM_ideal"], color="tab:cyan", label="ideal")
    axes[2, 1].plot(df.index, df["Flow_Rate_LPM"], color="tab:cyan", alpha=0.3, lw=0.5, label="measured")
    axes[2, 1].set_ylabel("Flow (L/min)")
    axes[2, 1].set_title("Flow rate")
    axes[2, 1].legend(fontsize=8)

    # (3,0) DC Power vs Hydraulic Power (new -- Limitations C/D)
    axes[3, 0].plot(df.index, df["DC_Power_W_ideal"], color="tab:red", label="DC power")
    axes[3, 0].plot(df.index, df["Hydraulic_Power_W_ideal"], color="tab:purple", label="Hydraulic power")
    axes[3, 0].set_ylabel("Power (W)")
    axes[3, 0].set_title("DC power vs hydraulic power (new in V2)")
    axes[3, 0].legend(fontsize=8)

    # (3,1) Efficiency proxy (new -- Limitation F)
    axes[3, 1].plot(df.index, df["Efficiency_Proxy_ideal"], color="black")
    axes[3, 1].set_ylabel("Efficiency proxy (P_hyd/P_dc)")
    axes[3, 1].set_title("System-level efficiency proxy (new in V2)")
    axes[3, 1].set_ylim(0, 1)

    fig.suptitle(title or "Digital Twin V2 scenario output", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved plot to {savepath}")
    plt.close(fig)
    return fig
