"""
This Leaky Integrate-and-Fire Neuron model uses Brian2 as an ODE solver.
This allows the user to create dynamic equations as you would see in published reports
"""

from brian2 import *
import matplotlib.pyplot as plt

# Start a new scope for Brian2 objects
start_scope()

# --- Model Parameters ---
tau = 10 * ms  # Membrane time constant
v_rest = -70 * mV  # Resting membrane potential
v_reset = -65 * mV  # Reset potential after a spike
v_threshold = -50 * mV  # Spike threshold

# Step current parameters
I_amplitude = 1.5  # Dimensionless current factor
t_start = 20 * ms  # Current turns on
t_stop = 80 * ms  # Current turns off

# --- Equations ---
# dv/dt = (v_rest - v + I_step) / tau
# We define I_step using a logical condition to create a step function
eqs = """
dv/dt = (v_rest - v + I_step) / ms : volt
I_step = I_amplitude * mV * (t >= t_start) * (t < t_stop) : volt
"""

# --- Create Neuron Group ---
neuron = NeuronGroup(
    1,
    eqs,
    threshold="v > v_threshold",
    reset="v = v_reset",
    method="exact",
)
neuron.v = v_rest  # Set initial potential

# --- Monitors ---
# Record membrane potential (v) and spike times
statemon = StateMonitor(neuron, "v", record=0)
spikemon = SpikeMonitor(neuron)

# --- Run Simulation ---
run(100 * ms)

# --- Plotting Results ---
plt.figure(figsize=(10, 5))

# Plot membrane potential
plt.plot(statemon.t / ms, statemon.v[0] / mV, label="Membrane Potential (v)", color="b")

# Draw threshold line
plt.axhline(
    v_threshold / mV,
    color="r",
    linestyle="--",
    label="Threshold (-50 mV)",
)

# Highlight current injection period
plt.axvspan(
    t_start / ms,
    t_stop / ms,
    color="gray",
    alpha=0.2,
    label="Step Current On",
)

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("LIF Neuron Response to a Step Current")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
