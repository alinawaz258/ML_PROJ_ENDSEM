"""
Lab 4: Hidden Markov Model & Viterbi Algorithm
Unit 4 Lab Exercise

Goal:
Implement the Viterbi Algorithm to decode the most probable sequence of hidden states
given a sequence of observed disease symptoms/features.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def viterbi(obs_seq, states, start_prob, trans_prob, emit_prob):
    """
    Standard Viterbi Algorithm Implementation.
    
    Parameters:
        obs_seq: list of observation indices/names
        states: list of hidden state names
        start_prob: initial state probabilities (dict)
        trans_prob: state transition probabilities (dict of dicts)
        emit_prob: emission probabilities (dict of dicts)
    
    Returns:
        best_path: most likely hidden state sequence
        V: Viterbi probability matrix
    """
    n_states = len(states)
    n_obs = len(obs_seq)
    
    # V[t][s] = max probability of state s at time step t
    V = [{}]
    path = {}
    
    # 1. Initialization (t = 0)
    for s in states:
        V[0][s] = start_prob[s] * emit_prob[s][obs_seq[0]]
        path[s] = [s]
        
    # 2. Recursion (t = 1 to n_obs - 1)
    for t in range(1, n_obs):
        V.append({})
        new_path = {}
        
        for s in states:
            # Calculate prob from each previous state p to current state s
            prob, state_prev = max(
                (V[t - 1][p] * trans_prob[p][s] * emit_prob[s][obs_seq[t]], p)
                for p in states
            )
            V[t][s] = prob
            new_path[s] = path[state_prev] + [s]
            
        path = new_path

    # 3. Termination
    max_prob, best_last_state = max((V[n_obs - 1][s], s) for s in states)
    best_path = path[best_last_state]
    
    return best_path, max_prob, V


# ============================================================
# Problem Setup: Plant Health Progression Monitoring
# ============================================================
# Hidden States: The actual condition of the plant across inspection stages
states = ['Healthy', 'Early_Infection', 'Severe_Disease']

# Observations: Visible symptoms extracted from leaf inspection
observations = ['Green_Clean', 'Mild_Yellowing', 'Dark_Spots', 'Necrosis_Lesions']

# Initial State Probabilities
start_prob = {
    'Healthy': 0.60,
    'Early_Infection': 0.30,
    'Severe_Disease': 0.10
}

# Transition Matrix A (P(State_t | State_{t-1}))
trans_prob = {
    'Healthy':         {'Healthy': 0.70, 'Early_Infection': 0.25, 'Severe_Disease': 0.05},
    'Early_Infection': {'Healthy': 0.10, 'Early_Infection': 0.60, 'Severe_Disease': 0.30},
    'Severe_Disease':  {'Healthy': 0.05, 'Early_Infection': 0.10, 'Severe_Disease': 0.85}
}

# Emission Matrix B (P(Obs_t | State_t))
emit_prob = {
    'Healthy': {
        'Green_Clean': 0.80, 'Mild_Yellowing': 0.15, 'Dark_Spots': 0.04, 'Necrosis_Lesions': 0.01
    },
    'Early_Infection': {
        'Green_Clean': 0.10, 'Mild_Yellowing': 0.50, 'Dark_Spots': 0.35, 'Necrosis_Lesions': 0.05
    },
    'Severe_Disease': {
        'Green_Clean': 0.01, 'Mild_Yellowing': 0.09, 'Dark_Spots': 0.40, 'Necrosis_Lesions': 0.50
    }
}

# Sample observed inspection timeline for a monitored crop batch
sample_observations = ['Green_Clean', 'Mild_Yellowing', 'Dark_Spots', 'Necrosis_Lesions', 'Necrosis_Lesions']

print("--- Hidden Markov Model: Viterbi Algorithm ---")
print(f"Inspection Observations Sequence:")
for idx, obs in enumerate(sample_observations, 1):
    print(f"  Day {idx}: {obs}")

# Execute Viterbi decoding
best_sequence, path_prob, V_matrix = viterbi(
    sample_observations, states, start_prob, trans_prob, emit_prob
)

print("\n--- Decoded Most Likely Hidden States (Viterbi Path) ---")
for idx, (obs, st) in enumerate(zip(sample_observations, best_sequence), 1):
    print(f"  Day {idx} [{obs}] -> Hidden State: {st}")
print(f"\nSequence Path Probability: {path_prob:.6e}")

# ============================================================
# Visualization: Viterbi Trellis Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

time_steps = len(sample_observations)
state_y = {state: i for i, state in enumerate(states)}

# Plot trellis nodes and values
for t in range(time_steps):
    for s in states:
        y_val = state_y[s]
        p_val = V_matrix[t][s]
        ax.scatter(t, y_val, s=250, color='lightblue', edgecolors='navy', zorder=3)
        ax.text(t, y_val + 0.13, f"{p_val:.2e}", ha='center', fontsize=8, color='black')

# Draw optimal Viterbi decoded path
best_y = [state_y[s] for s in best_sequence]
ax.plot(range(time_steps), best_y, color='crimson', linewidth=2.5, linestyle='-', marker='o',
        label='Decoded Viterbi Path', zorder=4)

ax.set_xticks(range(time_steps))
ax.set_xticklabels([f"Day {i+1}\n({sample_observations[i]})" for i in range(time_steps)], fontsize=9)
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states, fontsize=11, fontweight='bold')
ax.set_title("Viterbi Algorithm: Optimal Hidden State Trajectory", fontsize=13, fontweight='bold')
ax.set_xlabel("Time Step (Observation)", fontsize=11)
ax.set_ylabel("Hidden Health State", fontsize=11)
ax.set_ylim(-0.5, len(states) - 0.2)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("lab4_output.png", dpi=150)
print("\nPlot saved: lab4_output.png")
