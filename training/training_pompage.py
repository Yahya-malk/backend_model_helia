from pydantic import  BaseModel , Field
import numpy as np 
from model.RL_model import agent_
import random
from typing import  Literal
import matplotlib.pyplot as plt
import math
catalog = [
  {"name": "Kit de pompage solaire 220 V 1.5–3 CV", "voltage": "220 V", "power": 3, "price": "33 700 MAD"},
  {"name": "Kit de pompage solaire 380 V 3 CV", "voltage": "380 V", "power": 3, "price": "40 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 4–5.5 CV", "voltage": "380 V", "power": 5.5, "price": "49 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 7.5 CV", "voltage": "380 V", "power": 7.5, "price": "41 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 10 CV", "voltage": "380 V", "power": 10, "price": "50 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 15 CV", "voltage": "380 V", "power": 15, "price": "66 900 MAD"}
]

class Input(BaseModel):
    puissance: float = Field(ge=1, le=200)
    hmt: float = Field(ge=1, le=100)
    debit: float = Field(ge=1, le=200)

def state_encoder(input_ : Input) -> int:
    """
    Encode (puissance, hmt, debit) en un entier unique ∈ [0, 124],
    en divisant chaque variable en 5 bins.
    """
    puissance , hmt , debit = input_.puissance , input_.hmt , input_.debit
    # --- 1. Définir les bornes ---
    puissance_bins = np.linspace(1, 200, 6)  # 5 intervalles
    hmt_bins = np.linspace(1, 100, 6)        # 5 intervalles
    debit_bins = np.linspace(1, 200, 6)      # 5 intervalles

    # --- 2. Trouver les indices de bin ---
    p_idx = np.digitize([puissance], puissance_bins, right=True)[0] - 1
    h_idx = np.digitize([hmt], hmt_bins, right=True)[0] - 1
    d_idx = np.digitize([debit], debit_bins, right=True)[0] - 1

    # Clamp (par sécurité : éviter qu’un max tombe dans l’index 5)
    p_idx = min(max(p_idx, 0), 4)
    h_idx = min(max(h_idx, 0), 4)
    d_idx = min(max(d_idx, 0), 4)

    # --- 3. Encoder en un seul entier ---
    return p_idx * 25 + h_idx * 5 + d_idx

import math

def compute_heuristic_reward(input_: Input, action_idx: int) -> float:
    catalog = [
        {"name": "Kit 220V 1.5–3 CV", "voltage": "220 V", "power": 3},
        {"name": "Kit 380V 3 CV", "voltage": "380 V", "power": 3},
        {"name": "Kit 380V 4–5.5 CV", "voltage": "380 V", "power": 5.5},
        {"name": "Kit 380V 7.5 CV", "voltage": "380 V", "power": 7.5},
        {"name": "Kit 380V 10 CV", "voltage": "380 V", "power": 10},
        {"name": "Kit 380V 15 CV", "voltage": "380 V", "power": 15}
    ]
    
    kit = catalog[action_idx]
    kit_power = kit["power"]

    rho_g = 10000  # N/m³ ~ eau (ρg)
    Q = input_.debit / 3600.0  # m³/h -> m³/s
    H = input_.hmt             # m
    P_hyd = rho_g * Q * H / 735.5  # en CV (1 CV = 735.5 W)

    # === 2. Reward basé sur l’écart puissance ===
    sigma = 0.4 * P_hyd if P_hyd > 0 else 1  # tolérance ~20%
    reward_power = math.exp(-((kit_power - P_hyd) ** 2) / (2 * sigma ** 2))

    # === 3. Bonus si l’écart n’est pas trop grand (sous/sur dimensionnement) ===
    penalty = 0.0
    if kit_power < P_hyd * 0.8:   # sous-dimensionné
        penalty -= 0.3
    elif kit_power > P_hyd * 1.5: # sur-dimensionné
        penalty -= 0.2

    return (reward_power + penalty)*3

agent=agent_(125,6,decay=0.999,epsilon=0.5, gamma = 0,path ="tables/q_table2.npy")

#loop 

Episode = 10000
R=[]
Q=[]
for i in range(Episode):
    # 1. Generate random environment input
    input_ = Input(
        puissance=random.randint(1, 200),  # 1–20
     hmt=random.randint(1, 100),
     debit=random.randint(1, 200),   
    )


    # 2. Encode state
    state_id = state_encoder(
        input_
    )

    # 3. Choose action (kit index)
    action = agent.choose_action(state_id)
    print(action)
    current_q = agent.q_table[state_id][action]

    # 4. Compute reward for that action
    reward = compute_heuristic_reward(input_=input_, action_idx=action)

    # 5. Update Q-table
    delta=agent.update_q_table(state=state_id, action=action, reward=reward)
    Q.append(abs(agent.q_table[state_id][action]-current_q))
    R.append(reward)
    agent.epsilon*= agent.decay
    agent.alpha = 1 / ((1 +  i))


import matplotlib.pyplot as plt
import numpy as np

def smooth(data, window=200):
    """Retourne une version lissée de data avec une moyenne glissante."""
    return np.convolve(data, np.ones(window)/window, mode="valid")

# === Courbe des récompenses (lissée) ===
plt.figure(figsize=(10,5))
plt.plot(smooth(R), label="Reward (moyenne glissante)", linewidth=1.2)
plt.xlabel("Épisodes")
plt.ylabel("Récompense")
plt.title("Évolution des récompenses (lissée)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("rewards_plot_smooth.png", dpi=300)
plt.close()

# === Courbe des ΔQ (lissée) ===
plt.figure(figsize=(10,5))
plt.plot(smooth(Q), label="ΔQ (moyenne glissante)", linewidth=1.2, color="orange")
plt.xlabel("Épisodes")
plt.ylabel("ΔQ (update Q-table)")
plt.title("Évolution des mises à jour Q-table (lissée)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("deltaQ_plot_smooth.png", dpi=300)
plt.close()

# === Les deux ensemble (lissés) ===
plt.figure(figsize=(10,5))
plt.plot(smooth(R), label="Reward (lissé)", linewidth=1.2)
plt.plot(smooth(Q), label="ΔQ (lissé)", linewidth=1.2, alpha=0.8)
plt.xlabel("Épisodes")
plt.ylabel("Valeur")
plt.title("Récompense et ΔQ (lissés)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("reward_vs_deltaQ_smooth.png", dpi=300)
plt.close()

# Save learned Q-table
agent.save_q_table(path="tables/q_table2.npy")
print(agent.q_table)
print(R)




