from pydantic import  BaseModel , Field
import numpy as np 
from model.RL_model import agent_
import random
from typing import  Literal
import matplotlib.pyplot as plt
import math

class Input(BaseModel):
    number_of_people: int = Field(ge=1, le=20)
    intensity: Literal["faible", "moyen", "forte"]
    has_kids_or_elderly: bool
    building_type: Literal["appartement", "maison", "villa"]
    
def state_encoder(number_of_people: int,
                  intensity: str,
                  has_kids_or_elderly: bool,
                  building_type: str) -> int:
    """Map (people, usage, kids, building) → unique integer ∈ [0, 359]."""
    usage_map = {"faible": 0, "moyen": 1, "forte": 2}
    building_map = {"appartement": 0, "maison": 1, "villa": 2}

    usage_idx = usage_map[intensity]
    kid_idx = 1 if has_kids_or_elderly else 0
    build_idx = building_map[building_type]

    # 3×2×3 = 18 combinations for the last three features
    return (number_of_people - 1) * 18 + usage_idx * 6 + kid_idx * 3 + build_idx

import math

def compute_heuristic_reward(input_: Input, action_idx: int) -> float:
    catalog = [
        {"capacity": 150, "availability": "in stock"},
        {"capacity": 200, "availability": "in stock"},
        {"capacity": 300, "availability": "in stock"},
        {"capacity": 400, "availability": "in stock"},
        {"capacity": 500, "availability": "in stock"},
        {"capacity": 700, "availability": "in stock"},
        {"capacity": 800, "availability": "in stock"},



    ]
    
    kit = catalog[action_idx]
    cap = kit["capacity"]

    # === 1. Besoin de base ===
    goal = input_.number_of_people * 40  # 40L / personne

    # === 2. Ajustement par intensité ===
    if input_.intensity == "faible":
        goal *= 0.9
    elif input_.intensity == "forte":
        goal *= 1.1

    # === 3. Ajustement par type de bâtiment ===
    if input_.building_type == "appartement":
        goal *= 0.9
    elif input_.building_type == "villa":
        goal *= 1.1

    # === 4. Ajustement par enfants/personnes âgées ===
    if input_.has_kids_or_elderly:
        goal *= 1.1

    # === 5. Reward via gaussienne ===
    sigma = 0.4 * goal  # 20% de la demande
    gaussian = math.exp(-((cap - goal) ** 2) / (2 * sigma ** 2))

    relative_error = abs(cap - goal) / goal



    return round(gaussian*10, 3)

agent=agent_(360,7,decay=0.9999,epsilon=0.8, gamma = 0,path="tables/q_table.npy")

#loop 

Episode = 20000
R=[]
Q=[]
for i in range(Episode):
    # 1. Generate random environment input
    input_ = Input(
        number_of_people=random.randint(1, 20),  # 1–20
        intensity=random.choice(["faible", "moyen", "forte"]),
        has_kids_or_elderly=random.choice([True, False]),
        building_type=random.choice(["appartement", "maison", "villa"])
    )


    # 2. Encode state
    state_id = state_encoder(
        input_.number_of_people,
        input_.intensity,
        input_.has_kids_or_elderly,
        input_.building_type
    )

    # 3. Choose action (kit index)
    action = agent.choose_action(state_id)
    current_q = agent.q_table[state_id][action]

    # 4. Compute reward for that action
    reward = compute_heuristic_reward(input_=input_, action_idx=action)
    print(reward)

    # 5. Update Q-table
    agent.update_q_table(state=state_id, action=action, reward=reward)
    Q.append(abs(agent.q_table[state_id][action]-current_q))
    R.append(reward)
    agent.epsilon = max(0.05, agent.epsilon * agent.decay)
    agent.alpha = 1 / (1 + 0.0001 * i)


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
agent.save_q_table(path="tables/q_table.npy")





