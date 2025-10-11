from model.RL_model import agent_
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import json, math, random

# ---- Load OFF-GRID catalog ----
with open("catalog/catalog4.json", "r") as f:
    catalog = json.load(f)

# ---- Build action space from catalog ----
# Each action is (category, idx, item)
action_space = []
for category, items in catalog.items():
    for idx, item in enumerate(items):
        action_space.append((category, idx, item))

NB_ACTIONS = len(action_space)

# ---- Target energy (same idea, off-grid context) ----
def target_monthly_energy_kwh(building_type: str,
                              surface_m2: float,
                              monthly_consumption_kwh: float,
                              irradiance_kwh_m2_day: float = 4.0) -> float:
    PANEL_WATT = 610
    PANEL_AREA = 2.0
    PR = 0.80
    bldg_factor = {"appartement": 0.7, "maison": 0.85, "villa": 1.0}[building_type]
    usable_surface = surface_m2 * bldg_factor
    max_panels = int(usable_surface // PANEL_AREA)
    kw_dc = max_panels * PANEL_WATT / 1000.0
    monthly_prod = kw_dc * PR * irradiance_kwh_m2_day * 30
    # Off-grid: you generally size to cover consumption
    return min(monthly_prod, monthly_consumption_kwh)

# ---- Helper: panel monthly production (kWh) given current panel count ----
def monthly_prod_from_panels(panels: int, irradiance_kwh_m2_day: float = 4.0) -> float:
    # 0.610 kW per panel * PR(0.8) * irradiance * ~30 days
    return (panels * 0.610) * 0.80 * irradiance_kwh_m2_day * 30

# ---- Pydantic State ----
class State(BaseModel):
    nb_personnes: int = Field(ge=1, le=10)
    surface_m2: float = Field(ge=1, le=200)
    conso_totale_kwh_mois: float = Field(ge=1, le=3000)
    # appliance intensity flags (1..3 each)
    high: int = Field(ge=1, le=3)
    medium: int = Field(ge=1, le=3)
    low: int = Field(ge=1, le=3)

    # dynamic build-up
    panels: int = Field(ge=1, le=4)
    kits: int = Field(ge=0, le=1)        # 0/1 (off-grid kit)
    batteries: int = Field(ge=0, le=2)   # 0/1/2


# ---- Random initial state ----
def random_state() -> State:
    return State(
        nb_personnes=np.random.randint(1, 11),                       # 1..10
        surface_m2=float(np.random.uniform(1, 200)),                 # 1..200
        conso_totale_kwh_mois=float(np.random.uniform(1, 3000)),     # 1..3000
        high=int(np.random.randint(1, 4)),                           # 1..3
        medium=int(np.random.randint(1, 4)),                         # 1..3
        low=int(np.random.randint(1, 4)),                            # 1..3
        panels=1,
        kits=0,
        batteries=0
    )


# ---- Discretization bases (reduced) ----
SURF_BASE   = 3     # surface → 3 bins
CONS_BASE   = 4     # consumption → 4 bins
PPL_BASE    = 4     # nb_personnes grouped into 4
APPLI_BASE  = 5     # appliance profile → 5 bins
PAN_BASE    = 4     # panels 1..4
KITS_BASE   = 2     # kits 0..1
BATT_BASE   = 3     # batteries 0..2

NUM_STATES = (
    SURF_BASE * CONS_BASE * PPL_BASE *
    APPLI_BASE * PAN_BASE * KITS_BASE * BATT_BASE
)


# ---- Indexing function ----
def to_indx(s: State) -> int:
    # clamp continuous
    sa = max(1.0, min(200.0, s.surface_m2))
    mc = max(1.0, min(3000.0, s.conso_totale_kwh_mois))

    # binning surface
    surf_i = 0 if sa < 70 else 1 if sa < 140 else 2

    # binning consumption (0–750, 750–1500, 1500–2250, 2250–3000)
    if mc < 750:
        cons_i = 0
    elif mc < 1500:
        cons_i = 1
    elif mc < 2250:
        cons_i = 2
    else:
        cons_i = 3

    # binning nb_personnes (1–2, 3–5, 6–8, 9–10)
    if s.nb_personnes <= 2:
        ppl_i = 0
    elif s.nb_personnes <= 5:
        ppl_i = 1
    elif s.nb_personnes <= 8:
        ppl_i = 2
    else:
        ppl_i = 3

    # merge appliances → appliance profile
    appliance_score = s.low * 1 + s.medium * 2 + s.high * 3   # range 6..18
    if appliance_score <= 7:
        appli_i = 0
    elif appliance_score <= 9:
        appli_i = 1
    elif appliance_score <= 12:
        appli_i = 2
    elif appliance_score <= 15:
        appli_i = 3
    else:
        appli_i = 4

    # panels, kits, batteries
    pan_i  = min(max(s.panels, 1), 4) - 1
    kits_i = min(max(s.kits, 0), 1)
    batt_i = min(max(s.batteries, 0), 2)

    idx = ((((((surf_i * CONS_BASE + cons_i) * PPL_BASE + ppl_i)
               * APPLI_BASE + appli_i)
               * PAN_BASE + pan_i)
               * KITS_BASE + kits_i)
               * BATT_BASE + batt_i)

    # assert 0 <= idx < NUM_STATES, f"bad idx {idx}"
    return idx


# ---- Agent ----
agent = agent_(nb_state=NUM_STATES, nb_action=7, decay=0.96, epsilon=0.5, alpha=0.1, gamma=0.7)

# ---- Reward (step & terminal) ----
def reward(state: State, action: int, terminal: bool = False):
    base_target = target_monthly_energy_kwh(
        surface_m2=state.surface_m2,
        monthly_consumption_kwh=state.conso_totale_kwh_mois,
        building_type="maison"  # heuristic fallback
    )
    if base_target <= 0:
        return -1.0, state

    appliance_score = state.low * 1 + state.medium * 2 + state.high * 3
    demand_mult = 1.0 + 0.02 * (appliance_score - 10)  # ~±20% scaling
    Target = base_target * demand_mult

    if not terminal:
        category, idx, item = action_space[action]

        if category == "kits_off_grid":
            if state.kits == 0:
                state.kits = 1
                return +1.0, state
            else:
                return -0.5, state

        elif category == "panneaux":
            if state.panels < 4:
                prev_prod = monthly_prod_from_panels(state.panels)
                state.panels += 1
                new_prod = monthly_prod_from_panels(state.panels)
                improvement = (abs(Target - prev_prod) - abs(Target - new_prod)) / max(Target, 1.0)
                return max(-0.05, improvement), state
            else:
                return -0.3, state

        elif category == "batteries":
            if state.batteries < 2:
                state.batteries += 1
                return +0.2, state
            else:
                return -0.3, state

        return -0.5, state

    # terminal evaluation
    prod = monthly_prod_from_panels(state.panels)
    sigma = 0.25 * Target
    gauss = math.exp(-((prod - Target) ** 2) / (2 * sigma**2))

    completeness = 0.0
    if state.kits == 0:
        if state.batteries == 0:
            completeness -= 0.3

    if Target > 1500 and state.batteries == 0 and state.kits == 0:
        completeness -= 0.3

    final_reward = gauss + completeness
    return final_reward, state


# ---- Terminal condition for OFF-GRID ----
def is_terminal(state: State, max_steps_reached: bool = False) -> bool:
    if state.kits > 0:
        return True

    # custom system: must have panels + ≥1 battery
    has_core_system = (state.panels > 0 and state.batteries > 0)

    if has_core_system:
        base_target = target_monthly_energy_kwh(
            surface_m2=state.surface_m2,
            monthly_consumption_kwh=state.conso_totale_kwh_mois,
            building_type="maison"
        )
        appliance_score = state.low * 1 + state.medium * 2 + state.high * 3
        demand_mult = 1.0 + 0.02 * (appliance_score - 10)
        Target = base_target * demand_mult

        current_energy = monthly_prod_from_panels(state.panels, irradiance_kwh_m2_day=4.0)

        if Target > 0 and current_energy >= 0.85 * Target:
            return True

    if state.panels >= 4:
        return True
    if max_steps_reached:
        return True

    return False
# ---- Training loop (episodic, with TD(λ) updates) ----
episodes = 50000
max_steps = 6
Rew, delta_Q = [], []
theta = 10.0

for i in range(episodes):
    s = random_state()
    step = 0
    Terminal = False

    while not Terminal and step < max_steps:
        s_id = to_indx(s)
        a = agent.choose_action(s_id)

        # take action, get step reward
        r, s = reward(s, a, terminal=False)
        s_next_id = to_indx(s)

        # TD update
        old_q = agent.q_table[s_id][a]
        agent.update_q_table(s_id, a, r, next_state=s_next_id)
        delta_Q.append(abs(agent.q_table[s_id][a] - old_q))
        Rew.append(r)

        # check terminal condition
        Terminal = is_terminal(s, max_steps_reached=(step + 1 >= max_steps))

        if Terminal:
            # add terminal reward (Gaussian shaping)
            rT, s = reward(s, a, terminal=True)
            agent.update_q_table(s_next_id, a, rT, next_state=None)
            Rew.append(rT)

        step += 1

    # learning rate schedule
    agent.alpha = theta / math.sqrt(i + 1.0)

    # exploration decay
    if i % 1000 == 0:
        agent.epsilon = max(0.05, agent.epsilon * agent.decay)
        print(f"Episode {i}: epsilon={agent.epsilon:.3f}, alpha={agent.alpha:.3f}")

# save trained table
agent.save_q_table("tables/tables_offgrid.npy")
