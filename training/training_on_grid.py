from model.RL_model import agent_
from pydantic import BaseModel , Field
import numpy as np  
import json
import matplotlib.pyplot as plt
from typing import Literal
import math

with open("catalog/catalog3.json", "r") as f:
    catalog = json.load(f)


# Flatten all categories into a single action list
action_space = []


for category, items in catalog.items():
    for idx, item in enumerate(items):
        action_space.append((category, idx, item))






def target_monthly_energy_kwh(building_type: str,
                               surface_m2: float,
                               monthly_consumption_kwh: float,
                               irradiance_kwh_m2_day = 4.0) -> float:
    # Panel characteristics
    PANEL_WATT = 610       # W per panel
    PANEL_AREA = 2.0       # m² per panel (approx)
    PR = 0.80               # performance ratio

    # Building type factor (adjust surface usability or PR)
    building_factor = {
        "appartement": 0.7,
        "maison": 0.85,
        "villa": 1.0
    }[building_type]

    # Max number of panels that fit on usable surface
    usable_surface = surface_m2 * building_factor
    max_panels = int(usable_surface // PANEL_AREA)

    # Energy production if we fill available surface
    kw_dc = max_panels * PANEL_WATT / 1000.0
    monthly_prod = kw_dc * PR * irradiance_kwh_m2_day * 30  # ~30 days/month

    # Target = min(what we can produce, what is consumed)
    return min(monthly_prod, monthly_consumption_kwh)


#class off grid : [appareil : nom , nbre ] , consomation KW ,     nbre de personne , surface disponible m2

  












class State(BaseModel):
    # Static inputs
   # type de batiment [villa , maison , appartement]
    surface_area: float
    monthly_consumption: float 
    building_type: Literal["appartement", "maison", "villa"]
                                         #kwat !!!
    # Dynamic build-up
    panels: int = Field(ge=1,le=10)
    kits: int = Field(ge=0,le=1)
    compteur: int = Field(ge=0,le=1)




def random_state() -> State:
    return State(
        surface_area=np.random.uniform(10, 200),

        monthly_consumption=np.random.uniform(100, 1000),
        building_type = np.random.choice(["appartement", "maison", "villa"]),

        panels=  1,
        kits  = 0,
        compteur = 0

    )

# Fixed: Use 12 elements to include raw budget

SURF_BASE   = 4
CONS_BASE   = 3
BUILD_BASE  = 3
PAN_BASE    = 10
KITS_BASE   = 2
COMP_BASE   = 2

NUM_STATES = SURF_BASE * CONS_BASE * BUILD_BASE * PAN_BASE * KITS_BASE * COMP_BASE  # 1440

BUILDING_MAP = {"appartement": 0, "maison": 1, "villa": 2}

def to_indx(s: State) -> int:
    # Clamp continuous
    sa = max(0.0, min(200.0, s.surface_area))
    mc = max(0.0, min(1000.0, s.monthly_consumption))

    # Binning
    surf_i = 0 if sa < 50 else 1 if sa < 100 else 2 if sa < 150 else 3
    cons_i = 0 if mc < 300 else 1 if mc < 600 else 2
    bldg_i = BUILDING_MAP[s.building_type]

    # Clamp discrete BEFORE encoding
    pan   = min(max(s.panels,   1), 10)
    kits  = min(max(s.kits,     0), 1)
    comp  = min(max(s.compteur, 0), 1)

    pan_i  = pan - 1          # 1..4 -> 0..3
    kits_i = kits             # 0..1
    comp_i = comp             # 0..1

    idx = (((((surf_i) * CONS_BASE + cons_i) * BUILD_BASE + bldg_i)
            * PAN_BASE + pan_i) * KITS_BASE + kits_i) * COMP_BASE + comp_i

    # Optional sanity check in dev:
    # assert 0 <= idx < NUM_STATES, f"bad idx {idx} from s={s}"
    return idx



agent=agent_(nb_state=1440,nb_action=6,decay=0.96,epsilon=0.5,alpha=0.01,gamma=0.75,path="tables/tables3.npy") #alpha being 0 won t affect convergence because we perform only one step searsh in the future 



# --- very small helpers ----------------------------------------------------

# ==== Simple helpers for reward ====
def _kw_from_text(s: str) -> float:
    return float(str(s).lower().replace("kw", "").strip())

def _monthly_energy_panels(panels: int) -> float:
    # 610 W/panel, PR=0.80, 4 kWh/m²/day, ~30 days
    return (panels * 0.610) * 0.80 * 4.0 * 30

def _triangular_match(prod: float, target: float) -> float:
    """1 at exact target, linear drop with absolute error, in [0..1]."""
    if target <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(prod - target) / target)

def _compteur_price_choice(target_kwh: float) -> float:
    # industrial if target high
    return (catalog["compteurs_bidirectionnels"][1]["price_solution"]
            if target_kwh > 1200
            else catalog["compteurs_bidirectionnels"][0]["price_solution"])

# --- simple reward ---------------------------------------------------------
def reward(state: State, action: int, terminal: bool = False):
    """
    Simple rules:
      - Kits (0..2) only at initial step (panels==1, compteur==0, kits==0). If chosen, episode ends.
      - Panels (3): reward improvement toward target minus panel cost.
      - Compteur (4/5): one-time small bonus minus compteur cost. Re-adding penalized.
    """
    Target = target_monthly_energy_kwh(
        surface_m2=state.surface_area,
        monthly_consumption_kwh=state.monthly_consumption,
        building_type=state.building_type
    )

    COST_SCALE = 50000.0
    MIX_PENALTY = -1.5
    REPEAT_COMP_PEN = -0.5
    MAX_PANELS_PEN = -0.3
    COMPTEUR_BONUS = 0.2
    MIN_IMPROVEMENT = -0.05

    # terminal branch kept neutral (we shape along the way)
    if terminal:
        x = _monthly_energy_panels(state.panels)
        sigma = 0.5 * Target
        reward_value = math.exp(-((x - Target) ** 2) / (2 * sigma ** 2)) if Target > 0 else 0
        
        return reward_value - cost_pen, state

    # 0..2 => choose a kit (only at start)
    if 0 <= action <= 2:
        at_start = (state.panels == 1 and state.compteur == 0 and state.kits == 0)
        if not at_start:
            return MIX_PENALTY, state

        state.kits = 1
        kit_kw = _kw_from_text(catalog["kit_on_grid"][action]["capacity_solution"])
        kit_price = catalog["kit_on_grid"][action]["price_solution"]
        match = _triangular_match(kit_kw, Target)
        cost_pen = kit_price / COST_SCALE
        return (match - cost_pen), state

    # 3 => add a panel
    if action == 3:
        if state.kits > 0:
            return MIX_PENALTY, state
        if state.panels >= 4:
            return MAX_PANELS_PEN, state

        prev_prod = _monthly_energy_panels(state.panels)
        prev_err  = abs(Target - prev_prod)

        state.panels += 1

        new_prod = _monthly_energy_panels(state.panels)
        new_err  = abs(Target - new_prod)

        denom = Target if Target > 0 else 1.0
        improvement = (prev_err - new_err) / denom

        panel_price = catalog["panneaux"][0]["price_solution"]
        cost_pen = panel_price / COST_SCALE

        step_r = max(improvement, MIN_IMPROVEMENT) - cost_pen
        return step_r, state

    # 4/5 => add compteur
    if action in (4, 5):
        if state.kits > 0:
            return MIX_PENALTY, state
        if state.compteur >= 1:
            return REPEAT_COMP_PEN, state

        state.compteur = 1
        comp_price = _compteur_price_choice(Target)
        cost_pen = comp_price / COST_SCALE
        prod = _monthly_energy_panels(state.panels)
        near = _triangular_match(prod, Target) * 0.2
        step_r = COMPTEUR_BONUS + near - cost_pen
        return step_r, state

    # unknown action
    return -0.1, state


def is_terminal(state: State) -> bool:
    target = target_monthly_energy_kwh(
        building_type=state.building_type,
        surface_m2=state.surface_area,
        monthly_consumption_kwh=state.monthly_consumption
    )
    # If a kit was chosen at the beginning, stop immediately
    if state.kits == 1:
        return True
    # Panels path: require compteur and “close enough” to target or hard cap
    prod = _monthly_energy_panels(state.panels)
    return (
        (state.compteur == 1 and prod >= 0.90 * target)  # ~90% coverage with compteur
        or state.panels >= 4                             # safety cap
    )



episodes = 50000
Rew = []
theta = 10
delta_Q = []

for i in range(episodes):
    Terminal = False
    agent.elegibility.fill(0.0)
    agent.map.clear()
    input_state = random_state()
    while Terminal==False:
        input_id = to_indx(input_state)

        action = agent.choose_action(input_id)
        r , input_state = reward(input_state, action,Terminal)  # numeric reward
        s1_id = to_indx(input_state)
        if input_state.kits == 1:
            Terminal = True

        # (optional) tiny finish bonus for valid panels+compteur completion
        if not Terminal and is_terminal(input_state) and input_state.kits == 0 and input_state.compteur == 1:
            r += 0.1

        Rew.append(r)
        #here see if input state should be terminal 
        current_q = agent.q_table[input_id][action]
        agent.update_q_table(input_id, action, r, next_state=s1_id)
        
        delta_Q.append(abs(agent.q_table[input_id][action] - current_q))
        Rew.append(r)
        if is_terminal(input_state):
            Terminal = True

        # Learning rate decay

        if i % 10000 == 0:
            print(f"Episode {i}: reward = {round(r, 3)}")
            print(agent.q_table[input_id][action], "q_value") 
            print(abs(agent.q_table[input_id][action] - current_q), "delta_Q")
    agent.alpha = theta / math.sqrt(i + 1)
    agent.epsilon *= agent.decay  # exploration decay


agent.save_q_table("tables/tables3.npy")
print(agent.q_table)
