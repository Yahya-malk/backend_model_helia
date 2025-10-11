from fastapi import FastAPI
from pydantic import BaseModel , Field
from typing import Literal
import numpy as np
import json
import math
from fastapi.middleware.cors import CORSMiddleware
from model.RL_model import agent_
from model.RL_model2 import agent_2
from openai import OpenAI

catalog_pompage = [
  {"name": "Kit de pompage solaire 220 V 1.5–3 CV", "voltage": "220 V", "power": 3, "price": "33 700 MAD"},
  {"name": "Kit de pompage solaire 380 V 3 CV", "voltage": "380 V", "power": 3, "price": "40 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 4–5.5 CV", "voltage": "380 V", "power": 5.5, "price": "49 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 7.5 CV", "voltage": "380 V", "power": 7.5, "price": "41 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 10 CV", "voltage": "380 V", "power": 10, "price": "50 000 MAD"},
  {"name": "Kit de pompage solaire 380 V 15 CV", "voltage": "380 V", "power": 15, "price": "66 900 MAD"}
]
GITHUB_TOKEN = "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"
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"
with open("catalog/catalog3.json", "r") as f:
    catalog_On = json.load(f)
with open("catalog/catalog4.json", "r") as f:
    catalog_off = json.load(f)
action_space_off= []
for category, items in catalog_off.items():
    for idx, item in enumerate(items):
        action_space_off.append((category, idx, item))

NB_ACTIONS_off = len(action_space_off)
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

chauffage_catalog = [
    {
        "id": 1,
        "name": "Chauffe-eau solaire ECONOMY 150L",
        "category": "Chauffage",
        "capacity": 150,
        "brand": "ECONOMY",
        "price_mad": 5000
    },
    {
        "id": 2,
        "name": "Chauffe-eau solaire ECONOMY 200L",
        "category": "Chauffage",
        "capacity": 200,
        "brand": "ECONOMY",
        "price_mad": 6500
    },
    {
        "id": 3,
        "name": "Chauffe-eau solaire ECONOMY 300L",
        "category": "Chauffage",
        "capacity": 300,
        "brand": "ECONOMY",
        "price_mad": 8000
    },
    {
        "id": 4,
        "name": "Chauffe-eau solaire PRESSURISE 400L",
        "category": "Chauffage",
        "capacity": 400,
        "brand": "PRESSURISE",
        "price_mad": 9500
    },
    {
        "id": 5,
        "name": "Chauffe-eau solaire PRESSURISE 500L",
        "category": "Chauffage",
        "capacity": 500,
        "brand": "PRESSURISE",
        "price_mad": 11000
    },
    {
        "id": 5,
        "name": "Chauffe-eau solaire PRESSURISE 500L",
        "category": "Chauffage",
        "capacity": 500,
        "brand": "PRESSURISE",
        "price_mad": 11000
    },
    {
        "id": 5,
        "name": "Chauffe-eau solaire PRESSURISE 500L",
        "category": "Chauffage",
        "capacity": 500,
        "brand": "PRESSURISE",
        "price_mad": 11000
    },
    {
        "id": 5,
        "name": "Chauffe-eau solaire PRESSURISE 500L",
        "category": "Chauffage",
        "capacity": 500,
        "brand": "PRESSURISE",
        "price_mad": 11000
    }
]




onduleur_on =   {"onduleurs": [
    {
      "name": "SolaX X1-MINI-2.0K-G4",
      "capacity_solution": "2 kW",
      "price_solution": 5500
    },
    {
      "name": "SolaX X1-MINI-3.0K-G4",
      "capacity_solution": "3 kW",
      "price_solution": 6000
    },
    {
      "name": "SolaX X1-BOOST-4.2KW-G4",
      "capacity_solution": "4.2 kW",
      "price_solution": 8100
    },
    {
      "name": "SolaX X1-BOOST-5.0KW-G4",
      "capacity_solution": "5 kW",
      "price_solution": 9200
    },
    {
      "name": "Sungrow SG2.0RS (On-Grid)",
      "capacity_solution": "2 kW",
      "price_solution": 9800
    },
    {
      "name": "Sungrow SG3.0RS (On-Grid)",
      "capacity_solution": "3 kW",
      "price_solution": 8200
    },
    {
      "name": "Sungrow SG5.0RS",
      "capacity_solution": "5 kW",
      "price_solution": 10400
    },
    {
      "name": "SolaX X3-MIC-8KW-G2",
      "capacity_solution": "8 kW",
      "price_solution": 13800
    },
    {
      "name": "SolaX X3-MIC-10KW-G2",
      "capacity_solution": "10 kW",
      "price_solution": 15100
    }
  ]}



def choose_onduleur_on(n_panels: int, onduleurs_dict: dict) -> dict:
    onduleurs = onduleurs_dict["onduleurs"]  # extraire la liste
    
    total_kw = n_panels * 0.610
    recommended_min = total_kw / 1.3
    recommended_max = total_kw / 1.1

    candidates = []
    for ond in onduleurs:
        cap_kw = float(ond["capacity_solution"].replace("kW", "").replace(" ", ""))
        if recommended_min <= cap_kw <= recommended_max:
            candidates.append((abs(cap_kw - total_kw), ond))

    if candidates:
        return min(candidates, key=lambda x: x[0])[1]

    return min(onduleurs, key=lambda ond: abs(
        float(ond["capacity_solution"].replace("kW", "").replace(" ", "")) - total_kw
    ))



def state_encoder_chauffage(number_of_people: int,
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

# ==== Import agents and utils (you already have these) ====
from model.RL_model import agent_

# ---------- Shared utilities ----------
with open("catalog/catalog3.json", "r") as f:
    catalog = json.load(f)

# =============== STATE MODELS ==================
class input_on(BaseModel):
    surface_area: float
    monthly_consumption: float
    building_type: Literal["appartement", "maison", "villa"]
class State_on(BaseModel):
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


class State_off(BaseModel):
    nb_personnes: int = Field(ge=1, le=10)
    surface_m2: float = Field(ge=1, le=200)
    conso_totale_kwh_mois: float = Field(ge=1, le=3000)
    high: int = Field(ge=1, le=3)
    medium: int = Field(ge=1, le=3)
    low: int = Field(ge=1, le=3)
    panels: int = Field(ge=1, le=4)
    kits: int = Field(ge=0, le=1)
    batteries: int = Field(ge=0, le=2)


# ---- appliance classification map ----
APPLIANCE_MAP = {
    "Réfrigérateur": "medium",
    "Télévision": "medium",
    "Ordinateur": "medium",
    "Climatiseur": "high",
    "Machine à laver": "high",
    "Micro-ondes": "high",
    "Éclairage LED": "low",
    "Chargeur de téléphone": "low",
    # "Autre": default → medium
}
def package_to_state_off(pkg: dict) -> State_off:
    """Convert frontend package to State_off instance."""
    nb_personnes = int(pkg.get("nb_personnes", 1))
    surface = float(pkg.get("surface_m2", 1))
    conso = float(pkg.get("conso_totale_kwh_mois", 1))

    # start counters
    low_count, med_count, high_count = 0, 0, 0

    for app in pkg.get("appliances", []):
        nom = app.get("name", "Autre")
        quantite = int(app.get("quantity") or 0)

        category = APPLIANCE_MAP.get(nom, "medium")
        if category == "low":
            low_count += quantite
        elif category == "high":
            high_count += quantite
        else:
            med_count += quantite

    # ---- ensure defaults if nothing selected ----
    if low_count == 0 and med_count == 0 and high_count == 0:
        low_count, med_count, high_count = 1, 1, 1

    # clamp to Pydantic ranges (min=1, max=3)
    low_count = max(1, min(low_count, 3))
    med_count = max(1, min(med_count, 3))
    high_count = max(1, min(high_count, 3))

    return State_off(
        nb_personnes=nb_personnes,
        surface_m2=surface,
        conso_totale_kwh_mois=conso,
        high=high_count,
        medium=med_count,
        low=low_count,
        panels=1,      # start minimal system
        kits=0,
        batteries=0
    )


def monthly_prod_from_panels(panels: int, irradiance_kwh_m2_day: float = 4.0) -> float:
    # 0.610 kW per panel * PR(0.8) * irradiance * ~30 days
    return (panels * 0.610) * 0.80 * irradiance_kwh_m2_day * 30

class StateChauffage(BaseModel):
    number_of_people: int
    intensity: Literal["faible", "moyen", "forte"]
    has_kids_or_elderly: bool
    building_type: Literal["appartement", "maison", "villa"]

class StatePompage(BaseModel):
    puissance: float
    hmt: float
    debit: float



def generate_explanation_with_gpt_on(s : State_on ):
    
    client = OpenAI(
            base_url=ENDPOINT,
            api_key=GITHUB_TOKEN
        )

    prompt = (
        f"You are a solar heating assistant helping customers choose .\n"
        f"The user has the following needs:\n"
        f"- surface to be covered : {s.surface_area}\n"
        f"- building type: {s.building_type}\n"
        f"- monthly_consumptio: {s.monthly_consumption}\n"
        f"- number of panels recommended: {s.panels}\n"
        f"- an onduleur and compteur biderictionel the user can see their name\n"

        f"- Capacity: each panel is 0,61 w inwo \n"
        f"Explain in one sentence why this kit is a good match for the user. make sure your response is clear and interesting , and talk about the side things needed also like a structure "
    )
    response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0
        )


    return response.choices[0].message.content.strip()




def generate_explanation_with_gpt_chauffage(kit,data ):
    
    client = OpenAI(
            base_url=ENDPOINT,
            api_key=GITHUB_TOKEN
        )

    prompt = (
        f"You are a solar heating assistant helping customers choose .\n"
        f"The user has the following needs:\n"
        f"- Number of people: {data.number_of_people}\n"
        f"- Building type: {data.building_type}\n"
        f"- Usage intensity: {data.intensity}\n"
        f"- Special needs (children/elderly): {'yes' if data.has_kids_or_elderly else 'no'}\n\n"
        f"The recommended kit has:\n"
        f"- Capacity: {kit['capacity']} L\n"
        f"Explain in one sentence why this kit is a good match for the user. make sure your response is clear and interesting"
    )
    response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0
        )


    return response.choices[0].message.content.strip()



def generate_explanation_with_gpt_pompage(kit,data ):
    
    client = OpenAI(
            base_url=ENDPOINT,
            api_key=GITHUB_TOKEN
        )

    prompt = (
        f"You are a solar heating assistant helping customers choose .\n"
        f"The user has the following needs for a pompage set up:\n"
        f"- puissance: {data.puissance}\n"
        f"- hmt (depth): {data.hmt}\n"
        f"- debit: {data.debit}\n"
        f"The recommended kit is:\n"
        f"- kit: {kit['name']} L\n"
        f"Explain in one sentence why this kit is a good match for the user. make sure your response is clear and interestings"
    )
    response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0
        )


    return response.choices[0].message.content.strip()












def generate_explanation_with_gpt_offgrid(solution, state):
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=GITHUB_TOKEN
    )

    prompt = (
        f"You are a solar off-grid assistant helping customers choose.\n"
        f"The user has the following needs:\n"
        f"- household size: {state.nb_personnes} personnes\n"
        f"- surface area: {state.surface_m2} m²\n"
        f"- monthly consumption: {state.conso_totale_kwh_mois} kWh\n"
        f"- appliances profile: high={state.high}, medium={state.medium}, low={state.low}\n\n"
        f"The recommended off-grid solution is:\n"
        f"- {solution['recommended_model']}\n"
        f"Explain in one sentence why this solution is a good match for the user. "
        f"Make sure your response is clear and interesting."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        top_p=1.0
    )

    return response.choices[0].message.content.strip()


def target_monthly_energy_kwh_on(building_type: str,
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
def target_monthly_energy_kwh_off(building_type: str,
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

def _monthly_energy_panels(panels: int) -> float:
    # 610 W/panel, PR=0.80, 4 kWh/m²/day, ~30 days
    return (panels * 0.610) * 0.80 * 4.0 * 30
def is_terminal_on(state: State_on) -> bool:
    target = target_monthly_energy_kwh_on(
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



def state_encoder_pompage(input_ : StatePompage) -> int:
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


SURF_BASE   = 4
CONS_BASE   = 3
BUILD_BASE  = 3
PAN_BASE    = 10
KITS_BASE   = 2
COMP_BASE   = 2

NUM_STATES = SURF_BASE * CONS_BASE * BUILD_BASE * PAN_BASE * KITS_BASE * COMP_BASE  # 1440

BUILDING_MAP = {"appartement": 0, "maison": 1, "villa": 2}
def to_indx_on_grid(s: State_on) -> int:
    # Clamp continuous
    sa = max(0.0, min(200.0, s.surface_area))
    mc = max(0.0, min(1000.0, s.monthly_consumption))

    # Binning
    surf_i = 0 if sa < 50 else 1 if sa < 100 else 2 if sa < 150 else 3
    cons_i = 0 if mc < 300 else 1 if mc < 600 else 2
    bldg_i = BUILDING_MAP[s.building_type]

    # Clamp discrete BEFORE encoding
    pan   = min(max(s.panels,   1), 4)
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


#---- Indexing function ----
def to_indx(s: State_off) -> int:
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


# ---- Reward (step & terminal) ----
def reward_off(state: State_off, action: int, terminal: bool = False):
    base_target = target_monthly_energy_kwh_off(
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
        category, idx, item = action_space_off[action]

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
# =============== Q TABLE HELPERS ==================
def load_qtable(model_id: int):
    path = f"tables/tables{model_id}.npy"
    return np.load(path, allow_pickle=True)

def choose_action(q_table, state_idx):
    action = int(np.argmax(q_table[state_idx]))
    q_val = float(q_table[state_idx][action])
    return {"state_index": state_idx, "chosen_action": action, "q_value": q_val}

# =============== FASTAPI APP ==================
app = FastAPI(title="Helia Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["*"] pour tout autoriser
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS...
    allow_headers=["*"],
)
# =============== ON-GRID ==================
@app.post("/on-grid")

def predict_on_grid(state: input_on):
    # Start state (others fixed to 0)
    s = State_on(surface_area=state.surface_area,monthly_consumption= state.monthly_consumption, building_type= state.building_type, panels=1, kits= 0, compteur= 0)

    agent = agent_(nb_state=1440 , nb_action=NB_ACTIONS_off,
                   decay=0.96, epsilon=0,
                   alpha=0.1, gamma=0.7,
                   path="tables/tables3.npy")
    steps = []
    terminal = False

    while not terminal:
        idx = to_indx_on_grid(s) 
        action = agent.choose_action( idx)
        action=int(action)
        steps.append(action)
        if action < 3 and action >= 0 :
            s.kits += 1 
        elif action == 3 :
            s.panels +=1 
        else :
            s.compteur +=1 
        terminal=is_terminal_on(s)
        # For now stop after 2 steps to avoid infinite loop
        if len(steps) >= 10:
            terminal = True


    if s.kits == 0:
        n_panel=s.panels
        ondul=choose_onduleur_on(n_panels=s.panels, onduleurs_dict=onduleur_on)
        exp = generate_explanation_with_gpt_on(s)
        solution = {
            "recommended_model": f"{s.panels} panneaux + {s.compteur} compteur(s) + onduleur {ondul["name"]}",
            "capacity_l": round((s.panels * 0.610) * 0.80 * 4 * 30),  # monthly kWh approx
            "brand": "HELIANTHA On-Grid (panneaux unitaires)",
            "quantity": s.panels,
            "price_mad": s.panels * catalog["panneaux"][0]["price_solution"],
            "explanation": exp
        }
    else:
        kit_idx = min(action, len(catalog["kit_on_grid"]) - 1)
        kit = catalog["kit_on_grid"][kit_idx]
        solution = {
            "recommended_model": kit["name"],
            "capacity_l": kit["capacity_solution"],
            "brand": "HELIANTHA Kit On-Grid",
            "quantity": 1,
            "price_mad": kit["price_solution"],
            "explanation": (
                f"Basé sur votre consommation et surface, "
                f"nous recommandons directement le {kit['name']}."
            )
        }

    return {"solution": solution}
# =============== OFF-GRID ==================


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
def to_indx(s: State_off) -> int:
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

# ---- Terminal condition for OFF-GRID ----
def is_terminal(state: State_off, max_steps_reached: bool = False) -> bool:
    if state.kits > 0:
        return True

    # custom system: must have panels + ≥1 battery
    has_core_system = (state.panels > 0 and state.batteries > 0)

    if has_core_system:
        base_target = target_monthly_energy_kwh_off(
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



@app.post("/off-grid")
def predict_off_grid(state: dict):
    actions = []
    step = 0
    agent = agent_(
        nb_state=NUM_STATES,
        nb_action=NB_ACTIONS_off,
        decay=0.96,
        epsilon=0,
        alpha=0.1,
        gamma=0.7,
        path="tables/tables_offgrid.npy"
    )

    state_ = package_to_state_off(pkg=state)
    terminal = False  

    while not terminal and step < 5:
        idx = to_indx(state_) 
        action = agent.choose_action(idx)
        action = int(action)
        actions.append(action)
        step += 1
        _, state_ = reward_off(action=action, state=state_, terminal=terminal)
        terminal = is_terminal(state_, max_steps_reached=(step >= 5))

    # ==========================
    # Build solution from chosen actions
    # ==========================
    chosen_items = []
    for a in actions:
        category, idx, item = action_space_off[a]
        chosen_items.append({
            "category": category,
            "name": item.get("nom") or item.get("id"),
            "details": item,
        })

    kit_choices = [ci for ci in chosen_items if ci["category"] == "kits_off_grid"]

    if kit_choices:
        # take last kit chosen
        kit = kit_choices[-1]["details"]
        solution = {
            "recommended_model": kit["nom"],
            "capacity_l": f"{kit['composition']['panneaux']['nombre']} panneaux x {kit['composition']['panneaux']['puissance_individuelle_W']}W",
            "brand": "HELIANTHA Off-Grid (Kit complet)",
            "quantity": 1,
            "price_mad": kit["prix_MAD"],
            "explanation": (
                f"Ce kit inclut {kit['composition']['panneaux']['nombre']} panneaux, "
                f"{kit['composition']['batterie']['nombre']} batteries {kit['composition']['batterie']['type']} "
                f"et un onduleur de {kit['composition']['onduleur']['puissance_W']}W pour couvrir vos besoins."
            )
        }
    else:
        panneaux = [ci for ci in chosen_items if ci["category"] == "panneaux"]
        batteries = [ci for ci in chosen_items if ci["category"] == "batteries"]

        total_panels = len(panneaux)
        total_batteries = len(batteries)
        total_price = sum(ci["details"].get("prix_MAD", 0) for ci in chosen_items)

        exp = (
            f"Système construit avec {total_panels} panneau(x) "
            f"et {total_batteries} batterie(s), sélectionné(s) selon vos besoins énergétiques."
        )

        solution = {
            "recommended_model": ", ".join(ci["name"] for ci in chosen_items),
            "capacity_l": round(monthly_prod_from_panels(total_panels)),  # kWh/mois approx
            "brand": "HELIANTHA Off-Grid (sur mesure)",
            "quantity": len(chosen_items),
            "price_mad": total_price,
            "explanation": exp
        }

    return {"solution": solution, "chosen_items": chosen_items}

# =============== CHAUFFAGE ==================
@app.post("/chauffage")
def predict_chauffage(state: StateChauffage):
    id=state_encoder_chauffage(number_of_people=state.number_of_people,intensity=state.intensity,has_kids_or_elderly= state.has_kids_or_elderly,building_type=state.building_type)
    agent=agent_2(nb_state=576,nb_action=6,decay=0.96,epsilon=0,alpha=0.1,gamma=0.7,path="tables/tables3.npy") #alpha being 0 won t affect convergence because we perform only one step searsh in the future 
    action=agent.choose_action(id)
    explanation = generate_explanation_with_gpt_chauffage(data=state, kit = chauffage_catalog[action])

    solution = {
            "recommended_model": chauffage_catalog[action]["name"],
            "capacity_l": chauffage_catalog[action]["capacity"],
            "brand": chauffage_catalog[action]["brand"],
            "quantity": 1,
            "price_mad": chauffage_catalog[action]["price_mad"],
            "explanation":explanation}
    
    return {"solution": solution}

    


# =============== POMPAGE ==================
@app.post("/pompage")
def predict_pompage(state: StatePompage):
    agent_pompage=agent_2(125,6,decay=0.999,epsilon=0, gamma = 0,path ="tables/q_table2.npy")
    idx=state_encoder_pompage(state)
    result = agent_pompage.choose_action(idx)
    exp=generate_explanation_with_gpt_pompage(kit = catalog_pompage[result] , data = state)
    solution = {
        "recommended_model": catalog_pompage[result]["name"],
        "capacity_l": f"{catalog_pompage[result]["voltage"]} V",
        "brand": f"kit solaire a variateur {catalog_pompage[result]["power"]} KW",
        "quantity": 1,
        "price_mad": catalog_pompage[result]["price"],
        "explanation":  exp 
                       
    }

    return {"solution": solution}

    
# =============== HYBRIDE ==================



@app.post("/hybride")
def predict_hybride(package: dict):

    # ---- Extract info from request ----
    nb_personnes = int(package.get("nbPersonnes", 1))
    surface = float(package.get("surface", 0))
    conso_mensuelle = float(package.get("consommationMensuelleKWh", 0))

    appareils = []
    for app in package.get("appareils", []):
        appareils.append({
            "nom": app.get("nom"),
            "puissance": float(app.get("puissance", 0)),
            "duree": float(app.get("duree", 0))
        })

    # ---- Hybrid catalog (for GPT choice) ----
    catalog = {
        "panneaux": [
            { "id": "PAN-610", "nom": "Panneau Monocristallin 610W", "puissance_W": 610, "prix_MAD": 950 },
            { "id": "PAN-450", "nom": "Panneau Monocristallin 450W", "puissance_W": 450, "prix_MAD": 720 }
        ],
        "onduleurs_hybrides": [
            { "id": "HYB-3K", "nom": "Onduleur Hybride 3 kW 48V", "puissance_W": 3000, "prix_MAD": 11500 },
            { "id": "HYB-5K", "nom": "Onduleur Hybride 5 kW 48V", "puissance_W": 5000, "prix_MAD": 15500 }
        ],
        "batteries": [
            { "id": "BAT-LI5", "nom": "Batterie Lithium 5.12 kWh 48V", "capacite_kWh": 5.12, "prix_MAD": 21000 },
            { "id": "BAT-LI10", "nom": "Batterie Lithium 10 kWh 48V", "capacite_kWh": 10, "prix_MAD": 39500 },
            { "id": "BAT-GEL200", "nom": "Batterie GEL 200Ah 12V", "capacite_Ah": 200, "prix_MAD": 3500 }
        ],
        "regulateurs": [
            { "id": "MPPT-60A", "nom": "Régulateur MPPT 60A 48V", "prix_MAD": 3200 },
            { "id": "MPPT-100A", "nom": "Régulateur MPPT 100A 48V", "prix_MAD": 4800 }
        ],
        "accessoires": [
            { "id": "CAB-10MM", "nom": "Câble solaire 10 mm² (50 m)", "prix_MAD": 950 },
            { "id": "STRUC-4P", "nom": "Structure acier galvanisé pour 4 panneaux", "prix_MAD": 1800 }
        ]
    }

    # ---- Call GPT to decide (kit or setup) ----
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=GITHUB_TOKEN
    )

    prompt = (
        f"You are a solar hybrid assistant helping customers choose.\n\n"
        f"The user has the following needs:\n"
        f"- Household size: {nb_personnes} personnes\n"
        f"- Surface area: {surface} m²\n"
        f"- Monthly consumption: {conso_mensuelle} kWh\n"
        f"- Appliances: {[a['nom'] for a in appareils]}\n\n"
        f"You have access to this catalog:\n"
        f"- Panneaux: {[p['nom']+' '+str(p['puissance_W'])+'W' for p in catalog['panneaux']]}\n"
        f"- Onduleurs hybrides: {[o['nom'] for o in catalog['onduleurs_hybrides']]}\n"
        f"- Batteries: {[b['nom'] for b in catalog['batteries']]}\n"
        f"- Régulateurs: {[r['nom'] for r in catalog['regulateurs']]}\n"
        f"- Accessoires: {[a['nom'] for a in catalog['accessoires']]}\n\n"
        f"👉 Task:\n"
        f"- Decide if the best solution is to recommend one kit (if present) "
        f"OR to build a custom hybrid system from catalog items.\n"
        f"- Always output exactly ONE recommendation.\n"
        f"- Use only items from the catalog.\n"
        f"- Output in this JSON structure:\n"
        f"{{\n"
        f"  'recommended_model': 'string',\n"
        f"  'capacity_l': 'int (approx monthly kWh)',\n"
        f"  'brand': 'HELIANTHA Hybride',\n"
        f"  'quantity': int,\n"
        f"  'price_mad': int,\n"
        f"  'explanation': 'one clear sentence'\n"
        f"}}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful solar assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        top_p=1.0
    )

    import json
    try:
        solution = json.loads(response.choices[0].message.content.strip().replace("'", '"'))
    except Exception:
        solution = {"explanation": response.choices[0].message.content.strip()}

    return {"solution": solution}

# =============== ROOT ==================




    





@app.get("/")
def root():
    return {"message": "Helia Agent API running 🚀"}
