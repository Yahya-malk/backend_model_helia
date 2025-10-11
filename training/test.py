from model.RL_model import agent_

agent=agent_(360,7,decay=0.9999,epsilon=0, gamma = 0,path = "tables/q_table.npy")



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


id1=state_encoder(1,"faible",True,"appartement")
action=agent.choose_action(id1)
print(action)