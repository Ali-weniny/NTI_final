import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============ Load Models ============
overall_model = joblib.load('models/overall_model.pkl')
overall_features = joblib.load('models/overall_features.pkl')
value_model = joblib.load('models/value_model.pkl')
value_features = joblib.load('models/value_features.pkl')

st.set_page_config(page_title="FIFA Player Value Predictor", layout="wide")
st.title("⚽ Player Market Value Predictor")
st.markdown("Enter the player's attributes below. The Overall rating will be calculated first, then used to predict the Market Value.")

# ============ Section 1: Basic Player Info ============
st.header("📋 Basic Player Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 16, 45, 25)
    potential = st.slider("Potential", 40, 99, 75)
    height_cm = st.slider("Height (cm)", 150, 210, 180)
    weight_kg = st.slider("Weight (kg)", 50, 110, 75)

with col2:
    international_reputation = st.slider("International Reputation", 1, 5, 1)
    weak_foot = st.slider("Weak Foot Rating", 1, 5, 3)
    skill_moves = st.slider("Skill Moves", 1, 5, 2)
    league_level = st.slider("League Level (1 = Top Tier)", 1, 5, 1)

with col3:
    position_group = st.selectbox("Position Group", ["ATT", "MID", "DEF"])
    preferred_foot = st.selectbox("Preferred Foot", ["Right", "Left"])
    years_at_club = st.slider("Years at Current Club", 0.0, 20.0, 2.0)
    contract_years_remaining = st.slider("Contract Years Remaining", 0, 10, 3)

col4, col5 = st.columns(2)
with col4:
    is_loaned = st.selectbox("Is the player on loan?", ["No", "Yes"])
with col5:
    league_freq = st.slider("League Size (Frequency)", 1, 1000, 300,
                             help="Higher value = bigger/more popular league")
    nationality_freq = st.slider("Nationality Size (Frequency)", 1, 3000, 500,
                                  help="Higher value = more professional players from this nationality")

# ============ Section 2: Detailed Stats ============
st.header("⚙️ Detailed Technical Stats")
st.caption("These values will be used to calculate the Overall rating first")

with st.expander("Open to enter detailed stats", expanded=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.subheader("Main Stats")
        pace = st.slider("Pace", 1, 99, 70)
        shooting = st.slider("Shooting", 1, 99, 60)
        passing = st.slider("Passing", 1, 99, 65)
        dribbling = st.slider("Dribbling", 1, 99, 65)
        defending = st.slider("Defending", 1, 99, 50)
        physic = st.slider("Physic", 1, 99, 65)

    with c2:
        st.subheader("Attacking")
        attacking_crossing = st.slider("Crossing", 1, 99, 60)
        attacking_finishing = st.slider("Finishing", 1, 99, 60)
        attacking_heading_accuracy = st.slider("Heading Accuracy", 1, 99, 55)
        attacking_short_passing = st.slider("Short Passing", 1, 99, 65)
        attacking_volleys = st.slider("Volleys", 1, 99, 55)
        skill_dribbling = st.slider("Skill Dribbling", 1, 99, 65)
        skill_curve = st.slider("Curve", 1, 99, 55)
        skill_fk_accuracy = st.slider("FK Accuracy", 1, 99, 50)

    with c3:
        st.subheader("Movement & Power")
        skill_long_passing = st.slider("Long Passing", 1, 99, 60)
        skill_ball_control = st.slider("Ball Control", 1, 99, 65)
        movement_acceleration = st.slider("Acceleration", 1, 99, 70)
        movement_sprint_speed = st.slider("Sprint Speed", 1, 99, 70)
        movement_agility = st.slider("Agility", 1, 99, 65)
        movement_reactions = st.slider("Reactions", 1, 99, 65)
        movement_balance = st.slider("Balance", 1, 99, 65)
        power_shot_power = st.slider("Shot Power", 1, 99, 60)

    with c4:
        st.subheader("Mentality & Defending")
        power_jumping = st.slider("Jumping", 1, 99, 65)
        power_stamina = st.slider("Stamina", 1, 99, 70)
        power_strength = st.slider("Strength", 1, 99, 65)
        power_long_shots = st.slider("Long Shots", 1, 99, 55)
        mentality_aggression = st.slider("Aggression", 1, 99, 60)
        mentality_interceptions = st.slider("Interceptions", 1, 99, 50)
        mentality_positioning = st.slider("Positioning", 1, 99, 60)
        mentality_vision = st.slider("Vision", 1, 99, 60)
        mentality_penalties = st.slider("Penalties", 1, 99, 55)
        mentality_composure = st.slider("Composure", 1, 99, 60)
        defending_marking_awareness = st.slider("Marking Awareness", 1, 99, 50)
        defending_standing_tackle = st.slider("Standing Tackle", 1, 99, 50)
        defending_sliding_tackle = st.slider("Sliding Tackle", 1, 99, 50)

# ============ Predict Button ============
if st.button("🔮 Predict Market Value", type="primary", use_container_width=True):

    # Prepare Model 1 input (Overall)
    overall_input = pd.DataFrame([{
        'pace': pace, 'shooting': shooting, 'passing': passing, 'dribbling': dribbling,
        'defending': defending, 'physic': physic,
        'attacking_crossing': attacking_crossing, 'attacking_finishing': attacking_finishing,
        'attacking_heading_accuracy': attacking_heading_accuracy,
        'attacking_short_passing': attacking_short_passing, 'attacking_volleys': attacking_volleys,
        'skill_dribbling': skill_dribbling, 'skill_curve': skill_curve,
        'skill_fk_accuracy': skill_fk_accuracy, 'skill_long_passing': skill_long_passing,
        'skill_ball_control': skill_ball_control,
        'movement_acceleration': movement_acceleration, 'movement_sprint_speed': movement_sprint_speed,
        'movement_agility': movement_agility, 'movement_reactions': movement_reactions,
        'movement_balance': movement_balance,
        'power_shot_power': power_shot_power, 'power_jumping': power_jumping,
        'power_stamina': power_stamina, 'power_strength': power_strength,
        'power_long_shots': power_long_shots,
        'mentality_aggression': mentality_aggression, 'mentality_interceptions': mentality_interceptions,
        'mentality_positioning': mentality_positioning, 'mentality_vision': mentality_vision,
        'mentality_penalties': mentality_penalties, 'mentality_composure': mentality_composure,
        'defending_marking_awareness': defending_marking_awareness,
        'defending_standing_tackle': defending_standing_tackle,
        'defending_sliding_tackle': defending_sliding_tackle
    }])[overall_features]

    predicted_overall = overall_model.predict(overall_input)[0]

    # Prepare Model 2 input (Value) using the predicted Overall
    value_input = pd.DataFrame([{
        'predicted_overall': predicted_overall,
        'potential': potential,
        'age': age,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'league_level': league_level,
        'weak_foot': weak_foot,
        'skill_moves': skill_moves,
        'international_reputation': international_reputation,
        'years_at_club': years_at_club,
        'contract_years_remaining': contract_years_remaining,
        'is_loaned': 1 if is_loaned == "Yes" else 0,
        'nationality_freq': nationality_freq,
        'league_freq': league_freq,
        'preferred_foot_Right': 1 if preferred_foot == "Right" else 0,
        'position_group_DEF': 1 if position_group == "DEF" else 0,
        'position_group_MID': 1 if position_group == "MID" else 0,
    }])[value_features]

    predicted_value_log = value_model.predict(value_input)[0]
    predicted_value = np.expm1(predicted_value_log)   # reverse log1p transform

    # Show results
    st.success("Prediction complete!")
    res1, res2 = st.columns(2)
    with res1:
        st.metric("⭐ Predicted Overall", f"{predicted_overall:.1f}")
    with res2:
        st.metric("💰 Predicted Market Value", f"€ {predicted_value:,.0f}")