import numpy as np
import pandas as pd

# Fixed parameters
S = 0.3              # Wing area (m²)
m_total = 1.345      # Total mass (kg)
m_wing0 = 0.4035     # Wing mass (kg)
m_body0 = m_total - m_wing0  # Fuselage mass
c_paper = 0.254  
b_baseline = 1.206

# Paper values
Ixx_paper = 0.051
Iyy_paper = 0.078
Izz_paper = 0.112

# Aileron effectiveness
Cl_da_baseline = -2.11e-02

# Aerodynamic coefficients
alpha_0 = 4.44e-03        # Zero-lift AoA (rad)
Cl_alpha = 3.89           # Lift curve slope (1/rad)
Cl_q = 1.04e-01          # Pitch damping
Cl_de = -4.24e-01        # Elevator effectiveness
Cd_0 = 0.03              # Zero-lift drag
e_oswald = 0.7           # Oswald efficiency

# Constants
g = 9.81                 # Gravity (m/s²)
rho = 1.225             # Air density (kg/m³)

# Wing inertia estimation (35% contribution)
wing_contribution_factor = 0.35

Ixx_wing_estimated = Ixx_paper * wing_contribution_factor
Iyy_wing_estimated = Iyy_paper * wing_contribution_factor  
Izz_wing_estimated = Izz_paper * wing_contribution_factor

print(f"Estimated wing inertia contributions ({wing_contribution_factor*100}% of total):")
print(f"  Ixx_wing: {Ixx_wing_estimated:.6f}")
print(f"  Iyy_wing: {Iyy_wing_estimated:.6f}")
print(f"  Izz_wing: {Izz_wing_estimated:.6f}")

# Calculate body inertia
Ixx_body = Ixx_paper - (1/12) * m_wing0 * b_baseline**2
Iyy_body = Iyy_paper - (1/12) * m_wing0 * c_paper**2
Izz_body = Izz_paper - (1/12) * m_wing0 * (b_baseline**2 + c_paper**2)

print(f"\nCalculated body inertia:")
print(f"  Ixx_body: {Ixx_body:.6f}")
print(f"  Iyy_body: {Iyy_body:.6f}")
print(f"  Izz_body: {Izz_body:.6f}")

# Target wingspans
b_targets = [0.7, 0.9, 1.206, 1.5, 1.8, 2.0]
geom_data = []

for b in b_targets:
    c_actual = S / b  # Chord varies with wingspan
    m_wing = m_wing0
    print(f"b: {b:.6f}")
    print(f"c_actual: {c_actual:.6f}")
    
    # Wing inertia (flat plate model)
    Ixx_wing = (1/12) * m_wing * b**2
    Iyy_wing = (1/12) * m_wing * c_actual**2
    Izz_wing = (1/12) * m_wing * (b**2 + c_actual**2)

    # Total inertia
    Ixx_total = Ixx_body + Ixx_wing
    Iyy_total = Iyy_body + Iyy_wing
    Izz_total = Izz_body + Izz_wing

    # Aileron effectiveness scaling
    Cl_da_scaled = Cl_da_baseline * (b / b_baseline)
    
    # Aerodynamic performance
    AR = b**2 / S
    
    # Optimal L/D calculation
    CL_opt = np.sqrt(np.pi * AR * e_oswald * Cd_0)
    CD_opt = Cd_0 + (CL_opt**2) / (np.pi * AR * e_oswald)
    max_L_D = CL_opt / CD_opt
    
    # Optimal angle of attack
    alpha_opt_deg = np.degrees((CL_opt / Cl_alpha) - alpha_0)

    geom_data.append({
        'Wingspan (m)': b,
        'Chord (m)': c_actual,
        'Aspect Ratio': b**2 / S,
        'Ixx (kg·m²)': Ixx_total,
        'Iyy (kg·m²)': Iyy_total,
        'Izz (kg·m²)': Izz_total,
        'Cl_da': Cl_da_scaled,
        'Max L/D': max_L_D,
        'Optimal AoA (deg)': alpha_opt_deg,
        'CL_opt': CL_opt,
        'CD_opt': CD_opt,
        'Total Mass (kg)': m_total,
        'Wing Mass (kg)': m_wing
    })

# Results
df_geom = pd.DataFrame(geom_data).round(6)

print("\nInertia and Control Results:")
print(df_geom[['Wingspan (m)', 'Ixx (kg·m²)', 'Iyy (kg·m²)', 'Izz (kg·m²)', 'Cl_da']].to_string(index=False))

print("\nAerodynamic Performance Results:")
print(df_geom[['Wingspan (m)', 'Aspect Ratio', 'Max L/D', 'Optimal AoA (deg)', 'CL_opt', 'CD_opt']].to_string(index=False))

print(f"\nAileron Effectiveness:")
print(f"  Baseline Cl_da (1.206m): {Cl_da_baseline:.6f}")
for _, row in df_geom.iterrows():
    print(f"  {row['Wingspan (m)']}m: Cl_da = {row['Cl_da']:.6f} (scale: {row['Wingspan (m)']/b_baseline:.2f})")

print(f"\nMax L/D Trends:")
for _, row in df_geom.iterrows():
    improvement = ((row['Max L/D'] / df_geom[df_geom['Wingspan (m)'] == b_baseline]['Max L/D'].iloc[0]) - 1) * 100
    print(f"  {row['Wingspan (m)']}m: L/D = {row['Max L/D']:.2f} (α = {row['Optimal AoA (deg)']:.1f}°, {improvement:+.1f}%)")

print(f"\nAspect Ratio Performance:")
for _, row in df_geom.iterrows():
    print(f"  AR {row['Aspect Ratio']:.1f}: L/D = {row['Max L/D']:.2f}, CL = {row['CL_opt']:.3f}, CD = {row['CD_opt']:.4f}")

