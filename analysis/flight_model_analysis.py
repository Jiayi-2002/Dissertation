import matplotlib.pyplot as plt
import jax.numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.aircraft_model import AircraftModel
from flight.analysis.results_plotting import crop_and_save
from flight.simulator import config

# =========

# Load aircraft model
flight_model_name = 'wot4_imav_v2'
model_path = config.base_aircraft_model_path / f'{flight_model_name}.yaml'
am = AircraftModel(model_path)

base_save_folder = Path('.') / 'analysis' / 'flight_model_graphs' / flight_model_name
base_save_folder.mkdir(exist_ok=True, parents=True)

# =========

# NOTE [Important] All of the CL and CD calculations are performed with q = 0 and control surface deflections = 0.

max_alpha = am.max_alpha
min_alpha = am.min_alpha
print(min_alpha, max_alpha)

# Calculate different CL and CD values for different angle of attack values
alphas = np.linspace(min_alpha, max_alpha, 1000)
CLs = am.CL(alphas, 0, 0, 0)
CDs = am.CD(alphas, 0, 0, 0)

# Find best lift-to-drag ratio and the angle of attack at which it occurs
CLCDs = CLs/CDs
best_CLCD_ind = np.argmax(CLCDs)
best_CLCD_alpha = alphas[best_CLCD_ind]
best_CLCD = CLCDs[best_CLCD_ind]

# =========
# Plot coefficients of lift and drag against alpha (same figure)
fig, axs = plt.subplots(2)
axs[0].plot(np.degrees(alphas), CLs, label='$C_L$')
# axs[0].axvline(np.degrees(max_alpha), c='red', ls='--')
# axs[0].axvline(np.degrees(min_alpha), c='red', ls='--')
axs[0].axhline(0, c='grey', ls='--', alpha=0.1)
axs[0].axvline(0, c='grey', ls='--', alpha=0.1)
axs[0].set_xlabel(r"$\alpha$ ($^\circ$)")
axs[0].set_ylabel("Lift coefficient")
axs[0].set_title("Lift coefficient")

axs[1].plot(np.degrees(alphas), CDs, label='$C_D$')
# axs[1].axvline(np.degrees(max_alpha), c='red', ls='--')
# axs[1].axvline(np.degrees(min_alpha), c='red', ls='--')
axs[1].set_ylim(0, np.max(CDs)*1.05)
axs[1].axvline(0, c='grey', ls='--', alpha=0.1)
axs[1].set_xlabel(r"$\alpha$ ($^\circ$)")
axs[1].set_ylabel("Drag coefficient")
axs[1].set_title("Drag coefficient")

fig.suptitle("Aerodynamic coefficient functions")
fig.tight_layout()

crop_and_save(fig, base_save_folder / 'aero_coeff_fns')

# =========
# Plot L/D against alpha and CL against CD
fig, axs = plt.subplots(2)
axs[0].plot(np.degrees(alphas), CLCDs, label='$C_L/C_D$')
# axs[0].axvline(np.degrees(max_alpha), c='red', ls='--')
# axs[0].axvline(np.degrees(min_alpha), c='red', ls='--')
axs[0].axhline(0, c='grey', ls='--', alpha=0.1)
axs[0].axvline(0, c='grey', ls='--', alpha=0.1)
# Plot best CL/CD point
axs[0].axhline(best_CLCD, c='grey', ls=':', alpha=0.4)
axs[0].axvline(np.degrees(best_CLCD_alpha), c='grey', ls=':', alpha=0.4)
axs[0].scatter(np.degrees(best_CLCD_alpha), best_CLCD, c='orange', edgecolors='black', zorder=10)
# axs[0].annotate(f"({np.round(np.degrees(best_CLCD_alpha), 2)}$^\circ$, {np.round(best_CLCD, 2)})", (np.degrees(best_CLCD_alpha)+0.3, best_CLCD-0.3), xytext=(10, 2), arrowprops=dict(arrowstyle="->"))
axs[0].annotate(f"{np.round(best_CLCD, 2)} @ {np.round(np.degrees(best_CLCD_alpha), 2)}$^\circ$", (np.degrees(best_CLCD_alpha)+0.3, best_CLCD-0.3), xytext=(10, 2), arrowprops=dict(arrowstyle="->"))
axs[0].set_xlabel(r"$\alpha$ ($^\circ$)")
axs[0].set_ylabel("$C_L/C_D$")
axs[0].set_title("$C_L/C_D$")

axs[1].plot(CDs, CLs)
axs[1].axhline(0, c='grey', ls='--', alpha=0.1)
axs[1].set_xlim(0, np.max(CDs)*1.1)
axs[1].set_xlabel("Drag coefficient")
axs[1].set_ylabel("Lift coefficient")
axs[1].set_aspect(1/20)
axs[1].set_title("$C_L$ vs $C_D$")

fig.suptitle("Lift to drag performance")
fig.tight_layout()

crop_and_save(fig, base_save_folder / 'l_to_d_perf')

# =========
# Calculate and plot glide polar
# Not sure this is the right approach, but I'm going to try it.
# For alpha, find CL/CD -> get gamma -> calculate Va. Then will get a load of (Va, gamma) pairs.
pos_alphas = np.linspace(0, max_alpha, 1000)
pos_CLs = am.CL(pos_alphas, 0, 0, 0)
pos_CDs = am.CD(pos_alphas, 0, 0, 0)
gammas = np.arctan(pos_CDs/pos_CLs)

fig, ax = plt.subplots()
ax.plot(np.degrees(pos_alphas), np.degrees(gammas))
ax.set_xlabel(r"$\alpha$ ($^\circ$)")
ax.set_ylabel(r"$\gamma$ ($^\circ$)")

vas = []
for alpha, gamma in zip(pos_alphas, gammas):
    print(np.degrees(alpha), np.degrees(gamma))
    CL = am.CL(alpha, 0, 0, 0)
    print(CL)
    va = np.sqrt((2*am.m*am.g*np.cos(gamma))/(am.rho*am.S*CL))
    print(va)
    vas.append(va)
vas = np.array(vas)

va_restrict_inds = np.where(vas < am.max_va)

vas_restricted = vas[va_restrict_inds]
alphas_restricted = pos_alphas[va_restrict_inds]
gammas_restricted = gammas[va_restrict_inds]

fig, ax = plt.subplots()
ax.plot(np.degrees(gammas_restricted), vas_restricted)
ax.set_xlabel(r"$\gamma$ ($^\circ$)")
ax.set_ylabel(r"$v_a$ ($ms^{-1}$)")

sink = vas_restricted*np.sin(gammas_restricted)

fig, ax = plt.subplots()
ax.plot(vas_restricted, sink)
# ax.plot(ground_proj_speed, sink)
ax.set_xlabel("$v_a$ ($ms^{-1}$)")
ax.set_ylabel("Sink rate ($ms^{-1}$)")
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
ax.invert_yaxis()
# ax.set_xlim(0, am.max_va)
# ax.set_ylim(0, 20)

# Highlight and annotate the minimum sink speed
# Need sink speed and the airspeed at which it occurs
min_sink_ind = np.argmin(sink)
min_sink = sink[min_sink_ind]
min_sink_va = vas_restricted[min_sink_ind]
ax.scatter(min_sink_va, min_sink, c='orange', edgecolors='black', zorder=10)
ax.annotate(f"Min sink\n${np.round(min_sink, 2)}\,ms^{{-1}}$ @ ${np.round(min_sink_va, 2)}\,ms^{{-1}}$", (min_sink_va, min_sink+0.1), xytext=(4, 4), arrowprops=dict(arrowstyle="->"))

# Identifying the best L/D
# Find the ratio for all points and pick the one with the best value
# Then just find a line through it and draw it
best_ld_ind = np.argmin(sink/vas_restricted)
best_ld_sink = sink[best_ld_ind]
best_ld_va = vas_restricted[best_ld_ind]
ax.scatter(best_ld_va, best_ld_sink, c='orange', edgecolors='black', zorder=10)
ax.annotate(f"Best glide (L/D)\n${np.round(best_ld_sink, 2)}\,ms^{{-1}}$ @ ${np.round(best_ld_va, 2)}\,ms^{{-1}}$", (best_ld_va+0.3, best_ld_sink), xytext=(20, best_ld_sink), arrowprops=dict(arrowstyle="->"))

# Calculate the line from the origin
xs = np.linspace(0, am.max_va, 1000)
ys = (best_ld_sink/best_ld_va)*xs
ax.plot(xs, ys, c='grey', ls='--', alpha=0.3)

fig.suptitle("Glide polar")

crop_and_save(fig, base_save_folder / 'glide_polar')

# =========
# Plot glide angle
fig, ax = plt.subplots()
ax.plot(vas_restricted, np.degrees(gammas_restricted))
# ax.plot(ground_proj_speed, sink)
ax.set_xlabel("$v_a$ ($ms^{-1}$)")
ax.set_ylabel(r"$\gamma$ ($^\circ$)")
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_ylim(0, ax.get_ylim()[1])
# ax.set_xlim(0, am.max_va)
# ax.set_ylim(0, 20)
fig.suptitle("Glide angle")

crop_and_save(fig, base_save_folder / 'glide_angle')

plt.show()