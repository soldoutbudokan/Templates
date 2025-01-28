# %%
###############################################################################
# Imports
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

###############################################################################
# PATH SETUP
###############################################################################
# Base directory is two levels up from the code file
BASE_DIR = Path(__file__).parent.parent
# Output directory for figures
OUTPUT_DIR = BASE_DIR / 'output'
# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# 1) PARAMETERS
###############################################################################
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6))

# Demand: P = intercept - slope * (Q - shift)
#   => A positive 'shift' moves the demand curve LEFT by 'shift' units.
shift = 1
intercept = 15.0
slope = 2

# Key price points
p2 = 11.0   # Higher (monopoly) price
p1 = 7.5
mc = 4.0

# Derived Q-values:
#   P = intercept - slope*(Q - shift) => Q = shift + (intercept - P)/slope
q2 = shift + (intercept - p2) / slope   # ~ (2 + (15-11)/1.5) = 2 + 2.67 = 4.67
q1 = shift + (intercept - p1) / slope   # ~ (2 + (15-7.5)/1.5) = 2 + 5 = 7

###############################################################################
# 2) LAYOUT CONSTANTS (unchanged from your original code)
###############################################################################
x_left_margin   = -1.5
x_right_margin  = 7.9
y_lower_margin  = -1.0
y_upper_margin  = 16.0

mc_line_left    = -1.2 # Where MC curve cuts off
p_label_x       = x_left_margin - 0.19
mc_label_x      = -1.0
demand_label_x  = q2-1.25
demand_label_y  = p2+3        # We'll label near the top of the curve
q_label_y       = -0.75
surplus_label_x = 0.5 * (q2 + x_left_margin)
surplus_label_y = 0.5 * (p1 + p2)

###############################################################################
# 3) DEMAND & MC LINES
###############################################################################
# Demand curve array
x = np.linspace(1, x_right_margin-0.75, 200)
y = intercept - slope * (x - shift)  # SHIFT is applied here

plt.plot(x, y, 'black', linewidth=3)            # Demand
plt.hlines(mc, mc_line_left, x_right_margin-0.75, 
           color='black', linewidth=3)          # Marginal Cost

###############################################################################
# 4) HORIZONTAL & VERTICAL LINES
###############################################################################
# Horizontal lines for P2, P1 (stop at Q2, Q1)
plt.hlines(p2, x_left_margin, q2, color='black', linewidth=1)
plt.hlines(p1, x_left_margin, q1, color='black', linewidth=1)

# Vertical lines for Q2, Q1 (from y_lower_margin up to P2, P1)
plt.vlines(q2, y_lower_margin, p2, color='black', linewidth=1)
plt.vlines(q1, y_lower_margin, p1, color='black', linewidth=1)

###############################################################################
# 5) SHADED SURPLUS & DWL REGIONS
###############################################################################
# Surplus Transfer rectangle: from x_left_margin to x=q2, y in [p1, p2]
plt.fill(
    [x_left_margin, x_left_margin, q2, q2],
    [p1,            p2,            p2, p1],
    color='silver', alpha=0.3
)

# Consumer DWL (triangle): corners = (q2, p2), (q1, p1), (q2, p1)
plt.fill(
    [q2, q1, q2],
    [p2, p1, p1],
    color='grey', alpha=0.3
)

# Producer DWL (rectangle): corners = (q2, p1), (q1, p1), (q1, mc), (q2, mc)
plt.fill(
    [q2, q1, q1, q2],
    [p1, p1, mc, mc],
    color='grey', alpha=0.3
)

###############################################################################
# 6) LABELS
###############################################################################
# Price labels
plt.text(p_label_x, p2, 'P₂', va='center', weight='bold')
plt.text(p_label_x, p1, 'P₁', va='center', weight='bold')
plt.text(mc_label_x, mc + 0.3, 'Marginal Cost', va='center', weight='bold')

# Demand label near top-left
plt.text(demand_label_x, demand_label_y, 'Market Demand', va='center', weight='bold')

# Q2, Q1 labels slightly below axis
plt.text(q2 - 0.15, q_label_y, 'Q₂', ha='center', weight='bold')
plt.text(q1 - 0.15, q_label_y, 'Q₁', ha='center', weight='bold')

# Surplus Transfer label
plt.text(
    surplus_label_x, surplus_label_y,
    'Surplus Transfer from Consumers\nto Producers',
    ha='center', va='center'
)

# Consumer DWL label near centroid of the triangle
cx_c = (q2 + q1 + q2) / 3.0
cy_c = (p2 + p1 + p1) / 3.0
plt.text(cx_c, cy_c - 0.2, 'Consumer\nDWL', ha='center', va='center')

# Producer DWL label near midpoint of rectangle
px_c = 0.5 * (q2 + q1)
py_c = 0.5 * (mc + p1)
plt.text(px_c, py_c, 'Producer DWL', ha='center', va='center')

###############################################################################
# 7) FINAL FORMATTING
###############################################################################
plt.xlabel('Quantity', fontsize=14, fontweight='bold', labelpad=12)
plt.ylabel('Price',   fontsize=14, fontweight='bold', labelpad=12)

# Boxed chart
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# Add a note in the top-right corner (axes coordinates) with a simple bounding box
plt.text(
    0.95, 0.95,                # (x, y) in axes fraction (0=left/bottom, 1=right/top)
    "Assuming P₁ increases to P₂",
    ha='right', va='top',      # anchor text to top-right
    transform=ax.transAxes,    # place text relative to axes (not data)
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)

# Remove ticks for a cleaner look
plt.xticks([])
plt.yticks([])

# Visible region
plt.xlim(x_left_margin, x_right_margin)
plt.ylim(y_lower_margin, y_upper_margin)

plt.tight_layout()

###############################################################################
# 8) SAVE FIGURE
###############################################################################
# Save with high DPI for clarity
output_path = OUTPUT_DIR / 'DWL_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path.absolute()}")

plt.show()

plt.close()
