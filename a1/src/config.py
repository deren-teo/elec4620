
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Path to the project root
A1_ROOT = Path(__file__).parent.parent.resolve()

### PLOT CONFIG ################################################################

# Seaborn plot style customisation parameters
SNS_STYLE = "darkgrid"
SNS_PALETTE = "mako"

sns.set_palette(SNS_PALETTE, n_colors=2)
sns.set_style(SNS_STYLE)

# Matplotlib plot text customisation parameters
PLT_CONFIG = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
}

plt.rcParams.update(PLT_CONFIG)

# Matplotlib figure saving configuration
SAVEFIG_CONFIG = {
    "dpi": 300,
    "bbox_inches": "tight",
}
