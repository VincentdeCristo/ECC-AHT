import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import cmcrameri.cm as crameri
from environment import SimulationWorld
from algorithms import ECC_AHT


# ======================================================
# Global plotting configuration
# ======================================================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'stix',
    'font.family': 'Times New Roman'
})

PANEL_FIGSIZE = (4.2, 3.0)


def save_panel(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


# ======================================================
# Utility
# ======================================================
def calculate_f1(S_hat, S_true):
    S_hat, S_true = set(S_hat), set(S_true)
    if not S_hat or not S_true:
        return 0.0
    tp = len(S_hat & S_true)
    precision = tp / len(S_hat)
    recall = tp / len(S_true)
    return 2 * precision * recall / (precision + recall + 1e-12)


# ======================================================
# Main visualization routine
# ======================================================
def visualize_ecc_aht_panels():
    np.random.seed(42)

    K = 15
    n = 3
    mu_signal = 3.0
    B = 4.0
    T_steps = 20
    rho = 0.6

    Sigma = toeplitz(rho ** np.arange(K))

    env = SimulationWorld(K, n, mu_signal, Sigma)
    algo = ECC_AHT(K, n, Sigma, B, mu_signal)

    beliefs_hist = []
    actions_hist = []

    for _ in range(T_steps):
        beliefs_hist.append(algo.p_t.copy())
        c_t = algo.select_action()
        actions_hist.append(c_t.copy())
        y_t = env.get_observation(c_t)
        algo.update(c_t, y_t)

    beliefs_hist.append(algo.p_t.copy())

    beliefs_hist = np.array(beliefs_hist)
    actions_hist = np.array(actions_hist)
    S_true = env.S_true

    
    # ==================================================
    # (a) Initial beliefs
    # ==================================================
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    colors = ['crimson' if k in S_true else 'steelblue' for k in range(K)]
    ax.bar(range(K), beliefs_hist[0], color=colors, alpha=0.8, linewidth=0.8)

    ax.axhline(n / K, ls='--', lw=1, color='gray')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(K))
    ax.set_xlabel('Stream $k$')
    ax.set_ylabel('$p_0(k)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = [
        Patch(facecolor='crimson', label='Anomaly'),
        Patch(facecolor='steelblue', label='Normal')
    ]
    ax.legend(handles=legend, loc='upper right', frameon=False)

    save_panel(fig, 'inside_ecc_aht_a.pdf')


    # ==================================================
    # (b) Champion–Challenger at t=1
    # ==================================================
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    b = beliefs_hist[1]
    S_hat = np.argsort(b)[-n:]
    i_star = S_hat[np.argmin(b[S_hat])]
    not_in_top_n = np.argsort(b)[:-n]
    j_star = not_in_top_n[np.argmax(b[not_in_top_n])]

    colors = []
    for k in range(K):
        if k == i_star:
            colors.append('gold')
        elif k == j_star:
            colors.append('orange')
        elif k in S_true:
            colors.append('crimson')
        else:
            colors.append('steelblue')

    ax.bar(range(K), b, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(K))
    ax.set_xlabel('Stream $k$')
    ax.set_ylabel('$p_1(k)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('Champion\n$i^*$', xy=(i_star, b[i_star]), 
                xytext=(i_star, b[i_star] + 0.15),
                ha='center', fontsize=9, color='darkgoldenrod',
                arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))
    ax.annotate('Challenger\n$j^*$', xy=(j_star, b[j_star]), 
                xytext=(j_star, b[j_star] + 0.15),
                ha='center', fontsize=9, color='darkorange',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    save_panel(fig, 'inside_ecc_aht_b.pdf')


    # ==================================================
    # (c) Belief concentration at t=5
    # ==================================================
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    b = beliefs_hist[5]
    S_hat = np.argsort(b)[-n:]
    i_star = S_hat[np.argmin(b[S_hat])]
    not_in_top_n = np.argsort(b)[:-n]
    j_star = not_in_top_n[np.argmax(b[not_in_top_n])]

    colors = []
    for k in range(K):
        if k == i_star:
            colors.append('gold')
        elif k == j_star:
            colors.append('orange')
        elif k in S_true:
            colors.append('crimson')
        else:
            colors.append('steelblue')

    ax.bar(range(K), b, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(K))
    ax.set_xlabel('Stream $k$')
    ax.set_ylabel('$p_5(k)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('Champion\n$i^*$', xy=(i_star, b[i_star]), 
                xytext=(i_star, b[i_star] + 0.15),
                ha='center', fontsize=9, color='darkgoldenrod',
                arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))
    ax.annotate('Challenger\n$j^*$', xy=(j_star, b[j_star]), 
                xytext=(j_star, b[j_star] + 0.15),
                ha='center', fontsize=9, color='darkorange',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    save_panel(fig, 'inside_ecc_aht_c.pdf')


    # ==================================================
    # (d) Separation at t=20
    # ==================================================
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    b = beliefs_hist[20]
    colors = ['crimson' if k in S_true else 'steelblue' for k in range(K)]
    ax.bar(range(K), b, color=colors, alpha=0.8)

    ax.axhline(0.8, ls='--', lw=1, color='green')
    ax.axhline(0.2, ls='--', lw=1, color='red')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(K))
    ax.set_xlabel('Stream $k$')
    ax.set_ylabel('$p_{20}(k)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_panel(fig, 'inside_ecc_aht_d.pdf')


    # ==================================================
    # (e) Action heatmap
    # ==================================================
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    abs_max = np.nanmax(np.abs(actions_hist))

    norm = TwoSlopeNorm(
        vmin=-abs_max,
        vcenter=0.0,
        vmax=abs_max
    )

    im = ax.imshow(
        actions_hist,
        aspect='auto',
        cmap=crameri.vik,
        norm=norm
    )

    ax.set_xlabel('Stream $k$')
    ax.set_ylabel('Time $t$')
    ax.set_yticks(range(0, T_steps, 2))

    for k in S_true:
        ax.axvline(k, color='black', ls='--', lw=1, alpha=0.8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$c_t[k]$', rotation=270, labelpad=10)

    save_panel(fig, 'inside_ecc_aht_e.pdf')
    
    
    # ==================================================
    # (f) Correlation exploitation
    # ==================================================
    fig, ax1 = plt.subplots(figsize=PANEL_FIGSIZE)

    t_key = 5
    c = actions_hist[t_key]
    
    # Find the stream with the largest absolute weight (the Champion), and display the correlations centered around it.
    k_focus = np.argmax(np.abs(c))
    
    # 1. Plot the correlation Sigma (right axis, as background)
    ax2 = ax1.twinx()
    sigma_curve = Sigma[k_focus]
    # Use a filled chart to emphasize that this represents "environmental structure"
    ax2.fill_between(range(K), sigma_curve, color='orange', alpha=0.15, label=r'Correlation $\Sigma[i^*, :]$')
    ax2.plot(range(K), sigma_curve, color='orange', linestyle='--', linewidth=1, alpha=0.6)
    
    # Set the range of the right axis to ensure the main part is within view.
    ax2.set_ylabel(r'Correlation $\rho$', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(-0.2, 1.1)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('darkorange')
    ax2.spines['top'].set_visible(False)

    # 2. Plot the action vector c_t (left axis, as the foreground)
    line_c, = ax1.plot(c, 'o-', color='#1f77b4', lw=2, markersize=5, label=r'Action Weights $c_t$')
    
    # Mark the neutral wire for easy identification of positive and negative terminals.
    ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    
    ax1.set_xlabel('Stream $k$')
    ax1.set_ylabel(r'Weight Value', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    # Slightly enlarge the range to avoid being too close to the edge.
    y_abs_max = np.max(np.abs(c)) * 1.2
    ax1.set_ylim(-y_abs_max, y_abs_max)
    ax1.spines['top'].set_visible(False)

    # 3. Add a legend (combining legends from both axes)
    # Create a dummy patch for displaying the fill color in the legend.
    patch_sigma = mpatches.Patch(color='orange', alpha=0.3, label=r'Correlation w.r.t $i^*$')
    ax1.legend(handles=[line_c, patch_sigma], loc='upper left', frameon=False, fontsize=8)

    save_panel(fig, 'inside_ecc_aht_f.pdf')

if __name__ == "__main__":
    visualize_ecc_aht_panels()