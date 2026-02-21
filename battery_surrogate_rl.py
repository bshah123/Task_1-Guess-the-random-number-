"""
=============================================================================
OPTION 2: Pre-solve & Build Fast Surrogate Model
=============================================================================
Pipeline:
  1. Define a physics-based battery charging environment
  2. Pre-solve optimal charging trajectories (CC-CV + SOH-aware solver)
  3. Collect (state, optimal_action) dataset
  4. Train a fast Neural Network surrogate via Supervised Learning (SL)
  5. Optional: Fine-tune with RL (PPO) using the SL model as a warm start
  6. Visualize results
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Battery Physics Model
# ─────────────────────────────────────────────────────────────

class BatteryParams:
    """Lithium-ion battery parameters (typical 18650-style cell)."""
    C_nom    = 2.5      # Nominal capacity [Ah]
    V_max    = 4.2      # Max voltage [V]
    V_min    = 2.8      # Min voltage [V]
    V_cc_cv  = 4.1      # CC-CV transition voltage [V]
    I_max    = 2.5      # Max current [A]  (1C)
    I_min    = 0.0      # Min current [A]
    I_cut    = 0.05     # Cutoff current [A]
    R0       = 0.05     # Internal resistance [Ohm]
    R1       = 0.02     # RC-pair resistance [Ohm]
    C1       = 2000.0   # RC-pair capacitance [F]
    T_amb    = 25.0     # Ambient temperature [°C]
    T_max    = 45.0     # Max cell temperature [°C]
    alpha_T  = 0.5      # Thermal resistance [°C/W]
    dt       = 10.0     # Time step [s]


def ocv_from_soc(soc: float) -> float:
    """Open-circuit voltage as a function of SOC (polynomial fit)."""
    soc = np.clip(soc, 0.0, 1.0)
    return (3.0 + 1.2 * soc
            - 0.4 * soc**2
            + 0.3 * np.sin(np.pi * soc))


def capacity_from_soh(soh: float, p: BatteryParams) -> float:
    """Effective capacity degrades linearly with SOH."""
    return p.C_nom * soh


# ─────────────────────────────────────────────────────────────
# SECTION 2: Battery Charging Environment
# ─────────────────────────────────────────────────────────────

class BatteryChargingEnv:
    """
    State: [SOC, V_terminal, T_cell, I_prev, SOH, V_RC]
    Action: current I in [I_min, I_max]
    """

    def __init__(self, p: BatteryParams = None, init_soh: float = 1.0):
        self.p = p or BatteryParams()
        self.init_soh = init_soh
        self.reset()

    def reset(self, soc_init: float = 0.1, soh: float = None):
        p = self.p
        self.soh    = soh if soh is not None else self.init_soh
        self.soc    = soc_init
        self.V_RC   = 0.0          # RC capacitor voltage
        self.T_cell = p.T_amb
        self.I_prev = 0.0
        self.C_eff  = capacity_from_soh(self.soh, p)
        self.done   = False
        self.time   = 0.0
        return self._get_state()

    def _get_state(self):
        p = self.p
        V_oc = ocv_from_soc(self.soc)
        V_t  = V_oc + self.I_prev * p.R0 + self.V_RC
        return np.array([
            self.soc,
            V_t / p.V_max,           # normalized
            (self.T_cell - p.T_amb) / (p.T_max - p.T_amb),  # normalized
            self.I_prev / p.I_max,   # normalized
            self.soh,
            self.V_RC / p.V_max,     # normalized
        ], dtype=np.float32)

    def step(self, I: float):
        p = self.p
        I = np.clip(I, p.I_min, p.I_max)

        # RC circuit dynamics
        tau      = p.R1 * p.C1
        self.V_RC = self.V_RC * np.exp(-p.dt / tau) + p.R1 * I * (1 - np.exp(-p.dt / tau))

        # Voltage
        V_oc = ocv_from_soc(self.soc)
        V_t  = V_oc + I * p.R0 + self.V_RC

        # SOC update
        dSOC = (I * p.dt) / (3600.0 * self.C_eff)
        self.soc = np.clip(self.soc + dSOC, 0.0, 1.0)

        # Thermal dynamics
        P_heat     = I**2 * (p.R0 + p.R1)
        self.T_cell = p.T_amb + P_heat * p.alpha_T

        # SOH degradation (simplified Arrhenius-like)
        stress      = (I / p.I_max)**2 * (self.T_cell / p.T_max)
        self.soh   -= 1e-6 * stress * p.dt
        self.soh    = max(self.soh, 0.7)
        self.C_eff  = capacity_from_soh(self.soh, p)

        self.I_prev = I
        self.time  += p.dt

        # Termination
        done = (
            self.soc >= 0.99 or
            V_t >= p.V_max or
            I <= p.I_cut or
            self.time >= 7200
        )
        self.done = done

        # Reward (for RL stage)
        reward = self._compute_reward(I, V_t, done)

        return self._get_state(), reward, done, {
            "V_t": V_t, "T": self.T_cell, "SOH": self.soh, "SOC": self.soc
        }

    def _compute_reward(self, I, V_t, done):
        p = self.p
        reward = 0.0

        # Progress: reward charging
        reward += 0.5 * (I / p.I_max)

        # Voltage proximity shaping (exponential penalty near V_max)
        dV = p.V_max - V_t
        if dV < 0.1:
            reward -= 5.0 * np.exp((0.1 - dV) / 0.05)

        # Hard voltage violation
        if V_t > p.V_max:
            reward -= 50.0

        # Temperature penalty
        if self.T_cell > p.T_max:
            reward -= 10.0 * (self.T_cell - p.T_max)

        # SOH preservation
        reward += 0.2 * self.soh

        # CV region: penalize high current when near full charge
        if V_t > p.V_cc_cv:
            reward -= 2.0 * (I / p.I_max)**2

        # Completion bonus
        if done and self.soc >= 0.99:
            reward += 20.0

        return reward


# ─────────────────────────────────────────────────────────────
# SECTION 3: Pre-Solver — Generate Optimal Trajectories
# ─────────────────────────────────────────────────────────────

def cc_cv_optimal_action(state: np.ndarray, p: BatteryParams) -> float:
    """
    SOH-aware CC-CV policy:
    - CC phase: apply ~0.8C but reduce if SOH is degraded
    - CV phase: taper current to keep V_t just below V_max
    """
    soc   = state[0]
    V_t   = state[1] * p.V_max  # de-normalize
    soh   = state[4]
    I_prev= state[3] * p.I_max  # de-normalize

    # Scale max current by SOH
    I_cc = p.I_max * min(soh, 0.9)

    if V_t < p.V_cc_cv:
        # CC phase — ramp up gently
        I_target = I_cc * (0.5 + 0.5 * soc)
    else:
        # CV phase — taper: proportional controller
        V_err   = p.V_max - V_t
        I_target = max(I_prev * (V_err / 0.1) * 0.9, p.I_cut)

    return float(np.clip(I_target, p.I_min, p.I_max))


def collect_dataset(n_trajectories: int = 500,
                    soh_range: tuple = (0.8, 1.0),
                    soc_init_range: tuple = (0.05, 0.3),
                    p: BatteryParams = None) -> tuple:
    """
    Run the CC-CV expert policy across many initial conditions and SOH values.
    Returns (states, actions) arrays.
    """
    p = p or BatteryParams()
    env = BatteryChargingEnv(p)

    all_states  = []
    all_actions = []

    np.random.seed(42)
    for ep in range(n_trajectories):
        soh      = np.random.uniform(*soh_range)
        soc_init = np.random.uniform(*soc_init_range)
        state    = env.reset(soc_init=soc_init, soh=soh)

        for _ in range(720):  # max 2h / 10s = 720 steps
            action = cc_cv_optimal_action(state, p)
            all_states.append(state.copy())
            all_actions.append([action / p.I_max])  # normalize action
            state, _, done, _ = env.step(action)
            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep+1}/{n_trajectories} trajectories "
                  f"({len(all_states)} total samples)")

    X = np.array(all_states,  dtype=np.float32)
    y = np.array(all_actions, dtype=np.float32)
    print(f"\n✅ Dataset: {X.shape[0]} samples, "
          f"{X.shape[1]} state features → 1 action\n")
    return X, y


# ─────────────────────────────────────────────────────────────
# SECTION 4: Surrogate Neural Network (SL Training)
# ─────────────────────────────────────────────────────────────

class SurrogatePolicy(nn.Module):
    """Fast feed-forward network that mimics the optimal CC-CV policy."""

    def __init__(self, state_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()   # output in [0,1] → multiply by I_max
        )

    def forward(self, x):
        return self.net(x)


def train_surrogate(X: np.ndarray, y: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 1e-3) -> tuple:
    """Train surrogate via supervised learning (behaviour cloning)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Split 80/20
    n = len(X)
    idx   = np.random.permutation(n)
    split = int(0.8 * n)
    tr, va = idx[:split], idx[split:]

    X_tr = torch.FloatTensor(X[tr]).to(device)
    y_tr = torch.FloatTensor(y[tr]).to(device)
    X_va = torch.FloatTensor(X[va]).to(device)
    y_va = torch.FloatTensor(y[va]).to(device)

    ds     = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model  = SurrogatePolicy(state_dim=X.shape[1]).to(device)
    optim_ = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=epochs)
    loss_fn= nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optim_.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr)
        train_losses.append(ep_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        val_losses.append(val_loss)
        sched.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"Train MSE: {ep_loss:.5f}  Val MSE: {val_loss:.5f}")

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()
    model.to("cpu")
    print(f"\n✅ Best validation MSE: {best_val:.5f}")
    return model, train_losses, val_losses


# ─────────────────────────────────────────────────────────────
# SECTION 5: Evaluate Surrogate vs. Expert
# ─────────────────────────────────────────────────────────────

def rollout(policy_fn, env: BatteryChargingEnv,
            soc_init: float = 0.1, soh: float = 1.0):
    """Run a policy and return trajectory data."""
    state = env.reset(soc_init=soc_init, soh=soh)
    history = {"soc": [], "V": [], "I": [], "T": [], "SOH": [], "t": []}
    p = env.p
    t = 0.0
    while not env.done:
        I = policy_fn(state)
        history["soc"].append(state[0])
        history["V"].append(state[1] * p.V_max)
        history["I"].append(state[3] * p.I_max)
        history["T"].append(state[2] * (p.T_max - p.T_amb) + p.T_amb)
        history["SOH"].append(state[4])
        history["t"].append(t)
        state, _, done, _ = env.step(I)
        t += p.dt
        if done:
            break
    # append final state
    history["soc"].append(state[0])
    history["V"].append(state[1] * p.V_max)
    history["I"].append(state[3] * p.I_max)
    history["T"].append(state[2] * (p.T_max - p.T_amb) + p.T_amb)
    history["SOH"].append(state[4])
    history["t"].append(t)
    return history


# ─────────────────────────────────────────────────────────────
# SECTION 6: Visualization
# ─────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train MSE", color="#4C72B0", linewidth=2)
    ax.plot(val_losses,   label="Val MSE",   color="#DD8452", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Surrogate Model — Training Curves (Behaviour Cloning)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_trajectories(expert_h, surrogate_h, p: BatteryParams, save_path=None):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    titles = ["SOC", "Terminal Voltage [V]", "Charging Current [A]", "Cell Temperature [°C]"]
    keys   = ["soc", "V", "I", "T"]
    ylims  = [(0, 1.05), (2.7, 4.35), (-0.1, p.I_max + 0.3), (20, 50)]
    hlines = [0.99, p.V_max, None, p.T_max]

    for i, (title, key, ylim, hline) in enumerate(zip(titles, keys, ylims, hlines)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        t_e = np.array(expert_h["t"]) / 60
        t_s = np.array(surrogate_h["t"]) / 60
        ax.plot(t_e, expert_h[key],   "--", color="#4C72B0", label="CC-CV Expert",  linewidth=2)
        ax.plot(t_s, surrogate_h[key], "-", color="#DD8452", label="Surrogate (SL)", linewidth=2)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle=":", alpha=0.6, label=f"Limit ({hline})")
        ax.set_xlabel("Time [min]")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Option 2: Pre-solve → Surrogate Model (SL)\n"
        "Expert CC-CV Policy vs. Learnt Surrogate Neural Network",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_soh_comparison(p_values: list, save_path=None):
    """Show how charging time and final SOH vary across initial SOH levels."""
    env = BatteryChargingEnv()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    soh_levels = np.linspace(0.80, 1.0, 6)
    times_expert    = []
    times_surrogate = []

    print("\nEvaluating across SOH levels...")
    for soh in soh_levels:
        env_e = BatteryChargingEnv(init_soh=soh)
        env_s = BatteryChargingEnv(init_soh=soh)

        _p   = p_values[0]
        _mdl = p_values[1]
        h_e = rollout(lambda s: cc_cv_optimal_action(s, _p), env_e, soh=soh)
        with torch.no_grad():
            h_s = rollout(
                lambda s: float(_mdl(torch.FloatTensor(s).unsqueeze(0)).item()) * _p.I_max,
                env_s, soh=soh)

        times_expert.append(h_e["t"][-1] / 60)
        times_surrogate.append(h_s["t"][-1] / 60)

    axes[0].plot(soh_levels, times_expert,    "o--", color="#4C72B0", label="Expert CC-CV",  linewidth=2)
    axes[0].plot(soh_levels, times_surrogate, "s-",  color="#DD8452", label="Surrogate (SL)", linewidth=2)
    axes[0].set_xlabel("Initial SOH")
    axes[0].set_ylabel("Charging Time [min]")
    axes[0].set_title("Charging Time vs. Cell Health")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Inference speed comparison
    import time
    state = env.reset()
    n_calls = 10000

    t0 = time.perf_counter()
    for _ in range(n_calls):
        cc_cv_optimal_action(state, p_values[0])
    expert_ms = (time.perf_counter() - t0) / n_calls * 1000

    st = torch.FloatTensor(state).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_calls):
            p_values[1](st)
    surrogate_ms = (time.perf_counter() - t0) / n_calls * 1000

    methods  = ["Expert\n(scipy/loop)", "Surrogate NN\n(PyTorch)"]
    times_ms = [expert_ms,              surrogate_ms]
    bars = axes[1].bar(methods, times_ms, color=["#4C72B0", "#DD8452"], width=0.4)
    axes[1].set_ylabel("Inference Time [ms/step]")
    axes[1].set_title("Inference Speed Comparison")
    for bar, val in zip(bars, times_ms):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{val:.4f}ms", ha="center", va="bottom", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Option 2 — Surrogate Model Performance Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# SECTION 7: Main Pipeline
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  OPTION 2: Pre-solve & Build Fast Surrogate Model")
    print("  Battery Charging Control — SL Surrogate + Optional RL")
    print("=" * 65)

    p = BatteryParams()

    # ── STEP 1: Collect expert dataset ──────────────────────────
    print("\n[STEP 1] Pre-solving optimal CC-CV trajectories...")
    X, y = collect_dataset(n_trajectories=400, p=p)

    # ── STEP 2: Train surrogate model ───────────────────────────
    print("\n[STEP 2] Training surrogate neural network (Behaviour Cloning)...")
    model, train_losses, val_losses = train_surrogate(
        X, y, epochs=80, batch_size=256, lr=1e-3
    )

    # ── STEP 3: Plot training curves ────────────────────────────
    print("\n[STEP 3] Plotting training curves...")
    plot_training_curves(train_losses, val_losses,
                         save_path="battery_training_curves.png")

    # ── STEP 4: Compare expert vs. surrogate trajectories ───────
    print("\n[STEP 4] Rolling out policies and comparing trajectories...")
    env_expert    = BatteryChargingEnv(p, init_soh=0.92)
    env_surrogate = BatteryChargingEnv(p, init_soh=0.92)

    # Expert policy function
    def expert_fn(state):
        return cc_cv_optimal_action(state, p)

    # Surrogate policy function
    @torch.no_grad()
    def surrogate_fn(state):
        st = torch.FloatTensor(state).unsqueeze(0)
        return float(model(st).item()) * p.I_max

    h_expert    = rollout(expert_fn,    env_expert,    soh=0.92)
    h_surrogate = rollout(surrogate_fn, env_surrogate, soh=0.92)

    print(f"\n  Expert    → Final SOC: {h_expert['soc'][-1]:.3f}  "
          f"Time: {h_expert['t'][-1]/60:.1f} min  "
          f"Final SOH: {h_expert['SOH'][-1]:.4f}")
    print(f"  Surrogate → Final SOC: {h_surrogate['soc'][-1]:.3f}  "
          f"Time: {h_surrogate['t'][-1]/60:.1f} min  "
          f"Final SOH: {h_surrogate['SOH'][-1]:.4f}")

    plot_trajectories(h_expert, h_surrogate, p,
                      save_path="battery_trajectories.png")

    # ── STEP 5: SOH sensitivity analysis ───────────────────────
    print("\n[STEP 5] SOH sensitivity & inference speed analysis...")
    plot_soh_comparison([p, model], save_path="battery_soh_analysis.png")

    # ── STEP 6: Save model ──────────────────────────────────────
    torch.save(model.state_dict(), "surrogate_policy.pth")
    print("\n✅ Surrogate model saved to: surrogate_policy.pth")

    # ── SUMMARY ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Dataset size     : {X.shape[0]:,} samples")
    print(f"  Model params     : {sum(p_.numel() for p_ in model.parameters()):,}")
    print(f"  Best Val MSE     : {min(val_losses):.5f}")
    print(f"  Expert SOC final : {h_expert['soc'][-1]:.3f}")
    print(f"  Surrogate SOC    : {h_surrogate['soc'][-1]:.3f}")
    print()
    print("  Outputs:")
    print("    battery_training_curves.png")
    print("    battery_trajectories.png")
    print("    battery_soh_analysis.png")
    print("    surrogate_policy.pth")
    print("=" * 65)


if __name__ == "__main__":
    main()
