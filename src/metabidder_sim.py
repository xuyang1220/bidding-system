import numpy as np
from dataclasses import dataclass
from typing import Dict, List


# =========================
# 1) Landscape primitives
# =========================

from math import sqrt, pi as PI

def normal_cdf(z):
    a = 0.147
    x = z / sqrt(2)
    sign = np.sign(x)
    xx = x * x
    erf = sign * np.sqrt(1 - np.exp(-xx * (4 / PI + a * xx) / (1 + a * xx)))
    return 0.5 * (1 + erf)

def pwin_lognormal(b_cpm, mu, sigma):
    t = np.log(np.clip(b_cpm, 1e-12, None))
    z = (t - mu) / sigma
    return np.clip(normal_cdf(z), 1e-12, 1 - 1e-12)

def epay_cpm_lognormal(b_cpm, mu, sigma):
    """
    E[pay_cpm per impression] = E[M * 1(M <= b)] for lognormal M.
    """
    t = np.log(np.clip(b_cpm, 1e-12, None))
    z2 = (t - mu - sigma**2) / sigma
    return np.exp(mu + 0.5 * sigma**2) * np.clip(normal_cdf(z2), 1e-12, 1 - 1e-12)


# =========================
# 2) Channel batch format
# =========================

@dataclass
class ChannelBatch:
    name: str
    seg_id: np.ndarray           # shape (n,)
    pconv_hat: np.ndarray        # shape (n,)
    market_price_cpm: np.ndarray # shape (n,) for realized simulation
    mu_by_seg: np.ndarray        # shape (S,)
    sigma_by_seg: np.ndarray     # shape (S,)
    value_per_conv: float = 1.0  # optional channel weight/value
    bid_cpm_min: float = 0.01
    bid_cpm_max: float = 50.0


# =========================
# 3) Per-channel expected spend
# =========================

def expected_spend_channel_cpm(
    batch: ChannelBatch,
    lam: float,
    use_value_weight: bool = True,
) -> float:
    """
    Returns total expected spend in CPM-sum units for this channel batch.
    Convert to dollars by dividing by 1000.
    """
    seg = batch.seg_id
    mu = batch.mu_by_seg[seg]
    sig = batch.sigma_by_seg[seg]

    if use_value_weight:
        base_value_cpm = 1000.0 * batch.pconv_hat * batch.value_per_conv
    else:
        base_value_cpm = 1000.0 * batch.pconv_hat

    bid_cpm = base_value_cpm / lam
    bid_cpm = np.clip(bid_cpm, batch.bid_cpm_min, batch.bid_cpm_max)

    pay_exp_cpm = epay_cpm_lognormal(bid_cpm, mu, sig)
    return float(pay_exp_cpm.sum())


# =========================
# 4) Shared-budget lambda solver
# =========================

def solve_shared_lambda(
    channel_batches: List[ChannelBatch],
    budget_step_dollars: float,
    use_value_weight: bool = True,
    lam_lo: float = 1e-6,
    lam_hi: float = 1e6,
    iters: int = 60,
) -> float:
    """
    Solve one global lambda shared across all channels:
        sum_c sum_i E[pay_cpm_i(lambda)] = 1000 * budget_step_dollars
    """
    target_total_cpm = 1000.0 * budget_step_dollars

    def total_expected_cpm(lam: float) -> float:
        total = 0.0
        for batch in channel_batches:
            total += expected_spend_channel_cpm(
                batch=batch,
                lam=lam,
                use_value_weight=use_value_weight,
            )
        return total

    lo, hi = lam_lo, lam_hi
    s_lo = total_expected_cpm(lo)
    s_hi = total_expected_cpm(hi)

    # expand bracket if needed
    for _ in range(40):
        if s_lo < target_total_cpm:
            lo /= 10.0
            s_lo = total_expected_cpm(lo)
        elif s_hi > target_total_cpm:
            hi *= 10.0
            s_hi = total_expected_cpm(hi)
        else:
            break

    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        s_mid = total_expected_cpm(mid)
        if s_mid > target_total_cpm:
            lo = mid
        else:
            hi = mid

    return float(np.sqrt(lo * hi))


# =========================
# 5) Build bids for each channel
# =========================

def compute_channel_bids(
    batch: ChannelBatch,
    lam: float,
    use_value_weight: bool = True,
) -> np.ndarray:
    if use_value_weight:
        base_value_cpm = 1000.0 * batch.pconv_hat * batch.value_per_conv
    else:
        base_value_cpm = 1000.0 * batch.pconv_hat

    bid_cpm = base_value_cpm / lam
    bid_cpm = np.clip(bid_cpm, batch.bid_cpm_min, batch.bid_cpm_max)
    return bid_cpm


# =========================
# 6) Realized auction simulation
# =========================

def simulate_channel_realized(batch: ChannelBatch, bid_cpm: np.ndarray) -> Dict[str, float]:
    win = bid_cpm >= batch.market_price_cpm
    pay_real_cpm = batch.market_price_cpm * win

    return {
        "wins": int(win.sum()),
        "spend_real_dollars": float(pay_real_cpm.sum() / 1000.0),
        "win_rate": float(win.mean()),
    }


# =========================
# 7) One step of multi-channel allocation
# =========================

def run_multichannel_budget_step(
    channel_batches: List[ChannelBatch],
    budget_step_dollars: float,
    use_value_weight: bool = True,
) -> Dict:
    """
    Solve one shared lambda, then simulate all channels.
    """
    lam = solve_shared_lambda(
        channel_batches=channel_batches,
        budget_step_dollars=budget_step_dollars,
        use_value_weight=use_value_weight,
    )

    results = {
        "lambda_shared": lam,
        "channels": {},
        "expected_spend_total_dollars": 0.0,
        "realized_spend_total_dollars": 0.0,
    }

    for batch in channel_batches:
        bid_cpm = compute_channel_bids(
            batch=batch,
            lam=lam,
            use_value_weight=use_value_weight,
        )

        seg = batch.seg_id
        mu = batch.mu_by_seg[seg]
        sig = batch.sigma_by_seg[seg]
        pay_exp_cpm = epay_cpm_lognormal(bid_cpm, mu, sig)
        spend_exp_dollars = float(pay_exp_cpm.sum() / 1000.0)

        realized = simulate_channel_realized(batch, bid_cpm)

        results["channels"][batch.name] = {
            "n_imps": int(len(batch.seg_id)),
            "avg_bid_cpm": float(np.mean(bid_cpm)),
            "expected_spend_dollars": spend_exp_dollars,
            "realized_spend_dollars": realized["spend_real_dollars"],
            "wins": realized["wins"],
            "win_rate": realized["win_rate"],
        }

        results["expected_spend_total_dollars"] += spend_exp_dollars
        results["realized_spend_total_dollars"] += realized["spend_real_dollars"]

    return results


# =========================
# 8) Example synthetic channel data
# =========================

def make_dummy_channel_batch(
    name: str,
    n: int,
    S: int,
    seed: int,
    value_per_conv: float,
    mu_shift: float,
    sigma_level: float,
) -> ChannelBatch:
    """
    Simple synthetic channel generator.
    Feed / video / email can have different landscapes and pconv levels.
    """
    rng = np.random.default_rng(seed)

    seg_id = rng.integers(0, S, size=n)

    # channel-specific pconv scale
    if name == "feed":
        pconv_hat = np.clip(rng.lognormal(mean=-7.2, sigma=0.5, size=n), 1e-8, 0.02)
    elif name == "video":
        pconv_hat = np.clip(rng.lognormal(mean=-7.6, sigma=0.6, size=n), 1e-8, 0.02)
    else:  # email
        pconv_hat = np.clip(rng.lognormal(mean=-6.8, sigma=0.4, size=n), 1e-8, 0.02)

    # per-segment landscape params
    base = np.linspace(0.0, -0.5, S)
    mu_by_seg = np.log(1.5) + mu_shift + base
    sigma_by_seg = np.full(S, sigma_level)

    # realized market price from the same per-segment lognormal
    mu = mu_by_seg[seg_id]
    sig = sigma_by_seg[seg_id]
    market_price_cpm = np.exp(mu + sig * rng.normal(size=n))

    return ChannelBatch(
        name=name,
        seg_id=seg_id,
        pconv_hat=pconv_hat,
        market_price_cpm=market_price_cpm,
        mu_by_seg=mu_by_seg,
        sigma_by_seg=sigma_by_seg,
        value_per_conv=value_per_conv,
        bid_cpm_min=0.01,
        bid_cpm_max=50.0,
    )


# =========================
# 9) Example usage
# =========================

if __name__ == "__main__":
    S = 20

    feed_batch = make_dummy_channel_batch(
        name="feed",
        n=50_000,
        S=S,
        seed=1,
        value_per_conv=50.0,
        mu_shift=0.0,
        sigma_level=0.7,
    )

    video_batch = make_dummy_channel_batch(
        name="video",
        n=30_000,
        S=S,
        seed=2,
        value_per_conv=80.0,
        mu_shift=0.4,   # more expensive channel
        sigma_level=0.8,
    )

    email_batch = make_dummy_channel_batch(
        name="email",
        n=20_000,
        S=S,
        seed=3,
        value_per_conv=30.0,
        mu_shift=-0.5,  # cheaper channel
        sigma_level=0.6,
    )

    channel_batches = [feed_batch, video_batch, email_batch]

    out = run_multichannel_budget_step(
        channel_batches=channel_batches,
        budget_step_dollars=500.0,
        use_value_weight=True,
    )

    print("shared lambda:", out["lambda_shared"])
    print("expected total spend:", out["expected_spend_total_dollars"])
    print("realized total spend:", out["realized_spend_total_dollars"])
    for ch, metrics in out["channels"].items():
        print(ch, metrics)