# Cross-Channel Ads Bidding System Design (tCPA & MaxConvs)

## Overview

We design a hierarchical bidding system for a multi-channel ads product (Video, Feeds, Email) supporting both **tCPA** and **Max Conversions (MaxConvs)** strategies.

The system separates concerns across three layers:

1. **Budget Server** → fast pacing & spend safety  
2. **Metabidder** → cross-channel allocation & global optimization  
3. **Channel Bidders** → local auction execution  

This decomposition ensures stability, scalability, and channel-specific optimization.

---

## Architecture
            +----------------------+
            |    Budget Server     |
            |----------------------|
            | Spend tracking       |
            | Pacing curve         |
            | Throttling control   |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |     Metabidder       |
            |----------------------|
            | Global shadow tCPA   |
            | Channel allocation   |
            | Response modeling    |
            +----------+-----------+
                       |
    -----------------------------------------
    |                  |                    |
    v                  v                    v
    +---------------+ +---------------+ +---------------+
    | Video Bidder | | Feeds Bidder | | Email Bidder |
    |---------------| |---------------| |---------------|
    | Local HDMI | | Local HDMI | | Local HDMI |
    | Calibration | | Calibration | | Calibration |
    | Delay model | | Delay model | | Delay model |
    +---------------+ +---------------+ +---------------+


---

## Layer Responsibilities

### 1. Budget Server (Fast Control Loop)

**Goal:** Enforce budget pacing and prevent overspend.

**Inputs:**
- Spend so far (global + per channel)
- Target spend curve (daily pacing)
- Traffic volume

**Outputs:**
- Global throttle probability
- Optional channel-level throttle caps

**Key Properties:**
- Runs at high frequency (seconds/minutes)
- Direct actuator on spend
- Does NOT optimize conversion efficiency

---

### 2. Metabidder (Cross-Channel Allocator)

**Goal:** Allocate budget efficiently across channels.

**Inputs:**
- Remaining budget & time
- Channel-level response curves:
  - `S_c(λ)` = expected spend
  - `C_c(λ)` = expected *matured* conversions
- Calibration + delay-adjusted signals

**Outputs:**
- Global shadow tCPA (or λ)
- Optional channel allocation weights

**Update Frequency:**
- Every ~6 hours (or when sufficient signal matures)

**Key Idea:**
Solve for λ such that:

- Total spend matches budget
- Marginal conversions per dollar are equalized across channels

---

### 3. Channel Bidders (Local Execution)

**Goal:** Execute optimal bidding in each channel.

**Inputs:**
- Global shadow tCPA
- Channel allocation weight
- Local bid landscape (HDMI-style)
- Local models:
  - pCTR, pCVR
  - calibration factor
  - delay/maturity model

**Outputs:**
- Impression-level bid or multiplier

**Key Mechanism:**
- Segment-based multiplier:  
  `k(adgroup_id, segment_id)`
- Converts global target into local bid:
bid ∝ pCTR × pCVR_adj × value(shadow tCPA)


---

## Strategy Support

### tCPA Mode
- External target CPA provided
- Metabidder stabilizes toward target
- Budget server enforces spend curve

### MaxConvs Mode
- No explicit CPA target
- Metabidder solves for **internal shadow tCPA**
- Same downstream pipeline as tCPA

---

## Key Design Principles

### 1. Separation of Control Timescales

| Layer           | Timescale       | Responsibility              |
|----------------|----------------|-----------------------------|
| Budget Server  | Seconds/minutes| Spend safety                |
| Metabidder     | Hours          | Allocation efficiency       |
| Channel Bidder | Real-time      | Auction execution           |

Avoid multiple layers reacting to the same signal simultaneously.

---

### 2. Throttling vs Bid Control

- **Throttling** → fast, direct spend control  
- **Bid adjustment** → slower, efficiency optimization  

Rule:
> Use throttling for pacing, bids for value.

---

### 3. Channel-Decoupled Response Modeling

Instead of a single global bid landscape:

Model per-channel response:
- `S_video(λ), C_video(λ)`
- `S_feeds(λ), C_feeds(λ)`
- `S_email(λ), C_email(λ)`

This enables proper cross-channel substitution.

---

### 4. Delay-Aware Optimization

Metabidder uses **matured conversions**, not raw conversions.

Each channel estimates:
matured_value = pCVR × calibration × maturity_factor


Avoid bias toward short-delay channels.

---

### 5. Calibration vs Delay Separation

Avoid overloading `pCVR^α`.

Instead:
- Calibration correction (bias)
- Delay/maturity model (time-dependent)

Improves interpretability and stability.

---

## Potential Failure Modes

### 1. Double Control Instability
- Throttling and bid updates both reacting to spend error
- Leads to oscillation

**Mitigation:**
- Separate timescales
- Smooth updates (EMA, caps)

---

### 2. Channel Imbalance
- Single shadow tCPA may underinvest in delayed channels

**Mitigation:**
- Delay correction
- Channel-specific translation layer
- Minimum exploration

---

### 3. Noisy / Sparse Channels
- Email or low-volume segments unstable

**Mitigation:**
- Bayesian shrinkage
- Segment merging
- Exploration quotas

---

## Future Improvements

### 1. Replace PID with Optimization / MPC
- Explicitly solve budget allocation over time horizon

### 2. Learn Marginal ROI Curves
- Estimate Δconversions / Δspend per channel

### 3. Adaptive Segmentation
- Dynamic granularity for HDMI segments

### 4. Counterfactual Evaluation
- Evaluate policy changes offline using logged data

---

## Summary

This system implements a **hierarchical control architecture**:

- **Budget Server** → ensures spend correctness  
- **Metabidder** → ensures allocation efficiency  
- **Channel Bidders** → ensure execution quality  

Core principle:

> Global consistency + local specialization + multi-timescale control

This design supports both **tCPA** and **MaxConvs** with a unified framework and scales naturally to additional channels.

