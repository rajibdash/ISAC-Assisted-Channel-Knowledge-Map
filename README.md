# ISAC Assisted Channel Knowledge Map (CKM)

**Author**: Rajib Kumar Dash
**Disclaimer**: These informations below are draft understanding and need to be refined or filtered when full undestanding of CKM need will be unfolded. 

# Construction of Channel Knowledge Maps (CKM) for Physical Layer Authentication (PLA)

The construction of a **Channel Knowledge Map (CKM)** for **Physical Layer Authentication (PLA)** bridges environment-aware wireless communications and location-based security. By creating a deterministic, location-dependent database of channel characteristics—such as Channel State Information (CSI), path loss (PL), and Angle of Arrival (AoA)—systems can authenticate a transmitter's location to detect spoofing or impersonation attacks.

---

## Core Construction Methodologies

### 1. ISAC-Assisted Environment Reconstruction
Recent frameworks leverage **Integrated Sensing and Communication (ISAC)** to dynamically construct maps without requiring manual site surveys.
* **Layout Capture:** Multiple receivers utilize standard communication signals to sense the layout, geometry, and coordinates of surrounding physical objects and obstacles.
* **Spatial Modeling:** Algorithms (such as Poisson cluster processes with spatial tapering) map the localized environment without the need for dedicated or specialized sensing hardware.

### 2. Ray Tracing-Based Channel Synthesis
Once the structural environment layout is digitally reconstructed, a deterministic ray-tracing engine is deployed.
* **Simulating Positions:** The ray-tracer simulates transmitter signals originating from a massive grid of potential coordinate points across the entire indoor or outdoor coverage area.
* **Parameter Extraction:** For every simulated location, the engine calculates location-dependent channel parameters—such as the dominant channel tap path loss, multipath profiles, and AoA. These values are saved into the CKM database mapped precisely to physical coordinates.

### 3. Dynamic and Generative CKM Refinement
Because real-world wireless environments are rarely static, advanced CKM construction uses hybrid inference models to maintain data accuracy:
* **Bayesian Inference Frameworks:** Systems split channel parameters into quasi-static profiles (walls, large furniture) using historical measurements, and dynamic components (moving humans, dynamic scatterers) updated through limited real-time observations.
* **Generative AI and Diffusion Models:** Missing or partially observed spatial data blocks are filled using generative networks, ensuring a continuous and high-resolution channel reference map across unmapped coordinates.

---

## How the CKM is Used in PLA

```
[Transmitter] ---> Sends Signal ---> [Base Station/Receivers]
                                             |
                                    1. Estimates Channel
                                    2. Fetches Coarse Location
                                             |
                                             v
                                  [ PLA Verification Engine ]
                                             ^
                                             | Matches Coordinates
                                             v
                                  [ Channel Knowledge Map ]
```

When a device attempts to authenticate, the network leverages a cross-layer approach:
1. **Coarse Localization:** The system obtains an approximate position of the legitimate transmitter from upper network layers or its previous known space-time trajectory.
2. **Database Querying:** The receiver queries the CKM for the expected channel signature at those exact coordinates.
3. **Hypothesis Testing:** The actual estimated channel from the received signal is compared against the CKM entry. If an adversary (e.g., Trudy) attempts an impersonation attack from a different physical location, the multipath profile will significantly deviate from the CKM data, triggering a spoofing alert.

---

## Simulation Framework (Python Blueprint)

Below is a structured Python workflow utilizing a deterministic ray-tracing approach (Line-of-Sight + Single-bounce multipath) to build and query a multi-feature map storing RSSI values, dominant Angle of Arrival (AoA), and complex CSI coefficients.

### 1. CKM Multi-Feature Mapping Pipeline

```python
"""
Title: Deterministic CKM Synthesis for Physical Layer Authentication
Features Tracked: RSSI (dBm), AoA (rad), Simplified CSI Matrices (Complex H)
"""

import numpy as np
import pandas as pd

# 1. System Parameters & Geometric Setup
ROOM_DIM = 10.0      # 10m x 10m Room boundaries
FC = 5.8e9           # Carrier Frequency (5.8 GHz Wi-Fi / Sub-6G)
C = 3e8              # Speed of Light (m/s)
LAMBDA = C / FC      # Signal Wavelength
TX_POWER = 20.0      # Transmit Power in dBm

# Fixed Base Station (Receiver Anchor)
BS_POS = np.array([5.0, 5.0])

# Static Physical Scatterers (Concrete pillars/structural elements)
SCATTERERS = np.array([
    [2.0, 3.0], [8.0, 7.0], [3.0, 8.0]
])

# 2. Discrete Grid Discretization (0.5 meter steps)
GRID_RESOLUTION = 0.5
x_coords = np.arange(0.5, ROOM_DIM, GRID_RESOLUTION)
y_coords = np.arange(0.5, ROOM_DIM, GRID_RESOLUTION)

ckm_records = []

# 3. Ray-Tracing Execution Engine
for x in x_coords:
    for y in y_coords:
        tx_pos = np.array([x, y])
        
        # Skip grid coordinate overlapping with the Base Station itself
        if np.linalg.norm(tx_pos - BS_POS) < 0.1:
            continue
            
        # --- Path 1: Line of Sight (LoS) Ray ---
        d_los = np.linalg.norm(BS_POS - tx_pos)
        phase_los = (2 * np.pi * d_los) / LAMBDA
        pl_los = 20 * np.log10(4 * np.pi * d_los / LAMBDA) # Friis Path Loss
        amp_los = 10**(-pl_los / 20)
        h_los = amp_los * np.exp(-1j * (phase_los % (2 * np.pi)))
        
        # Angle of Arrival (AoA) vector extraction
        vec_los = tx_pos - BS_POS
        aoa = np.arctan2(vec_los[1], vec_los[0])
        
        # --- Paths 2+: Non-Line of Sight (NLoS) Reflection Rays ---
        h_nlos = 0 + 0j
        refl_coeff = 0.3 # Reflection loss factor (30% power preservation)
        
        for scat in SCATTERERS:
            d_scat = np.linalg.norm(scat - tx_pos) + np.linalg.norm(BS_POS - scat)
            phase_nlos = (2 * np.pi * d_scat) / LAMBDA
            pl_nlos = 20 * np.log10(4 * np.pi * d_scat / LAMBDA)
            amp_nlos = (10**(-pl_nlos / 20)) * refl_coeff
            h_nlos += amp_nlos * np.exp(-1j * (phase_nlos % (2 * np.pi)))

        # 4. Multidimensional Attribute Synthesis
        h_total = h_los + h_nlos
        rssi = TX_POWER - (20 * np.log10(np.abs(1 / h_total)))
        
        ckm_records.append({
            'tx_x': tx_pos[0],
            'tx_y': tx_pos[1],
            'rssi_dbm': rssi,
            'aoa_rad': aoa,
            'csi_real': np.real(h_total),
            'csi_imag': np.imag(h_total)
        })

# Store Generated Channel Knowledge Map
df_ckm = pd.DataFrame(ckm_records)
```

### 2. Implementing the PLA Hypothesis Tester

When a device requests access, it transmits its claimed coordinate location $(x_{claim}, y_{claim})$. The verification engine pulls the baseline data from the CKM and applies a multi-feature test metric:

```python
def authenticate_user(claimed_x, claimed_y, observed_rssi, observed_aoa, threshold=3.5):
    """
    Validates physical origin claims using localized CKM parameters
    """
    # Fetch nearest map grid point matching the coordinate claim
    map_entry = df_ckm.loc[
        (df_ckm['tx_x'] == claimed_x) & (df_ckm['tx_y'] == claimed_y)
    ]
    
    if map_entry.empty:
        return "Rejected: Location outside CKM bounds"
        
    expected_rssi = map_entry['rssi_dbm'].values[0]
    expected_aoa = map_entry['aoa_rad'].values[0]
    
    # Calculate Mahalanobis-like Euclidean variance across spatial fingerprints
    rssi_delta = abs(observed_rssi - expected_rssi)
    aoa_delta = abs(np.angle(np.exp(1j * (observed_aoa - expected_aoa)))) # Phase wrap safety
    
    anomaly_score = (rssi_delta * 0.5) + (aoa_delta * 5.0)
    
    if anomaly_score <= threshold:
        return f"Authenticated (Score: {anomaly_score:.2f})"
    else:
        return f"Spoofing Detected! Intruder Warning (Score: {anomaly_score:.2f})"
```

