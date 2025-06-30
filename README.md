# Elevator Standby Scheduling Strategy

This repo implements the paper:  
**_Optimization of Elevator Standby Scheduling Strategy in Smart Buildings_**

## Overview

Simulate and compare elevator dispatching strategies, focusing on standby scheduling to improve wait times.

---

## Requirements

```bash
pip install simpy numpy pandas matplotlib
```

---

## Usage

**1. Generate Data**  
```bash
python generate_passengers.py
```
Creates simulated passenger data. See `arrival_rate.png` for the arrival rate.

**2. Run Standby Strategy**  
```bash
python standby_strategy.py
```
Closest car algorithm with standby scheduling.

**3. Run Baselines**  
```bash
python closest_car_algorithm.py
python THV_algorithm.py
python fuzzy_logic.py
```

---

## Citation

If used, please cite:  
_Optimization of Elevator Standby Scheduling Strategy in Smart Buildings_
