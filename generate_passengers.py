# generate_passengers.py

import simpy
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

class Passenger:
    def __init__(self, id, origin, destination, arrival_time):
        self.id = id
        self.origin = origin
        self.destination = destination
        self.arrival_time = arrival_time

def get_dynamic_arrival_rate(current_time):
    import math
    t = current_time % 1440
    def gaussian(x, mu, sigma, amp):
        return amp * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    rate = 0.3 * (
        gaussian(t, 8.8 * 60, 70, 3) +
        gaussian(t, 12.5 * 60, 45, 4) +
        gaussian(t, 18 * 60, 80, 4) +
        0.15
    )
    return max(rate, 0.05)

def directional_bias(env, num_floors):
    hour = int(env.now // 60) % 24
    r = random.random()
    if 5<= hour < 6:
        if r < 0.5:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0
    elif 6 <= hour < 8:
        if r < 0.6:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0
    elif 8 <= hour < 10:
        if r < 0.8:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.9:
            return random.randint(1, num_floors - 1), 0
    elif 10 <= hour < 11:
        if r < 0.6:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0
    elif 11 <= hour < 12:
        if r < 0.4:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.7:
            return random.randint(1, num_floors - 1), 0
    elif 12 <= hour < 13:
        if r < 0.2:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.7:
            return random.randint(1, num_floors - 1), 0 
    elif 13 <= hour < 14:
        if r < 0.5:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0   
    elif 14 <= hour < 16:
        if r < 0.3:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.6:
            return random.randint(1, num_floors - 1), 0
    elif 16 <= hour < 17:
        if r < 0.2:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.7:
            return random.randint(1, num_floors - 1), 0
    elif 17 <= hour < 18:
        if r < 0.15:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0
    elif 18 <= hour < 20:
        if r < 0.1:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.9:
            return random.randint(1, num_floors - 1), 0
    elif 20 <= hour < 22:
        if r < 0.2:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0
    else: # Night time
        if r < 0.3:
            return 0, random.randint(1, num_floors - 1)
        elif r < 0.8:
            return random.randint(1, num_floors - 1), 0
    origin = random.randint(0, num_floors - 1)
    possible = list(range(num_floors))
    possible.remove(origin)
    destination = random.choice(possible)
    return origin, destination

def build_smoothed_rate_lookup():
    base = [get_dynamic_arrival_rate(minute) for minute in range(1440)]
    series = pd.Series(base)
    smoothed = series.rolling(window=15, center=True, min_periods=1).mean()
    return smoothed.to_dict()

def passenger_generator(env, num_floors, smoothed_rate_map, logs):
    id = 0
    while True:
        t = int(env.now) % 1440
        rate = smoothed_rate_map.get(t, 0.1)
        yield env.timeout(random.expovariate(rate))
        origin, destination = directional_bias(env, num_floors)
        passenger = Passenger(id, origin, destination, env.now)
        logs.append({
            'id': id,
            'origin': origin,
            'destination': destination,
            'arrival_time': env.now
        })
        id += 1

def generate_passengers(sim_time=1440*5, num_floors=10, output_csv='passenger_arrivals3.csv'):
    env = simpy.Environment()
    logs = []
    smoothed_rate_map = build_smoothed_rate_lookup()
    env.process(passenger_generator(env, num_floors, smoothed_rate_map, logs))
    env.run(until=sim_time)
    df = pd.DataFrame(logs)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Generated {len(df)} passengers and saved to {output_csv}")

    def classify_trip(row):
        if row['origin'] == 0:
            return 'incoming'
        elif row['destination'] == 0:
            return 'outgoing'
        else:
            return 'interfloor'

    df['trip_type'] = df.apply(classify_trip, axis=1)
    # Adjust Time Index
    df['arrival_bin'] = (df['arrival_time'] // 5) * 5   # grouping passenger arrivals into 5-minute time bins
    df['arrival_bin'] = df['arrival_bin'] % 1440   # wrap around to 24-hour format

    num_days = df['arrival_time'].max() / 1440
    # Count passengers in each 5-min bin and trip type
    rate_df = df.groupby(['arrival_bin', 'trip_type']).size().unstack(fill_value=0)
    # Normalize by number of simulation days to get average passengers per 5-min bin
    rate_df = rate_df.div(num_days)

    rate_df = rate_df.sort_index()
    rate_df.index = rate_df.index.astype(int)
    rate_df['time'] = pd.to_datetime(rate_df.index * 60, unit='s').strftime('%H:%M')
    rate_df = rate_df.set_index('time')

    rate_df.rolling(window=2, center=True, min_periods=1).mean().plot.area(
        stacked=True,
        color=['green', 'gold', 'darkorange'],
        figsize=(12, 6),
        alpha=0.8
    )
    plt.title('#passengers vs. Time')
    plt.ylabel('Num of Passengers')
    plt.xlabel('Time of Day')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('arrival_rate.png')
    print(f"[INFO] Generated arrival_rate.png")

if __name__ == '__main__':
    generate_passengers()
