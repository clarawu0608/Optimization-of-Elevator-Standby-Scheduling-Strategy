import argparse
import simpy
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


# ---------- Config ----------
def get_config():
    parser = argparse.ArgumentParser(description="THV Duplex Elevator Simulation")
    parser.add_argument("--speed", type=float, default=12.0)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--sim_time", type=int, default=1440*5)
    parser.add_argument("--num_floors", type=int, default=20)
    parser.add_argument("--num_elevators", type=int, default=4)
    parser.add_argument("--input", type=str, default="passengers_dataset/20floors/")
    return parser.parse_args()

# ---------- Entities ----------
class Passenger:
    def __init__(self, id, origin, destination, arrival_time):
        self.id = id
        self.origin = origin
        self.destination = destination
        self.arrival_time = arrival_time
        self.pickup_time = None
        self.dropoff_time = None

class Elevator:
    def __init__(self, env, id, num_floors, dispatcher, config):
        self.env = env
        self.id = id
        self.dispatcher = dispatcher
        self.current_floor = 0
        self.speed = config.speed
        self.capacity = config.capacity
        self.num_floors = num_floors
        self.passengers = []
        self.requests = []
        self.direction = 0
        self.total_floors_traveled = 0
        self.timeline = []
        self.action = env.process(self.run())

    def move_to(self, floor):
        travel_time = abs(floor - self.current_floor) / self.speed
        self.total_floors_traveled += abs(floor - self.current_floor)
        yield self.env.timeout(travel_time)
        self.current_floor = floor

    def run(self):
        while True:
            if not self.requests and not self.passengers:
                self.direction = 0
                yield self.env.timeout(0.1)
                continue

            all_stops = list({*[p.origin for p in self.requests], *[p.destination for p in self.passengers]})
            all_stops.sort(key=lambda x: abs(self.current_floor - x))

            for target in all_stops:
                start_time = self.env.now
                start_floor = self.current_floor
                yield self.env.process(self.move_to(target))

                status = 'response' if any(p.origin == target for p in self.requests) else 'servicing'
                self.timeline.append({
                    'start_time': start_time,
                    'end_time': self.env.now,
                    'start_floor': start_floor,
                    'end_floor': target,
                    'status': status
                })

                drop = [p for p in self.passengers if p.destination == target]
                for p in drop:
                    p.dropoff_time = self.env.now
                    self.dispatcher.logs.append({
                        'id': p.id,
                        'elevator_id': self.id,
                        'origin': p.origin,
                        'destination': p.destination,
                        'arrival_time': p.arrival_time,
                        'pickup_time': p.pickup_time,
                        'dropoff_time': p.dropoff_time
                    })
                    self.passengers.remove(p)

                pick = [p for p in self.requests if p.origin == target]
                for p in pick:
                    if len(self.passengers) < self.capacity:
                        p.pickup_time = self.env.now
                        self.passengers.append(p)
                        self.requests.remove(p)

# ---------- THV Dispatcher ----------
class Dispatcher:
    def __init__(self, env, elevators):
        self.env = env
        self.elevators = elevators
        self.logs = []

    def compute_thv(self, elevator, passenger):
        travel_time = abs(elevator.current_floor - passenger.origin) / elevator.speed
        num_stops = len(elevator.requests) + len(elevator.passengers)
        direction_penalty = 0
        if elevator.passengers:
            car_direction = np.sign(elevator.passengers[0].destination - elevator.current_floor)
            desired_direction = np.sign(passenger.origin - elevator.current_floor)
            if car_direction != desired_direction:
                direction_penalty = 10  # arbitrary
        return travel_time + num_stops * 2 + direction_penalty

    def assign_passenger(self, passenger):
        costs = [(self.compute_thv(e, passenger), e) for e in self.elevators]
        chosen = min(costs, key=lambda x: x[0])[1]
        chosen.requests.append(passenger)

# ---------- Passenger Generator ----------
def passenger_generator(env, dispatcher, df):
    df = df.sort_values('arrival_time')
    for _, row in df.iterrows():
        delay = row['arrival_time'] - env.now
        if delay > 0:
            yield env.timeout(delay)
        p = Passenger(row['id'], row['origin'], row['destination'], row['arrival_time'])
        dispatcher.assign_passenger(p)

# ---------- Run ----------
def run_simulation(config, input_file):
    df = pd.read_csv(input_file)
    env = simpy.Environment()
    elevators = [Elevator(env, i, config.num_floors, None, config) for i in range(config.num_elevators)]
    dispatcher = Dispatcher(env, elevators)
    for elevator in elevators:
        elevator.dispatcher = dispatcher

    env.process(passenger_generator(env, dispatcher, df))
    env.run(until=config.sim_time)
    
    result_df = pd.DataFrame(dispatcher.logs)
    result_df['wait_time'] = result_df['pickup_time'] - result_df['arrival_time']

    avg_wait = result_df['wait_time'].mean()
    max_wait = result_df['wait_time'].max()
    min_wait = result_df['wait_time'].min()
    q1 = np.percentile(result_df['wait_time'], 25)
    q2 = np.percentile(result_df['wait_time'], 50)
    q3 = np.percentile(result_df['wait_time'], 75)
    total_floors_traveled = sum(elevator.total_floors_traveled for elevator in elevators)

    print(f"\n========== Results for {input_file} ==========")
    print(f"Average waiting time:     {avg_wait:.2f} minutes")
    print(f"Max waiting time:         {max_wait:.2f} minutes")
    print(f"Min waiting time:         {min_wait:.2f} minutes")
    print(f"Q1 waiting time:          {q1:.2f} minutes")
    print(f"Median (Q2) waiting time: {q2:.2f} minutes")
    print(f"Q3 waiting time:          {q3:.2f} minutes")
    print(f"Total floors traveled:    {total_floors_traveled:.2f} floors")

    return {
        "avg": avg_wait,
        "max": max_wait,
        "min": min_wait,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "total_floors_traveled": total_floors_traveled
    }


if __name__ == '__main__':
    config = get_config()

    files = ["passenger_arrivals1.csv", "passenger_arrivals2.csv", "passenger_arrivals3.csv"]
    all_results = []

    for file in files:
        print(f"\n[INFO] Running simulation for {file}")
        file_path = f"{config.input.rsplit('/', 1)[0]}/{file}"  # build path robustly
        stats = run_simulation(config, file_path)
        all_results.append(stats)

    avg_overall = {
        key: np.mean([res[key] for res in all_results])
        for key in all_results[0]
    }

    print("\n========== Overall Averages Across 3 Runs ==========")
    print(f"Average waiting time:     {avg_overall['avg']:.2f} minutes")
    print(f"Max waiting time:         {avg_overall['max']:.2f} minutes")
    print(f"Min waiting time:         {avg_overall['min']:.2f} minutes")
    print(f"Q1 waiting time:          {avg_overall['q1']:.2f} minutes")
    print(f"Median (Q2) waiting time: {avg_overall['q2']:.2f} minutes")
    print(f"Q3 waiting time:          {avg_overall['q3']:.2f} minutes")
    print(f"Total floors traveled:    {avg_overall['total_floors_traveled']:.2f} floors")
