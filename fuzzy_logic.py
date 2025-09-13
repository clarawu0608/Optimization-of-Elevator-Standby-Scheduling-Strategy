import argparse
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# -------------- Config --------------
def get_config():
    parser = argparse.ArgumentParser(description="Fuzzy Logic Elevator Simulation")
    parser.add_argument("--speed", type=float, default=12.0)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--sim_time", type=int, default=1440*5)
    parser.add_argument("--num_floors", type=int, default=20)
    parser.add_argument("--num_elevators", type=int, default=4)
    parser.add_argument("--input", type=str, default="passengers_dataset/20floors/")
    parser.add_argument("--floor_height_m", type=float, default=3.5,
                    help="Height per floor (meters)")
    parser.add_argument("--car_mass_kg", type=float, default=1000.0,
                        help="Empty car mass (kg)")
    parser.add_argument("--avg_passenger_mass_kg", type=float, default=60.0,
                        help="Average mass per passenger (kg)")
    parser.add_argument("--start_energy_kj", type=float, default=5.0,
                        help="Energy to start from rest per move (kJ)")
    parser.add_argument("--stop_energy_kj", type=float, default=3.0,
                        help="Energy to stop at a floor per move (kJ)")
    parser.add_argument("--g", type=float, default=9.8,
                        help="Gravity (m/s^2)")
    return parser.parse_args()

# -------------- Entities --------------
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

# -------------- Fuzzy Dispatcher --------------
class Dispatcher:
    def __init__(self, env, elevators, config):
        self.env = env
        self.elevators = elevators
        self.logs = []
        self.max_distance = config.num_floors

    def fuzzy_distance(self, dist):
        if dist <= 2:
            return 0.1  # near
        elif dist <= 5:
            return 0.5  # medium
        else:
            return 0.9  # far

    def fuzzy_load(self, load_ratio):
        if load_ratio < 0.3:
            return 0.1  # low
        elif load_ratio < 0.7:
            return 0.5  # medium
        else:
            return 0.9  # high

    def fuzzy_direction(self, elevator, passenger):
        if elevator.passengers:
            car_dir = np.sign(elevator.passengers[0].destination - elevator.current_floor)
            req_dir = np.sign(passenger.origin - elevator.current_floor)
            if car_dir == req_dir:
                return 0.1  # match
            else:
                return 0.9  # conflict
        else:
            return 0.1  # idle -> match

    def compute_fuzzy_score(self, elevator, passenger):
        dist = abs(elevator.current_floor - passenger.origin)
        distance_score = self.fuzzy_distance(dist)
        load_ratio = (len(elevator.passengers) + len(elevator.requests)) / elevator.capacity
        load_score = self.fuzzy_load(load_ratio)
        direction_score = self.fuzzy_direction(elevator, passenger)

        # weighted sum, you can adjust weights
        return 0.5 * distance_score + 0.3 * load_score + 0.2 * direction_score

    def assign_passenger(self, passenger):
        scores = [(self.compute_fuzzy_score(e, passenger), e) for e in self.elevators]
        chosen = min(scores, key=lambda x: x[0])[1]
        chosen.requests.append(passenger)

# -------------- Passenger Generator --------------
def passenger_generator(env, dispatcher, df):
    df = df.sort_values('arrival_time')
    for _, row in df.iterrows():
        delay = row['arrival_time'] - env.now
        if delay > 0:
            yield env.timeout(delay)
        p = Passenger(row['id'], row['origin'], row['destination'], row['arrival_time'])
        dispatcher.assign_passenger(p)

def compute_total_energy_joules(elevators, dispatcher, config):
    """
    Energy model (Joules):
      E_total = E_car_up + E_passenger_up + E_start_stop

      - E_car_up: car mass * g * (total upward distance moved by the car)
      - E_passenger_up: sum over passengers only when destination > origin of
                        (avg_passenger_mass * g * (dest-origin)*floor_height)
      - E_start_stop: (#segments across all elevators) * (E_start + E_stop)
    """
    g = config.g
    h = config.floor_height_m
    m_car = config.car_mass_kg
    m_p = config.avg_passenger_mass_kg
    E_start = config.start_energy_kj * 1000.0
    E_stop = config.stop_energy_kj * 1000.0

    # 1) Car travel energy from timeline
    total_up_floors_car = 0.0
    total_segments = 0
    for el in elevators:
        total_segments += len(el.timeline)
        for seg in el.timeline:
            df = seg['end_floor'] - seg['start_floor']
            if df > 0:
                total_up_floors_car += df
            elif df < 0:
                total_up_floors_car += -df

    E_car_up = m_car * g * (total_up_floors_car * h)

    # 2) Passenger travel energy from completed trips in logs
    # dispatcher.logs rows are created at drop-off time
    E_passenger_up = 0.0
    for row in dispatcher.logs:
        df = row['destination'] - row['origin']
        if df > 0:
            E_passenger_up += m_p * g * (df * h)
        elif df < 0:
            E_passenger_up += m_p * g * (-df * h)

    # 3) Start/stop energy per segment
    E_start_stop = total_segments * (E_start + E_stop)

    return {
        "E_car_up_J": E_car_up,
        "E_passenger_up_J": E_passenger_up,
        "E_start_stop_J": E_start_stop,
        "E_total_J": E_car_up + E_passenger_up + E_start_stop
    }

# -------------- Run --------------
def run_simulation(config, input_file):
    df = pd.read_csv(input_file)
    env = simpy.Environment()
    elevators = [Elevator(env, i, config.num_floors, None, config) for i in range(config.num_elevators)]
    dispatcher = Dispatcher(env, elevators, config)
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
    std_wait = np.std(result_df['wait_time'], ddof=1) if result_df['wait_time'].size > 1 else 0.0
    total_floors_traveled = sum(elevator.total_floors_traveled for elevator in elevators)

    energy = compute_total_energy_joules(elevators, dispatcher, config)
    energy_kwh = energy["E_total_J"] / 3.6e6  # J -> kWh

    # print(f"\n========== Fuzzy Results for {input_file} ==========")
    # print(f"Average waiting time:     {avg_wait:.2f} minutes")
    # print(f"Max waiting time:         {max_wait:.2f} minutes")
    # print(f"Min waiting time:         {min_wait:.2f} minutes")
    # print(f"Q1 waiting time:          {q1:.2f} minutes")
    # print(f"Median (Q2) waiting time: {q2:.2f} minutes")
    # print(f"Q3 waiting time:          {q3:.2f} minutes")
    # print(f"Total floors traveled:    {total_floors_traveled:.2f} floors")

    return {
        "avg": avg_wait,
        "max": max_wait,
        "min": min_wait,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "std": std_wait,
        "total_floors_traveled": total_floors_traveled,
        "energy_total_J": energy["E_total_J"],
        "energy_total_kWh": energy_kwh,
    }


if __name__ == '__main__':
    config = get_config()

    files = ["passenger_arrivals1.csv", "passenger_arrivals2.csv", "passenger_arrivals3.csv"]
    all_results = []

    for file in files:
        print(f"\n[INFO] Running Fuzzy Logic simulation for {file}")
        file_path = f"{config.input.rsplit('/', 1)[0]}/{file}"
        stats = run_simulation(config, file_path)
        all_results.append(stats)

    avg_overall = {
        key: np.mean([res[key] for res in all_results])
        for key in all_results[0]
    }

    print("\n========== Overall Fuzzy Averages Across 3 Runs ==========")
    print(f"Average waiting time:     {avg_overall['avg']:.2f} minutes")
    print(f"Max waiting time:         {avg_overall['max']:.2f} minutes")
    print(f"Min waiting time:         {avg_overall['min']:.2f} minutes")
    print(f"Q1 waiting time:          {avg_overall['q1']:.2f} minutes")
    print(f"Median (Q2) waiting time: {avg_overall['q2']:.2f} minutes")
    print(f"Q3 waiting time:          {avg_overall['q3']:.2f} minutes")
    print(f"Total floors traveled:    {avg_overall['total_floors_traveled']:.2f} floors")
    print(f"Standard deviation:       {avg_overall['std']:.2f} minutes")
    print(f"Total floors traveled:    {avg_overall['total_floors_traveled']:.2f} floors")
    print(f"Average total energy:     {avg_overall['energy_total_kWh']:.4f} kWh")
