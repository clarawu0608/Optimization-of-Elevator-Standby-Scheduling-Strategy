import argparse
import simpy
import pandas as pd
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

def get_config():
    parser = argparse.ArgumentParser(description="Elevator simulation configuration")

    parser.add_argument("--speed", type=float, default=12.0,
                        help="Elevator speed in floors per minute")
    parser.add_argument("--capacity", type=int, default=8,
                        help="Maximum number of passengers per elevator")
    parser.add_argument("--standby_threshold", type=float, default=0.2,
                        help="Time (minutes) before an elevator enters standby mode")
    parser.add_argument("--sim_time", type=int, default=1440 * 5,
                        help="Total simulation time in minutes")
    parser.add_argument("--num_floors", type=int, default=20,
                        help="Number of floors in the building")
    parser.add_argument("--num_elevators", type=int, default=4,
                        help="Number of elevators in the system")
    parser.add_argument("--input", type=str, default="passengers_dataset/20floors",
                        help="Input CSV file with passenger arrival data")
    parser.add_argument("--output_csv", type=str, default="standby_elevator_results.csv",
                        help="Output CSV file for storing results")
    parser.add_argument("--w", type=float, default=0.7,
                        help="Weight for wait time in standby floor calculation")
    parser.add_argument("--std", type=float, default=1.0,
                        help="Standard deviation for energy score in standby floor calculation")
    parser.add_argument("--window", type=float, default=300.0,
                        help="Time window for decay-weighted arrivals in standby floor calculation")
    parser.add_argument("--decay_mode", type=str, default="quadratic", choices=["linear", "exponential", "quadratic"],
                        help="Decay mode for scoring passenger arrival times")
    parser.add_argument("--use_sliding_window", type=bool, default=False,
                        help="Use sliding window for combining scores in standby floor calculation")
    parser.add_argument("--optimal_cost_func", type=bool, default=True,
                        help="Use optimal cost function for standby floor calculation instead of simplely use the argmax")
    parser.add_argument("--dynamic_window_size", type=bool, default=False)
    parser.add_argument("--type_parameter", type=int, default=1, choices=[1, 1.5, 3, 6],
                        help="A parameter to control window shape")
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
        self.direction = 0
        self.passengers = []
        self.requests = []
        self.speed = config.speed
        self.num_floors = num_floors
        self.capacity = config.capacity
        self.standby_threshold = config.standby_threshold
        self.w = config.w
        self.std = config.std
        self.window = config.window
        self.last_active_time = env.now
        self.action = env.process(self.run())
        self.decay_mode = config.decay_mode
        self.use_sliding_window = config.use_sliding_window
        self.optimal_cost_func = config.optimal_cost_func
        self.dynamic_window_size = config.dynamic_window_size
        self.state = "idle"
        self.standby_proc = None  # to store standby process reference
        self.interrupt_times = []  # track interruptions
        self.standby_success_flags = []  # track success of uninterruptted standbys
        self.standby_score = []  # track scores for standby decisions
        self.standby_score_denominator = []  # denominator for standby score calculation
        self.last_standby_origin = None
        self.last_standby_destination = None
        self.total_floors_traveled = 0
        self.type = config.type_parameter
        self.timeline = []

    def move_to(self, floor):
        travel_time = abs(floor - self.current_floor) / self.speed
        floors_moved = abs(floor - self.current_floor)
        self.total_floors_traveled += floors_moved
        yield self.env.timeout(travel_time)
        self.current_floor = floor

    def run(self):
        while True:
            if not self.requests and not self.passengers:
                self.state = "idle"
                idle_duration = self.env.now - self.last_active_time
                if idle_duration >= self.standby_threshold:
                    standby_floor = self.compute_standby_floor()
                    if standby_floor != self.current_floor:
                        self.state = "moving_to_standby"
                        travel_start = self.env.now
                        start_floor = self.current_floor
                        target_floor = standby_floor
                        self.last_standby_origin = start_floor
                        self.last_standby_destination = target_floor

                        segment = {
                            'start_time': travel_start,
                            'end_time': None,
                            'start_floor': start_floor,
                            'end_floor': target_floor,
                            'status': 'standby'
                        }

                        self.standby_proc = self.env.process(self.move_to(target_floor))

                        try:
                            yield self.standby_proc
                            segment['end_time'] = self.env.now
                            self.timeline.append(segment)
                            self.last_active_time = self.env.now
                            self.state = "idle"
                        except simpy.Interrupt:
                            interrupted_at = self.env.now
                            travel_time = abs(target_floor - start_floor) / self.speed
                            ratio = (interrupted_at - travel_start) / travel_time
                            ratio = max(0, min(1, ratio))
                            direction = np.sign(target_floor - start_floor)
                            new_floor = start_floor + ratio * abs(target_floor - start_floor) * direction
                            self.current_floor = new_floor

                            segment['end_time'] = interrupted_at
                            segment['end_floor'] = new_floor
                            segment['status'] = 'standby (interrupted)'
                            self.timeline.append(segment)

                            self.last_active_time = self.env.now
                            self.state = "idle"

                yield self.env.timeout(0.1)
                continue

            self.last_active_time = self.env.now
            self.state = "servicing"

            all_stops = list(set(
                [p.origin for p in self.requests] +
                [p.destination for p in self.passengers]
            ))
            all_stops.sort(key=lambda x: abs(self.current_floor - x))

            for target_floor in all_stops:
                move_start = self.env.now
                start_floor = self.current_floor

                yield self.env.process(self.move_to(target_floor))

                # ✅ Decide status: response or servicing
                is_pickup = any(p.origin == target_floor for p in self.requests)
                is_dropoff = any(p.destination == target_floor for p in self.passengers)

                if is_pickup:
                    segment_status = 'response'
                elif is_dropoff:
                    segment_status = 'servicing'
                else:
                    segment_status = 'raw'  # fallback if neither

                self.timeline.append({
                    'start_time': move_start,
                    'end_time': self.env.now,
                    'start_floor': start_floor,
                    'end_floor': target_floor,
                    'status': segment_status
                })

                # Drop off passengers
                for p in [p for p in self.passengers if p.destination == target_floor]:
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

                # Pick up waiting passengers
                for p in [p for p in self.requests if p.origin == target_floor]:
                    if len(self.passengers) < self.capacity:
                        p.pickup_time = self.env.now
                        self.passengers.append(p)
                        self.requests.remove(p)



    def compute_standby_floor(self):
        num_floors = self.num_floors
        return num_floors // 2  # Default to middle floor if no requests


class Dispatcher:
    def __init__(self, env, elevators, num_floors):
        self.env = env
        self.elevators = elevators
        self.logs = []
        self.history = {floor: deque() for floor in range(num_floors)}

    def assign_passenger(self, passenger):
        self.history[passenger.origin].append(passenger.arrival_time)

        idle_or_standby = []
        servicing = []

        for elevator in self.elevators:
            is_idle = not elevator.passengers and not elevator.requests
            is_moving_to_standby = elevator.state == "moving_to_standby"
            is_servicing = not is_idle and not is_moving_to_standby

            distance = abs(elevator.current_floor - passenger.origin)

            if is_idle or is_moving_to_standby:
                direction_match = (
                    (passenger.destination > passenger.origin and elevator.current_floor <= passenger.origin) or
                    (passenger.destination < passenger.origin and elevator.current_floor >= passenger.origin)
                )
                idle_or_standby.append((elevator, distance, direction_match, is_idle))
            elif is_servicing:
                queue_len = len(elevator.requests)
                all_stops = [p.destination for p in elevator.passengers] + [p.origin for p in elevator.requests]
                last_stop = all_stops[-1] if all_stops else elevator.current_floor
                distance_to_request = abs(last_stop - passenger.origin)
                servicing.append((elevator, queue_len, distance_to_request))

        if idle_or_standby:
            # Sort by: distance → direction match → is_idle
            random.shuffle(idle_or_standby)
            idle_or_standby.sort(key=lambda x: (x[1], not x[2], not x[3]))
            chosen = idle_or_standby[0][0]

            if chosen.state == "moving_to_standby":
                chosen.standby_proc.interrupt()  # interrupt standby move
                chosen.state = "idle"
                chosen.last_active_time = self.env.now  # reset idle timer

            chosen.requests.append(passenger)
        else:
            random.shuffle(servicing)
            servicing.sort(key=lambda x: (x[1], x[2]))
            chosen = servicing[0][0]
            chosen.requests.append(passenger)

        # Check for standby success flag update
        if hasattr(chosen, 'last_standby_origin') and chosen.last_standby_origin is not None:
            origin = chosen.last_standby_origin
            dest = chosen.last_standby_destination
            req_origin = passenger.origin
            success = abs(origin - req_origin) > abs(dest - req_origin)
            score = abs(origin - req_origin) - abs(dest - req_origin)
            denominator = abs(origin - dest)
            chosen.standby_success_flags.append(success)
            chosen.standby_score.append(score)
            chosen.standby_score_denominator.append(denominator)
            chosen.last_standby_origin = None
            chosen.last_standby_destination = None



def passenger_generator(env, dispatcher, passenger_df):
    passenger_df = passenger_df.sort_values(by='arrival_time').reset_index(drop=True)
    for _, row in passenger_df.iterrows():
        delay = row['arrival_time'] - env.now
        if delay > 0:
            yield env.timeout(delay)
        passenger = Passenger(row['id'], row['origin'], row['destination'], row['arrival_time'])
        dispatcher.assign_passenger(passenger)

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

    # 1) Car upward energy from timeline
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

    # 2) Passenger upward energy from completed trips in logs
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

def run_simulation(config, input_file):
    df = pd.read_csv(input_file)
    env = simpy.Environment()
    dispatcher = Dispatcher(env, [], config.num_floors)
    elevators = [Elevator(env, i, config.num_floors, dispatcher, config) for i in range(config.num_elevators)]
    dispatcher.elevators = elevators

    env.process(passenger_generator(env, dispatcher, df))
    env.run(until=config.sim_time)
    # for elevator in elevators:
    #     plot_elevator_timeline(elevator, csv_file=f"standby_elevator_timeline_{elevator.id}.png")

    result_df = pd.DataFrame(dispatcher.logs)
    result_df['wait_time'] = result_df['pickup_time'] - result_df['arrival_time']
    # result_df.to_csv(config.output_csv, index=False)

    avg_wait = result_df['wait_time'].mean()
    max_wait = result_df['wait_time'].max()
    min_wait = result_df['wait_time'].min()
    q1 = np.percentile(result_df['wait_time'], 25)
    q2 = np.percentile(result_df['wait_time'], 50)
    q3 = np.percentile(result_df['wait_time'], 75)
    std_wait = np.std(result_df['wait_time'], ddof=1) if result_df['wait_time'].size > 1 else 0.0

    # Calculate standby stats
    total_successes = sum(sum(e.standby_success_flags) for e in elevators)
    total_attempts = sum(len(e.standby_success_flags) for e in elevators)
    avg_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0.0
    total_standby_score = sum(sum(e.standby_score) for e in elevators)
    total_standby_denominator = sum(sum(e.standby_score_denominator) for e in elevators)
    standby_score = (total_standby_score / total_standby_denominator) if total_standby_denominator > 0 else 0.0
    interruptions = sum(len(e.interrupt_times) for e in elevators)
    total_floors_traveled = sum(e.total_floors_traveled for e in elevators)

    energy = compute_total_energy_joules(elevators, dispatcher, config)
    energy_kwh = energy["E_total_J"] / 3.6e6  # J -> kWh

    return {
        "avg": avg_wait,
        "max": max_wait,
        "min": min_wait,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "std": std_wait,
        "total_floors_traveled": total_floors_traveled,
        "success_rate": avg_success_rate,
        "standby_score": standby_score,
        "standby_total_floors": total_standby_denominator,
        "interruptions": interruptions,
        "standby_avrage_moving_floors": total_standby_denominator / total_attempts if total_attempts > 0 else 0.0,
        "energy_total_J": energy["E_total_J"],
        "energy_total_kWh": energy_kwh,  
    }


def plot_elevator_timeline(elevator, start_time=1080, end_time=1140, csv_file="elevator_timeline.png"):
    """
    Plots the floor-time timeline of a single elevator for a specified time window.
    
    Args:
        elevator: Elevator instance with a .timeline list.
        start_time: Start of time window (minutes).
        end_time: End of time window (minutes).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        'response': {'color': 'green', 'linestyle': ':', 'marker': 'o', 'label': 'response'},
        'servicing': {'color': 'black', 'linestyle': '-', 'marker': 'o', 'label': 'servicing'},
        'standby': {'color': 'blue', 'linestyle': '--', 'marker': '^', 'label': 'standby'},
        'standby (interrupted)': {'color': 'red', 'linestyle': '-.', 'marker': 'x', 'label': 'standby (interrupted)'},
        'raw': {'color': 'grey', 'linestyle': '-', 'marker': '.', 'label': 'raw'}
    }

    for segment in elevator.timeline:
        seg_start = segment['start_time']
        seg_end = segment['end_time']

        if seg_end < start_time or seg_start > end_time:
            continue

        plot_start = max(seg_start, start_time)
        plot_end = min(seg_end, end_time)

        total_duration = seg_end - seg_start
        if total_duration == 0:
            continue

        f1 = (plot_start - seg_start) / total_duration if seg_start < plot_start else 0.0
        f2 = (plot_end - seg_start) / total_duration if seg_end > plot_end else 1.0

        floor_start = segment['start_floor'] + f1 * (segment['end_floor'] - segment['start_floor'])
        floor_end = segment['start_floor'] + f2 * (segment['end_floor'] - segment['start_floor'])

        style = styles.get(segment['status'], styles['raw'])

        ax.plot([plot_start-start_time, plot_end-start_time],
                [floor_start, floor_end],
                linestyle=style['linestyle'],
                color=style['color'],
                label=style['label'])

        ax.plot(plot_start-start_time, floor_start,
                marker=style['marker'],
                color=style['color'],
                linestyle='None')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Status")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Floor")
    ax.set_title(f"Elevator {elevator.id} Timeline ({start_time}–{end_time} min)")

    plt.savefig(csv_file)
    plt.show()


if __name__ == '__main__':
    config = get_config()
    files = ["passenger_arrivals1.csv", "passenger_arrivals2.csv", "passenger_arrivals3.csv"]
    all_results = []

    for file in files:
        # print(f"\n[INFO] Running simulation for {file}")
        file_path = f"{config.input}/{file}"
        stats = run_simulation(config, file_path)
        all_results.append(stats)

        # print(f"Average waiting time:     {stats['avg']:.2f} minutes")
        # print(f"Max waiting time:         {stats['max']:.2f} minutes")
        # print(f"Min waiting time:         {stats['min']:.2f} minutes")
        # print(f"Q1 waiting time:          {stats['q1']:.2f} minutes")
        # print(f"Median (Q2) waiting time: {stats['q2']:.2f} minutes")
        # print(f"Q3 waiting time:          {stats['q3']:.2f} minutes")
        # print(f"Total floors traveled:    {stats['total_floors_traveled']} floors")
        # print(f"Standby success rate:     {stats['success_rate']:.2f}%")
        # print(f"Standby score:            {stats['standby_score']:.2f}")
        # print(f"Standby total moving floors: {stats['standby_total_floors']}")
        # print(f"Interruptions:            {stats['interruptions']}")
        # print(f"Standby average moving floors: {stats['standby_avrage_moving_floors']:.2f}\n")

    # Compute averages across trials
    avg_result = {
        key: np.mean([res[key] for res in all_results])
        for key in all_results[0]
    }

    print("\n========== Overall Averages Across 3 Runs ==========")
    print(f"Average waiting time:     {avg_result['avg']:.2f} minutes")
    print(f"Max waiting time:         {avg_result['max']:.2f} minutes")
    print(f"Min waiting time:         {avg_result['min']:.2f} minutes")
    print(f"Q1 waiting time:          {avg_result['q1']:.2f} minutes")
    print(f"Median (Q2) waiting time: {avg_result['q2']:.2f} minutes")
    print(f"Q3 waiting time:          {avg_result['q3']:.2f} minutes")
    print(f"Standard deviation:       {avg_result['std']:.2f} minutes")
    print(f"Total floors traveled:    {avg_result['total_floors_traveled']} floors")
    print(f"Average standby success rate: {avg_result['success_rate']:.2f}%")
    print(f"Average standby score:    {avg_result['standby_score']:.2f}")
    print(f"Average standby total moving floors: {avg_result['standby_total_floors']}")
    print(f"Total interruptions:      {avg_result['interruptions']}")
    print(f"Average standby moving floors: {avg_result['standby_avrage_moving_floors']:.2f}")
    print(f"Average total energy:     {avg_result['energy_total_kWh']:.4f} kWh")
    print("\n\n")

