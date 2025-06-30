import argparse
import simpy
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_elevator_timeline(elevator, start_time=1080, end_time=1140, csv_file="elevator_timeline.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        'response': {'color': 'grey', 'linestyle': ':', 'marker': 'o', 'label': 'response'},
        'servicing': {'color': 'black', 'linestyle': '-', 'marker': 'o', 'label': 'servicing'},
        'raw': {'color': 'red', 'linestyle': '--', 'marker': 'x', 'label': 'raw'}  # fallback
    }

    for segment in elevator.timeline:
        seg_start = segment['start_time']
        seg_end = segment['end_time']

        # Skip if fully outside
        if seg_end < start_time or seg_start > end_time:
            continue

        # Clip to window
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
                linestyle='None')  # Make sure it's only a dot!

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Status")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Floor")
    ax.set_title(f"Elevator {elevator.id} Timeline ({start_time}–{end_time} min)")

    plt.savefig(csv_file)
    plt.show()



def get_config():
    parser = argparse.ArgumentParser(description="Elevator simulation configuration")

    parser.add_argument("--speed", type=float, default=12.0,
                        help="Elevator speed in floors per minute")
    parser.add_argument("--capacity", type=int, default=8,
                        help="Maximum number of passengers per elevator")
    parser.add_argument("--sim_time", type=int, default=1440 * 5,
                        help="Total simulation time in minutes")
    parser.add_argument("--num_floors", type=int, default=10,
                        help="Number of floors in the building")
    parser.add_argument("--num_elevators", type=int, default=2,
                        help="Number of elevators in the system")
    parser.add_argument("--input", type=str, default="passengers_dataset/fixed_distribution3",
                        help="Input CSV file with passenger arrival data")
    parser.add_argument("--output_csv", type=str, default="elevator_results.csv",
                        help="Output CSV file for storing results")

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
        self.requests = []  # queue of waiting passengers
        self.speed = config.speed
        self.num_floors = num_floors
        self.capacity = config.capacity
        self.action = env.process(self.run())
        self.total_floors_traveled = 0
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
                self.direction = 0
                yield self.env.timeout(0.1)
                continue

            all_stops = []
            for p in self.requests:
                all_stops.append(p.origin)
            for p in self.passengers:
                all_stops.append(p.destination)

            all_stops = sorted(set(all_stops), key=lambda x: abs(self.current_floor - x))

            for target_floor in all_stops:
                start_time = self.env.now
                start_floor = self.current_floor

                yield self.env.process(self.move_to(target_floor))

                # Determine type:
                if any(p.origin == target_floor for p in self.requests):
                    segment_status = 'response'
                elif any(p.destination == target_floor for p in self.passengers):
                    segment_status = 'servicing'
                else:
                    segment_status = 'raw'  # fallback, should not happen here

                self.timeline.append({
                    'start_time': start_time,
                    'end_time': self.env.now,
                    'start_floor': start_floor,
                    'end_floor': target_floor,
                    'status': segment_status
                })

                drop_passengers = [p for p in self.passengers if p.destination == target_floor]
                for p in drop_passengers:
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

                pickup_passengers = [p for p in self.requests if p.origin == target_floor]
                for p in pickup_passengers:
                    if len(self.passengers) < self.capacity:
                        p.pickup_time = self.env.now
                        self.passengers.append(p)
                        self.requests.remove(p)


class Dispatcher:
    def __init__(self, env, elevators):
        self.env = env
        self.elevators = elevators
        self.logs = []

    def assign_passenger(self, passenger):
        idle_elevators = []
        servicing_elevators = []

        for elevator in self.elevators:
            if not elevator.passengers and not elevator.requests:
                # Idle elevator
                distance = abs(elevator.current_floor - passenger.origin)
                idle_elevators.append((elevator, distance))
            else:
                # Servicing elevator
                queue_size = len(elevator.requests)
                all_dests = [p.destination for p in elevator.passengers]
                last_stop = all_dests[-1] if all_dests else elevator.current_floor
                distance_to_request = abs(last_stop - passenger.origin)
                servicing_elevators.append((elevator, queue_size, distance_to_request))

        if idle_elevators:
            # Sort by: nearest distance
            random.shuffle(idle_elevators)
            idle_elevators.sort(key=lambda x: x[1])
            chosen_elevator = idle_elevators[0][0]
        else:
            # Sort by: fewest requests → nearest last stop
            random.shuffle(servicing_elevators)
            servicing_elevators.sort(key=lambda x: (x[1], x[2]))
            chosen_elevator = servicing_elevators[0][0]

        chosen_elevator.requests.append(passenger)


    def log_passenger(self, passenger, elevator_id):
        self.logs.append({
            'id': passenger.id,
            'elevator_id': elevator_id,
            'origin': passenger.origin,
            'destination': passenger.destination,
            'arrival_time': passenger.arrival_time,
            'pickup_time': passenger.pickup_time,
            'dropoff_time': passenger.dropoff_time
        })

def passenger_generator(env, dispatcher, passenger_df):
    passenger_df = passenger_df.sort_values(by='arrival_time').reset_index(drop=True)

    for _, row in passenger_df.iterrows():
        delay = row['arrival_time'] - env.now
        if delay > 0:
            yield env.timeout(delay)

        passenger = Passenger(
            id=row['id'],
            origin=row['origin'],
            destination=row['destination'],
            arrival_time=row['arrival_time']
        )
        dispatcher.assign_passenger(passenger)

def run_simulation(config, input_file):
    df = pd.read_csv(input_file)
    env = simpy.Environment()
    elevators = [Elevator(env, i, config.num_floors, None, config) for i in range(config.num_elevators)]
    dispatcher = Dispatcher(env, elevators)
    for elevator in elevators:
        elevator.dispatcher = dispatcher

    env.process(passenger_generator(env, dispatcher, df))
    env.run(until=config.sim_time)
    for elevator in elevators:
        plot_elevator_timeline(elevator, csv_file=f"baseline_elevator_timeline_{elevator.id}.png")

    result_df = pd.DataFrame(dispatcher.logs)
    result_df['wait_time'] = result_df['pickup_time'] - result_df['arrival_time']

    # Compute summary statistics
    avg_wait = result_df['wait_time'].mean()
    max_wait = result_df['wait_time'].max()
    min_wait = result_df['wait_time'].min()
    q1 = np.percentile(result_df['wait_time'], 25)
    q2 = np.percentile(result_df['wait_time'], 50)
    q3 = np.percentile(result_df['wait_time'], 75)
    total_floors_traveled = sum(elevator.total_floors_traveled for elevator in elevators)

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
        file_path = f"{config.input}/{file}"
        stats = run_simulation(config, file_path)
        all_results.append(stats)

        print(f"Average waiting time:     {stats['avg']:.2f} minutes")
        print(f"Max waiting time:         {stats['max']:.2f} minutes")
        print(f"Min waiting time:         {stats['min']:.2f} minutes")
        print(f"Q1 waiting time:          {stats['q1']:.2f} minutes")
        print(f"Median (Q2) waiting time: {stats['q2']:.2f} minutes")
        print(f"Q3 waiting time:          {stats['q3']:.2f} minutes")
        print(f"Total floors traveled:    {stats['total_floors_traveled']} floors")

    # Compute averages over all runs
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
    print(f"Total floors traveled:    {avg_overall['total_floors_traveled']} floors")