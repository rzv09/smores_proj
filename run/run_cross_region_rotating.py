import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step.cross_region_rotating import CrossRegionRotating

def main():
    runs = 1
    experiment = CrossRegionRotating('DO_allsites_allyears_20250611.csv', '3OEC_current_flow.csv', 100, '5min',
                                   12, 'mps')
    save_dir = f"./out/cross_region_rotating_{runs}runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(runs, save_dir, train_labels=["FL1", "FL2", "FL3", "SB1", "SB2", "SB3"],
    test_labels=["FL4", "SB4"],
    # rotate_pairs=[("FL4", "SB4"), ("FL3", "SB3"), ("FL2", "SB2")],
    include_base=True,)

if __name__=='__main__':
    main()