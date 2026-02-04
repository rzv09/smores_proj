import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step.cross_region_forgetting import CrossRegionForgetting

def main():
    runs = 1
    exp = CrossRegionForgetting(
    data_path_sb='DO_allsites_allyears_20250611.csv',
    data_path_fl='3OEC_current_flow.csv',
    num_train_epochs=100,
    sampling_freq='5min',
    sequence_len=12,
    device='mps',
)
    save_dir = f"./out/cross_region_rotating_{runs}runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    exp.run_experiment(
    num_runs=1,
    save_dir=save_dir,
    train_order=["FL1", "FL2", "FL3", "FL4", "SB1", "SB2", "SB3", "SB4"],
    eval_all=False,
)

if __name__=='__main__':
    main()