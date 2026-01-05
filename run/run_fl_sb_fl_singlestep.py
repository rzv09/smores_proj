import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import fl_sb_fl

def main():
    experiment = fl_sb_fl.FederatedLearningSBFL('DO_allsites_allyears_20250611.csv', '3OEC_current_flow.csv', 10, 40, 'avg', '5min',
                                   12, 'mps')
    save_dir = f"./out/Fl_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(1, save_dir)

if __name__=='__main__':
    main()