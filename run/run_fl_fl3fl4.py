import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import fl_fl3fl4

def main():
    experiment = fl_fl3fl4.FederatedLearningFL3FL4('3OEC_current_flow.csv', 2, 100, 'avg', '5min',
                                   12, 'mps')
    save_dir = f"./out/Fl_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(1, save_dir)

if __name__=='__main__':
    main()