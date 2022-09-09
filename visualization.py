import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import argparse
import configparser
import os


parser = argparse.ArgumentParser()
parser.add_argument("--output_file", default='output_epoch_72_test.npz', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    print('Read output file: ',args.output_file)
    os.makedirs('img_visualize', exist_ok=True)
    path = os.path.join('img_visualize', args.output_file.split('.')[0]+'.png')
    data = np.load(args.output_file)
    sample_output = data['prediction'][0]  # prediction
    sample_labels = data['data_target_tensor'][0] # truth
    figure(figsize=(30,4), dpi=80)
    for i in range(50):
        new_i = i * 12
        plt.plot(range(0+new_i,12+new_i),sample_output[i], color = 'red')
        plt.plot(range(0+new_i,12+new_i),sample_labels[i], color='blue')
    plt.savefig(path)
    print('Save img successfully to :', path)