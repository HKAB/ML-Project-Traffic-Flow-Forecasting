from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from functools import partial
from statsforecast.models import AutoARIMA
import warnings
import argparse
warnings.filterwarnings('ignore')

def run_arima(x_past):
    model = AutoARIMA(season_length=4)
    model.fit(x_past)
    return model.predict(12)['mean']

parser = argparse.ArgumentParser()
parser.add_argument("--sensor_s", default=0, type=int,
                    help="Index of sensor")
parser.add_argument("--sensor_e", default=0, type=int,
                    help="Index of sensor")
parser.add_argument("--num_workers", default=8, type=int,
                    help="Number of workers")
args = parser.parse_args()

if __name__ == '__main__':
    
    # given a dataframe, df 
    data = np.load('./data/PEMS04/PEMS04_r1_d0_w0_astcgn.npz')
    
    sensor_s = int(args.sensor_s)
    sensor_e = int(args.sensor_e)
    workers = int(args.num_workers)
    
    model = run_arima

    
    for sensor_idx in range(sensor_s, sensor_e + 1):
        
        p = mp.Pool(workers)
        
        chunked_data = data['val_x'][:, :, 0, :].swapaxes(0, 1)[sensor_idx]

        # pass the model and its params to a new partial object
        model_ = partial(model)

        # iterate over the partial object and the data
        # wrap the object inside tqdm to get a progress bar
        results = list(tqdm(p.imap(model_, chunked_data)))

        # close out the pool
        p.close()
        p.join()
        with open(f'output_sensor_{sensor_idx}.np', 'wb+') as f:
            np.save(f, results)
