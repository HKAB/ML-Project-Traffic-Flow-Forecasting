import numpy as np
from sklearn.metrics import mean_absolute_error
import sys

def ha(test_x, test_target):
    test_x = np.mean(test_x, axis=2)
    test_x = np.expand_dims(test_x, -1)
    prediction = np.repeat(test_x[:, :, :], 12, axis=2)
    mae = mean_absolute_error(prediction.reshape(-1, 1), test_target.reshape(-1, 1))
    return prediction, test_target, mae


data1 = np.load('data/PEMS04/PEMS04_r1_d0_w0_astcgn.npz')
test_x = data1['test_x'][:,:,0,:]
test_target = data1['test_target']
mean = data1['mean'][:,:,0,:]
std = data1['std'][:,:,0,:]
test_x = test_x*std + mean
prediction, data_target_tensor, mae = ha(test_x, test_target)


data2 = np.load('test_hdw.npz')
test_x_hdw = data2['test_x'][:,:,:,0,:]
data_target_tensor_hdw = data2['data_target_tensor']
mean_hdw = data2['mean'][:,:,:,:,0]
std_hdw = data2['std'][:,:,:,:,0]

test_x_hdw = test_x_hdw*std_hdw+mean_hdw
test_x_hdw = np.mean(test_x_hdw, axis=1)

prediction_hdw, data_target_tensor, mae_hdw = ha(test_x_hdw, data_target_tensor_hdw)

print('MAE HA:', mae)
print('MAE HA HDW:', mae_hdw)

np.savez('output_ha_test.npz', prediction=prediction, data_target_tensor=data_target_tensor)
np.savez('output_ha_hdw_test.npz', prediction=prediction_hdw, data_target_tensor=data_target_tensor_hdw)




