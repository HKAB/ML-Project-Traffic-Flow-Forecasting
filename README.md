
# Traffic flow forecasting

Traffic flow forecasting project

## Run

- on PEMS04 dataset

  ```shell
  python prepareData.py --config configurations/PEMS04_[MODEL].conf
  ```

  ```shell
  python train_[MODEL]_r.py --config configurations/PEMS04_[MODEL].conf
  ```

## Methods

- Historical Average ARIMA
- LSTM (1h), (hdw)
- ASTGCN (1h), (hdw)
- GCN + LSTM (1h), (hdw)

