import time, logging
import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from Projects.AnomalyDetection.tvld.baselines_src.xlstmad.dataset import SlidingWindowDataset
from Projects.AnomalyDetection.tvld.baselines_src.xlstmad.xlstmad import xLSTMAD

logging.basicConfig(level=logging.WARNING)

batch_size = 32
window_size = 20
validation_size = 0.2

full_data = pd.read_csv('data/sample_dataset.csv').values
train_length = 4000
train_val_split = int(train_length * (1 - validation_size))
train_data = full_data[:train_val_split]
val_data = full_data[train_val_split:train_length]
train_x, train_y = train_data[:, :-1], train_data[:, -1]
val_x, val_y = val_data[:, :-1], val_data[:, -1]

train_loader = DataLoader(SlidingWindowDataset(train_x, train_y, window_size=window_size), batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(SlidingWindowDataset(val_x, val_y, window_size=window_size), batch_size=4*batch_size, shuffle=False, num_workers=0)

t0=time.time()
model = xLSTMAD(embedding_dim=40, features_no=train_x.shape[1], window_size=window_size, slstm_backend="vanilla")
print("model init time", time.time()-t0)

trainer = L.Trainer(max_epochs=1, accelerator="cpu", callbacks=[], logger=False, enable_progress_bar=False, limit_train_batches=10, limit_val_batches=5, enable_checkpointing=False)
t0=time.time()
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
print("10 train batches + 5 val batches time:", time.time()-t0)
