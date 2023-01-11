import model
from Data import Data_pre, Dataloader
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import pytorch_lightning as pl
from transformers import BertTokenizer
import pandas as pd

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
batch = 2
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

x = "text"
y_1 = "UserPlase_latitude"
y_2 = "UserPlase_longitude"
df = pd.read_csv(
    "/data1/ohnishi/202271month_per_hour_geotaged_adGeocode_undersampled.csv"
)
df_train, df_val, df_test = Data_pre(df, x)
dataloader_train = Dataloader(df_train, x, y_1, y_2, batch)
dataloader_val = Dataloader(df_val, x, y_1, y_2, batch)
dataloader_test = Dataloader(df_test, x, y_1, y_2, batch)

model = model.BertForSequenceliner_pl(model_name=MODEL_NAME, lr=1e-5)
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="../models",
)
trainer = pl.Trainer(gpus=1, max_epochs=5, callbacks=[checkpoint])
trainer.fit(model, dataloader_train, dataloader_val)

test = trainer.test(dataloaders=dataloader_test)
