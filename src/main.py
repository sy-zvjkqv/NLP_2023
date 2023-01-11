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
y = "code"
df = pd.read_csv(
    '/data1/ohnishi/202271month_per_hour_geotaged_adGeocode_undersampled.csv'
)
df_train, df_val, df_test, num_class = Data_pre(df, x, y)
dataloader_train = Dataloader(df_train, x, y, batch)
dataloader_val = Dataloader(df_val, x, y, batch)
dataloader_test = Dataloader(df_test, x, y, batch)

model = model.BertForSequenceClassifier_pl(
    model_name=MODEL_NAME, lr=1e-5, num_class=num_class
)
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
