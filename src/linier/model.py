from transformers import BertModel
from torch import nn
import torch
import pytorch_lightning as pl

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
batch = 2


class BertForSequenceClassifier_pl(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds

    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", loss)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def validation_epoch_end(self, outputs, mode="val"):
        # loss計算
        epoch_preds = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class BertForSequenceliner_pl(pl.LightningModule):
        
    def __init__(self, model_name, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()
        
        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters()

        # BERTのロード
        self.bert = BertModel.from_pretrained(model_name)
        self.Linear = nn.Linear(self.bert.config.hidden_size,2)
        self.criterion = nn.MSELoss()

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, latitude=None,longitude=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.Linear(output.pooler_output)
        preds_latitude=preds[:,0]
        preds_longitude=preds[:,1]
        #print(f"this is preds{preds} preds_latitude{preds_latitude} preds_longitude{preds_longitude}")
        loss = 0
        #print(f"pres1 is{preds_latitude}     pres2is{preds_longitude}")
        #print(output.pooler_output)
        if latitude is not None:
            loss1 = self.criterion(preds_latitude, latitude)
            loss2 = self.criterion(preds_longitude, longitude)
            loss=loss1+loss2
        return loss, preds_latitude,preds_longitude
    
       # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds_latitude,preds_longitude = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    latitude=batch["latitude"],
                                    longitude=batch["longitude"])
        self.log('train_loss', loss)
        return {'loss': loss,
                'preds_latitude': preds_latitude,
                'preds_longitude': preds_longitude,
                'batch_latitude': batch["latitude"],
                'batch_longitude': batch["longitude"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds_latitude,preds_longitude = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    latitude=batch["latitude"],
                                    longitude=batch["longitude"])
        self.log('train_loss', loss)
        return {'loss': loss,
                'preds_latitude': preds_latitude,
                'preds_longitude': preds_longitude,
                'batch_latitude': batch["latitude"],
                'batch_longitude': batch["longitude"]}

    def test_step(self, batch, batch_idx):
        loss, preds_latitude,preds_longitude = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    latitude=batch["latitude"],
                                    longitude=batch["longitude"])
        self.log('train_loss', loss)
        return {'loss': loss,
                'preds_latitude': preds_latitude,
                'preds_longitude': preds_longitude,
                'batch_latitude': batch["latitude"],
                'batch_longitude': batch["longitude"]}

    # epoch終了時にvalidationのlossとaccuracyを記録
    def validation_epoch_end(self, outputs, mode="val"):
        # loss計算
        epoch_preds_latitude= torch.cat([x['preds_latitude'] for x in outputs])
        epoch_preds_longitude = torch.cat([x['preds_longitude'] for x in outputs])
        epoch_latitude = torch.cat([x['batch_latitude'] for x in outputs])
        epoch_longitude = torch.cat([x['batch_longitude'] for x in outputs])
        epoch_loss1 = self.criterion(epoch_preds_latitude, epoch_latitude)
        epoch_loss2 = self.criterion(epoch_preds_longitude, epoch_longitude)
        epoch_loss=epoch_loss1+epoch_loss2
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        self.log(f"{mode}_latitude", epoch_preds_latitude, logger=True)
        self.log(f"{mode}_longitude", epoch_preds_longitude, logger=True)

    # testデータのlossとaccuracyを算出（validationの使いまわし）
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)