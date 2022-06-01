import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from dataset import Img2Vec
from tabulate import tabulate
import pandas as pd
import time
import tracemalloc
import torch.nn as nn
import pytorch_lightning as pl
import math
import torchmetrics

cities = ['Philadelphia']
category = 'Restaurants'

reviews = pd.read_csv("yelp_dataset/yelp_academic_dataset_review.csv")
businesses = pd.read_csv("yelp_dataset/yelp_academic_dataset_business.csv",
                         converters={'categories': lambda x: set([x.strip() for x in x.split(',')])})
businesses.reset_index(inplace=True)
photos = pd.read_csv("yelp_dataset/photos.csv")

photos = photos[photos['label'] == 'inside']

tracemalloc.start()

reviews['review_id'] = reviews['review_id'].astype(str)
reviews['user_id'] = reviews['user_id'].astype(str)
reviews['business_id'] = reviews['business_id'].astype(str)
reviews['rating'] = reviews['rating'].astype(int)
reviews['timestamp'] = pd.to_datetime(reviews['timestamp'], format='%Y-%m-%d %H:%M:%S')

businesses = businesses[businesses.city.isin(cities)]
businesses = businesses[businesses['categories'].apply(lambda x: category in x)]
business_map = {k: v for v, k in enumerate(businesses.index)}
businesses['index'] = businesses['index'].map(business_map)
businesses = businesses.drop(['state', 'review_count'], axis=1)

photos = photos.groupby("business_id")["photo_id"].apply(list).reset_index(name="photo_ids")
photos["photo_ids"] = photos["photo_ids"].apply(lambda x: x[0])
photos = pd.merge(photos, businesses, left_on='business_id', right_on='id', how='right')
photos = photos.drop(['id', 'name', 'city', 'categories', 'business_id'], axis=1)
photos.columns = ['photo_ids', 'business_id']
photos = photos.dropna()

from PIL import UnidentifiedImageError

model = Img2Vec(model="resnet34", cuda=True)

failed = 0

def img_to_vec(photo_id):
    global failed
    try:
        if isinstance(photo_id, list):
            return model.get_vec([Image.open(f'photos/{x}.jpg') for x in photo_id], tensor=True)
        return model.get_vec(Image.open(f'photos/{photo_id}.jpg'))
    except UnidentifiedImageError:
        failed += 1
        return None


start_time = time.time()
photos.photo_ids = photos.photo_ids.apply(img_to_vec)
print(f'Time to produce {len(photos)} image embeddings: {(time.time() - start_time)//60} mins')
photos_processed = len(photos)
print(f'Total: {photos_processed}, failed: {failed}')
photos = photos.dropna()

reviews = reviews.drop(['review_id'], axis = 1)
reviews.columns = ['user_id', 'business_id', 'rating', 'timestamp']

df = pd.merge(reviews, businesses, left_on="business_id", right_on="id")
df = df.drop(['business_id', 'name', 'city', 'categories', 'id'], axis=1)
df.columns = ['user_id', 'rating', 'timestamp', 'business_id']
# df.to_csv('dataset_reset_index/reviews.csv', sep=",")
df = pd.merge(df, photos, on="business_id", how="left")
df = df.dropna()
df['business_id'] = df['business_id'].astype(int)
dct = {k: v for v, k in enumerate(df.business_id.unique())}
df.business_id = df.business_id.map(dct)
business_max = int(df.business_id.max())

def one_hot_rating(rating):
    if rating < 3:
        return 0
    return 1

df.rating = df.rating.map(one_hot_rating)

reviews_group = df.sort_values(by=["timestamp"]).groupby("user_id")


reviews_data = pd.DataFrame(
    data={
        "user_id": list(reviews_group.groups.keys()),
        "business_ids": list(reviews_group.business_id.apply(list)),
        "ratings": list(reviews_group.rating.apply(list)),
        "timestamps": list(reviews_group.timestamp.apply(list)),
        "photo_ids": list(reviews_group.photo_ids.apply(list)),
    }
)

sequence_length = 8
step_size = 2


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


reviews_data.business_ids = reviews_data.business_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

reviews_data.ratings = reviews_data.ratings.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

reviews_data.photo_ids = reviews_data.photo_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

del reviews_data["timestamps"]

reviews_data = reviews_data[reviews_data['ratings'].map(len) > step_size]

dct = {k: v for v, k in enumerate(reviews_data.user_id.unique())}
reviews_data.user_id = reviews_data.user_id.map(dct)

user_max = int(reviews_data.user_id.max())

businesses.columns = ['business_id', 'business_yelp_id', 'name', 'city', 'categories']

reviews_data_businesses = reviews_data[["user_id", "business_ids"]].explode(
    "business_ids", ignore_index=True
)
reviews_data_photos = reviews_data[["photo_ids"]].explode("photo_ids", ignore_index=True)
reviews_data_rating = reviews_data[["ratings"]].explode("ratings", ignore_index=True)

reviews_data_transformed = pd.concat([reviews_data_businesses, reviews_data_rating], axis=1)
reviews_data_transformed = pd.concat([reviews_data_transformed, reviews_data_photos], axis=1)
reviews_data_transformed = reviews_data_transformed.dropna()
reviews_data_transformed.business_ids = reviews_data_transformed.business_ids.apply(
    lambda x: ",".join(str(i) for i in x)
)

reviews_data_transformed.columns = ['user_id', 'sequence_business_ids', 'sequence_ratings', 'sequence_photo_ids']

train_percentage = 0.8

random_selection = np.random.rand(len(reviews_data_transformed.index)) <= train_percentage
train_data = reviews_data_transformed[random_selection]
test_data = reviews_data_transformed[~random_selection]

# business_max = int(businesses.business_id.max())


seq_num = len(reviews_data_transformed)
print(f'Sequences: {seq_num}')

del df
del reviews_data
del photos
del model
del reviews_group
del reviews_data_transformed
del businesses
del reviews

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
train_data.reset_index(inplace=True, drop=True)

class YelpDataset(data.Dataset):
    """Yelp dataset."""

    def __init__(
            self, ratings_file, test=False
    ):
        self.ratings_frame = ratings_file
        self.test = test

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        business_history = eval(data.sequence_business_ids)
        business_history_ratings = data.sequence_ratings
#         print([x.shape for x in data.sequence_photo_ids])
        photo_history = torch.stack(list(map(torch.from_numpy, data.sequence_photo_ids)))
#         photo_history = torch.stack(data.sequence_photo_ids)
        target_photo = photo_history[-1:][0]

        target_business_id = business_history[-1:][0]
        target_business_rating = business_history_ratings[-1:][0]

        photo_history = photo_history[:-1]
        business_history = torch.LongTensor(business_history[:-1])
        business_history_ratings = torch.LongTensor(business_history_ratings[:-1])

        return user_id, business_history, photo_history, target_business_id, business_history_ratings, target_business_rating, target_photo


batch_size = 64


class BST(pl.LightningModule):
    def __init__(
            self, learning_rate=0.002, args=None
    ):
        super().__init__()
        super(BST, self).__init__()
        self.learning_rate = learning_rate

        self.current_step = 0
        #         self.automatic_optimization = False

        self.save_hyperparameters()
        self.args = args
        # -------------------
        # Embedding layers
        ##Users
        self.embeddings_user_id = nn.Embedding(
            user_max + 1, int(math.sqrt(user_max)) + 1
        )

        ##Businesses
        self.embeddings_business_id = nn.Embedding(
            business_max + 1, 512
        )

        self.embeddings_positions = nn.Embedding(
            sequence_length, 512
        )

        #         self.positional_embedding = PositionalEmbedding(sequence_length, 512)
        #         self.embeddings_position = self.position_embedding(sequence_length - 1, 512)

        # Network
        self.transfomerlayer = nn.TransformerEncoderLayer(512, sequence_length, dropout=0.2)
        self.linear = nn.Sequential(
            nn.Linear(
                4692,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.criterion = torch.nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def encode_input(self, inputs):
        user_id, business_history, photo_history, target_business_id, business_history_ratings, target_business_rating, target_photo = inputs

        # Businesses
        business_history = self.embeddings_business_id(business_history)
        target_business = self.embeddings_business_id(target_business_id)
        target_business = torch.unsqueeze(target_business, 1)

        # Positions
        positions = torch.arange(0, sequence_length - 1, dtype=int, device=self.device)
        positions = self.embeddings_positions(positions)

        target_photo = torch.unsqueeze(target_photo, 1)

        encoded_sequence_businesses_with_position_and_rating = (positions + photo_history) * \
                                                               business_history_ratings[..., None]

        transformer_features_x = torch.cat((encoded_sequence_businesses_with_position_and_rating, target_business), 1)
        transfomer_features = torch.cat((transformer_features_x, target_photo), 1)

        # Users
        user_id = self.embeddings_user_id(user_id)
        user_features = user_id

        return transfomer_features, user_features, target_business_rating.float()

    def forward(self, batch):
        transfomer_features, user_features, target_business_rating = self.encode_input(batch)
        transformer_output = self.transfomerlayer(transfomer_features)
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        # Concat with other features
        features = torch.cat((transformer_output, user_features), dim=1)

        output = self.linear(features)
        return output, target_business_rating

    def training_step(self, batch, batch_idx):
        if self.opt is None:
            self.opt = self.optimizers()

        self.current_step += 1

        out, target_business_rating = self(batch)
        out = out.flatten()
        loss = self.criterion(out, target_business_rating)

        mae = self.mae(out, target_business_rating)
        mse = self.mse(out, target_business_rating)
        rmse = torch.sqrt(mse)
        self.log(
            "train/mae", mae, on_step=True, on_epoch=False, prog_bar=False
        )

        self.log(
            "train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=False
        )

        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        print(batch.shape)
        print(batch_idx)

    def validation_step(self, batch, batch_idx):
        out, target_business_rating = self(batch)
        out = out.flatten()
        loss = self.criterion(out, target_business_rating)

        mae = self.mae(out, target_business_rating)
        mse = self.mse(out, target_business_rating)
        rmse = torch.sqrt(mse)

        return {"val_loss": loss, "mae": mae.detach(), "rmse": rmse.detach()}

    #     def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, optimizer_closure=None, second_order_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
    #         self.opt.step(closure=optimizer_closure)
    #         self.opt.zero_grad()
    #         if self.trainer.global_step % self.config.val_check_interval == 0:
    #             self.reduce_lr_on_plateau.step(self.current_val_loss)

    def training_epoch_end(self, outputs):
        for param_group in self.opt.param_groups:
            print(f'Current learning rate: {param_group["lr"]}')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()

        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", avg_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=False)

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-6, cooldown=0,
                                                               patience=1)
            self.opt = optimizer
            return optimizer

    @staticmethod
    def position_embedding(sequence_length, embedding_dim, cuda=True):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(sequence_length, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.stack((torch.sin(emb), torch.cos(emb)), dim=0).view(
            sequence_length, -1).t().contiguous().view(sequence_length, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(sequence_length, 1)], dim=1)
        if cuda:
            return emb.cuda()
        return emb

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        print("Loading datasets")
        self.train_dataset = YelpDataset(train_data)
        self.val_dataset = YelpDataset(test_data)
        self.test_dataset = YelpDataset(test_data)
        print("Done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

trainer_gpu = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)
model_gpu = BST(learning_rate=0.00012)
print(model_gpu)
print(f'Learning rate: {model_gpu.hparams.learning_rate}')

start_time = time.time()
trainer_gpu.fit(model_gpu)
train_time = (time.time() - start_time)//60
tab_data = [[batch_size, photos_processed, seq_num, train_time]]
print(tabulate(tab_data, headers=["Batch","Photos", "Sequences", "Train time"]))
# print(f'Training time: {train_time} mins')
# print(f'Batch size: {batch_size}')
# print(f'Images: {photos_processed}; Sequences: {seq_num}')

trainer_gpu.test(model_gpu)