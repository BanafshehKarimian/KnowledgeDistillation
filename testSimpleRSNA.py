from torch.utils.data import DataLoader, Dataset
from Data.rsnaLoader import MammographyDataset
import torch
from Models.ResNet50Model import MammographyModel
from tqdm.auto import tqdm
from Trainers.simple import RSNATrainer
import polars as pl
import random
from torchvision import transforms

df_train = pl.read_csv('./Data/rsna-breast-cancer-detection/train.csv')
df_test = pl.read_csv('./Data/rsna-breast-cancer-detection/test.csv')
age_mean = round(df_train.get_column('age').mean())
age_min = df_train.get_column('age').min()
age_max = df_train.get_column('age').max()

df_train = \
    df_train.with_columns(
    
        pl.when(pl.col('age').is_null())\
        .then(age_mean)\
        .otherwise(pl.col('age'))\
        .alias('age')
    
    ).with_columns(
    
        ((pl.col('age') - age_min)/(age_max - age_min)).alias('age')
    
    ).to_dummies(
    
        ['view', 'laterality', 'implant']
    
    )
df_train = df_train\
    .with_columns(pl.lit('_').alias('underscore'))\
    .with_columns(
        pl.concat_str(
            [
                pl.col('patient_id'),
                pl.col('underscore'),
                pl.col('image_id')
            ]
        ).alias('fname')
    ).drop('underscore')

df_target1 = df_train.filter(pl.col('cancer') == 1)
df_target0 = df_train.filter(pl.col('cancer') == 0)
print("number of + and - cases:")
print(len(df_target1), len(df_target0))

n_idx0 = len(df_target0) # Number of noncancerous patients
balancing = True
if balancing:
    n = df_target1
else:
    n = df_target0
df_target0 = df_target0\
    .with_row_index()\
    .filter(pl.col('index').is_in(random.sample(range(n_idx0), len(n)))).drop('index')

n_idx1 = len(df_target1) # Number of noncancerous patients

#df_target1 = df_target1.with_row_index()

n = len(df_target0)
ntrain = round(n*.80)
df_target0 = df_target0.with_row_index().with_columns(
        pl.when(pl.col('index') <= ntrain)\
        .then(1)\
        .otherwise(0)\
        .alias('trainvalid')
    )\
    .drop('index')
df_target1 = df_target1.with_row_index().with_columns(
        pl.when(pl.col('index') <= ntrain)\
        .then(1)\
        .otherwise(0)\
        .alias('trainvalid')
    )\
    .drop('index')


# Final df
df_keep = \
    pl.concat([df_target1, df_target0], how='vertical')\
    .select(pl.all().shuffle(seed=19970507))





df_train_meta = df_keep.filter(pl.col('trainvalid') == 1)
df_valid_meta = df_keep.filter(pl.col('trainvalid') == 0)

train_fnames = df_train_meta.get_column('fname')
valid_fnames = df_valid_meta.get_column('fname')

train_labels = df_train_meta.get_column('cancer').to_numpy()
valid_labels = df_valid_meta.get_column('cancer').to_numpy()

# Defining the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initializing the datasets
train_dataset = MammographyDataset(
    meta_df=df_train_meta,
    img_dir='./Data/rsna-breast-cancer-detection/train_images',
    transform=transform,
)

valid_dataset = MammographyDataset(
    meta_df=df_valid_meta,
    img_dir='./Data/rsna-breast-cancer-detection/train_images',
    transform=transform,
)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=16)
valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=16)
device = "cuda" if torch.cuda.is_available() else "cpu"
learner_model = MammographyModel()
learner = RSNATrainer(learner_model, device)
num_epochs = 10
print("acc before training:")
print(learner.eval(valid_dataloader))
print(learner.train(num_epochs, train_dataloader))
print("acc after training:")
print(learner.eval(valid_dataloader))