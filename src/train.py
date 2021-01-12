import dataset
import model 
import engine
from torch.utils.data import Dataloader 



if "__name__" == '__main__':
    DIR_INPUT = '../input/'
    BATCH_SIZE = 64


    train_df = pd.read_csv(DIR_INPUT + '/train.csv')
    train_df['sample_type'] = 'train'

    sample_idx = train_df.sample(frac=0.2, random_state=42).index
    train_df.loc[sample_idx, 'sample_type'] = 'valid'

    valid_df = train_df[train_df['sample_type'] == 'valid']
    valid_df.reset_index(drop=True, inplace=True)

    train_df = train_df[train_df['sample_type'] == 'train']
    train_df.reset_index(drop=True, inplace=True)

    dataset_train = PlantDataset(df=train_df, transforms=transforms_train)
    dataset_valid = PlantDataset(df=valid_df, transforms=transforms_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    device = torch.device("cuda:0")

    model = PlantModel(num_classes=[1, 1, 1, 1])
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    plist = [{'params': model.parameters(), 'lr': 5e-5}]
    optimizer = torch.optim.Adam(plist, lr=5e-5)