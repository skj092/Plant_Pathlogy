import pandas as pd 
from sklearn import model_selection

if __name__=="__main__":
    df = pd.read_csv('input/train.csv')
    df['label'] = df.apply(lambda x: " ".join([cat for cat in df.columns if x[cat] == 1]), axis=1)
    df = df.drop(['rust','scab',"healthy","multiple_diseases"], axis=1)
    target_incoding = {"rust":0,'scab':1, 'healthy':2,'multiple_diseases':3}
    df.label = df.label.map(target_incoding)
    df = df.sample(frac=0.1).reset_index(drop=True)
    y = df.label.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f 
    df.to_csv('input/train_fold.csv', index=False)