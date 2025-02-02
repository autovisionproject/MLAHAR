import pandas as pd
import os
import numpy as np
import scipy.io
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer


class op_prepare_dataset:

  def __init__(self,path):
    self.path=path
    self.train_data = {}
    self.train_label = {}

  def get_data(self):
    extracted_col=[37,38,39,40,41,42,43,44,45,50,51,52,53,54,55,56,57,58,63,64,65,66,67,68,69,70,71,76,
                   77,78,79,80,81,82,83,84,89,90,91,92,93,94,95,96,97,102,103,104,105,106,107,108,109,110,111,112,113,
                   114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,249]
      
    label_mappeing={0:0}
    a=1
    with open(self.path[2],'r') as l:
        for i in l.readlines():
            if 'ML_Both_Arms'  in i:
                value=int(i.strip().split("-")[0].strip())
                label_mappeing[value] = a
                a+=1
    with open(self.path[1],'r') as i:
        header=[j.strip().split(' ')[2] for j in i.readlines() if "Column" in j]
        
    for i in os.listdir(self.path[0]):
         #if ("ADL" or 'Drill') in i:# in traget:
            file ='/'.join([self.path[0],i])
            df_filtered = pd.read_csv(file, sep=r'\s+', header=None)
        
            df_filtered.iloc[:,-1]=[label_mappeing[i] for i in df_filtered.iloc[:,-1].values]
            df_filtered=df_filtered.iloc[:,extracted_col]
            nan_check = df_filtered.notnull().all(axis=1)
            df_filtered = df_filtered[nan_check]
            label=df_filtered.iloc[:,-1]
            df_filtered=df_filtered.iloc[:, :-1].astype(float) / 1000
            df_filtered[249]=label
            df_filtered.reset_index(inplace=True, drop=True)        
            new_row = pd.DataFrame([[99 for var in range(len(df_filtered.columns))]], columns=df_filtered.columns)
            df_filtered = pd.concat([df_filtered, new_row], ignore_index=True)
            label=df_filtered.iloc[:,-1]
            n = len(df_filtered.columns)
            df_filtered.columns = range(n)
            if len(label.unique())> 2:    
                train_data = pd.DataFrame(columns=df_filtered.columns)
                df_dummy = []
                for j in range(len(label)-1):
                    if label[j] == label[j + 1]:
                        df_dummy.append(df_filtered.iloc[j])
                    else:
                        if len(df_dummy) < 1001 and label[j]==0:
                            df_dummy = []
                        elif len(df_dummy) > 1000 :
                            x_train = df_dummy[0:1000]
                            train_data =  pd.concat([train_data, pd.DataFrame(x_train)],axis=0, ignore_index=True)
                            df_dummy = []
                        else:
                            train_data =  pd.concat([train_data, pd.DataFrame(df_dummy)],axis=0, ignore_index=True)
                            df_dummy = []                                       
                                
                print(i,": ",train_data.shape,":  Label",train_data.iloc[:,-1].sort_values().astype(int).unique())
                self.train_label[i]=train_data.iloc[:,-1].astype(int)
                train_data.drop(train_data.columns[[-1]], axis=1, inplace=True)
                #self.train_data[i]= pd.DataFrame(scaler.fit_transform(train_data))
                self.train_data[i] = train_data


class pa_prepare_dataset:

  def __init__(self,path):
    self.path=path
    self.train_data = {}
    self.train_label = {}
    self.mapping={1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 12:7, 13:8, 16:9, 17:10, 24:11}

  def get_data(self):
    scaler = StandardScaler()
    for i in os.listdir(self.path):
        df = pd.read_csv('/'.join([self.path,i]), sep=' ',header=None)
        df = df[df.iloc[: , 1] != 0]
        df.reset_index(inplace=True, drop=True)
        df.iloc[:,1]=[self.mapping[i] for i in df.iloc[:,1].values]
        df = df.drop(df.columns[[0,2,3,16,17,18,19,20,33,34,35,36,37,50,51,52,53]], axis=1)
        df.interpolate(method='linear', axis=0, inplace=True)
        label = df[1]
        print(i,": ",df.shape,":  Label",label.sort_values().astype(int).unique())
        self.train_label[i.split('.')[0]]=label.astype(int)
        df.drop(df.columns[[1]], axis=1, inplace=True)   
        standardized_data = pd.DataFrame(scaler.fit_transform(df))
        self.train_data[i.split('.')[0]]=standardized_data
        
        
class rd_prepare_dataset:

  def __init__(self,path):
    self.path=path
    self.train_data = {}
    self.train_label = {}

  def get_data(self):
    scaler = StandardScaler()
    for i  in os.listdir(self.path):
        if 'ideal.log' in i:
            file ='/'.join([self.path,i])
            df=pd.read_csv(file, delim_whitespace=True, header=None)
            df= df.iloc[:,2:]
            df = df[df.iloc[: , -1] != 0]
            df.reset_index(inplace=True, drop=True)
            new_row = pd.DataFrame([[99 for var in range(len(df.columns))],], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            for b in df.columns:
                if len(df[b].unique()) <4:
                    print(b,len(df[b].unique()))
            label=df.iloc[:,-1]
            count=sorted(list(df.iloc[:,-1].unique()))
            if len(count)>10:
                train_data = pd.DataFrame(columns=df.columns)
                df_dummy = []
                for j in range(len(label)-1):
                    if label[j] == label[j + 1]:
                        df_dummy.append(df.iloc[j])
                    else:   
                       train_data =  pd.concat([train_data, pd.DataFrame(df_dummy)],axis=0, ignore_index=True)
                       df_dummy = []
                            
                label=train_data.iloc[:,-1]-1
                print(i.split('_')[0],": ",train_data.shape,":  Label",label.sort_values().astype(int).unique())
                train_data=train_data.iloc[:,:-1]
                self.train_label[i.split('_')[0]]=label.astype(int)   
                self.train_data[i.split('_')[0]]=pd.DataFrame(scaler.fit_transform(train_data))
                self.train_data[i.split('_')[0]]=train_data

class mh_prepare_dataset:
    
  def __init__(self,path):
    self.path=path
    self.train_data = {}
    self.train_label = {}

  def get_data(self):
    for i in os.listdir(self.path):
      file='/'.join([self.path,i])
      df = pd.read_csv(file, delim_whitespace=True, header=None)
      df = df[df.iloc[: , -1] != 0]
      df.reset_index(inplace=True, drop=True) 
      for b in df.columns:
        if len(df[b].unique()) <3:
            print(b,len(df[b].unique()))
      label=df.iloc[: , -1]-1
      self.train_label[i.split('_')[1].split('.')[0]]=label
      df.drop(df.columns[[-1]], axis=1, inplace=True)
      self.train_data[i.split('_')[1].split('.')[0]]= df
      print(i.split('_')[1].split('.')[0],": ",df.shape,": ","Label",label.sort_values().astype(int).unique()) 

def print_details(x_train,y_train,x_val,y_val,x_test,y_test):
    print("\nDataset details for training, testing, and validation ########################\n")
    for i in range(1):
      print("Train size: {}  Train Label: {} \nVal size:   {}  Val Label : {} \nTest size:  {}  Text Label : {}".format(x_train.shape,y_train.shape,
                                                                                                                        x_val.shape,y_val.shape,
                                                                                                                       x_test.shape,y_test.shape))
    #This code snippet is used to verify the count of samples for each class used in training, testing, and validation.
    print("\nclasswise distribution for training set .....")
    print("{}:{} ".format('Train Dataset',list(y_train.value_counts().sort_index().apply(int).items())))
    print("\nVal label classwise distribution")
    print("{}:{} ".format('Val Dataset',list(y_val.value_counts().sort_index().apply(int).items())))
    print("\nTest label classwise distribution")
    print("{}:{} ".format('Val Dataset',list(y_test.value_counts().sort_index().apply(int).items())))
    print("############################################################################################\n")



class SlidingWindowDataset(Dataset):
    def __init__(self, features, labels, window_size, step_size):
        features = features.values if isinstance(features, pd.DataFrame) else features
        labels = labels.values if isinstance(labels, (pd.DataFrame, pd.Series)) else labels

        self.features = self._create_sliding_window(features, window_size, step_size)
        self.labels = self._create_label_windows(labels, window_size, step_size)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def _create_sliding_window(data, window_size, step_size):
        shape = ((data.shape[0] - window_size) // step_size + 1, window_size, data.shape[1])
        strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).astype(np.float32)

    @staticmethod
    def _create_label_windows(labels, window_size, step_size):
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        labels = np.asarray(labels)
        return np.lib.stride_tricks.as_strided(labels,
                                               shape=((labels.shape[0] - window_size) // step_size + 1, window_size),
                                               strides=(labels.strides[0] * step_size, labels.strides[0]))[:, -1].astype(np.uint8)

def split_dataset(df):
    x1 = pd.DataFrame(columns=df.columns)
    x2 = pd.DataFrame(columns=df.columns)
    df =  pd.concat([df, pd.DataFrame([[99 for i in range(len(df.columns))]],columns=df.columns)], axis=0, ignore_index=True)
    label=df['label']
    df_dummy = []
    for i in range(len(label)-1):
        if label[i] == label[i + 1]:
            df_dummy.append(df.iloc[i])
        else:
            train_size = int(len(df_dummy)/2)
            x = df_dummy[0:train_size]
            y = df_dummy[train_size:]
            x1 =  pd.concat([x1, pd.DataFrame(x)],axis=0, ignore_index=True)
            x2 =  pd.concat([x2 , pd.DataFrame(y)],axis=0, ignore_index=True)
            df_dummy = []
    print("Output shape file1",x1.shape ,'Output shape file2',x2.shape)

    return x1,x2


