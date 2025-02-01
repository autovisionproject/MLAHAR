import yaml
import argparse,os
from utills.prepareDataset import *
import pandas as pd
import torch, random
import numpy as np
import scipy.io
import torch.nn as nn
from models.train import train, model_eval
from utills.logger import save_logs
from model_linear import MLA

def set_seed(seed):
    #seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_config(dsname):
    """Load default values for a specific dataset from YAML."""
    if os.path.isfile('config.yaml'):
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            datasets = config.get('datasets', {})
            if dsname in datasets:
                return datasets[dsname]
            else:
                raise ValueError(f"Dataset '{dsname}' not found in configuration file.")
    else:
        raise FileNotFoundError(f"YAML configuration file config.yaml not found")


def main(dsname,epoch,seed):
    defaults = load_config(dsname)
    if epoch==False:
        epoch=defaults['epoch']
        
    print(f"\nLoading dataset {dsname}.............")
    if dsname=='opportunity':
        dataset=op_prepare_dataset([defaults['path'],defaults['col'],defaults['lbl']])
        t_file=['S1-ADL1.dat', 'S2-ADL1.dat', 'S3-ADL1.dat', 'S4-ADL1.dat', 'S1-Drill.dat', 'S2-ADL2.dat', 'S3-ADL2.dat', 'S4-ADL2.dat', 'S2-Drill.dat', 'S1-ADL4.dat', 'S1-ADL3.dat', 'S2-ADL3.dat', 'S3-ADL3.dat', 'S4-ADL3.dat', 'S3-Drill.dat', 'S3-ADL4.dat', 'S1-ADL5.dat', 'S2-ADL5.dat', 'S4-ADL5.dat', 'S4-Drill.dat']
        v_file = ['S1-ADL2.dat', 'S4-ADL4.dat']
        test_file=['S2-ADL4.dat', 'S3-ADL5.dat']

    elif dsname=='pamap2':
        dataset=pa_prepare_dataset(defaults['path'])
        t_file=['subject105', 'subject103', 'subject101', 'subject102', 'subject108', 'subject109']
        v_file=['subject104']
        test_file=['subject107', 'subject106']

    elif dsname=='mhealth':
        dataset=mh_prepare_dataset(defaults['path'])
        t_file=['subject10','subject1','subject2','subject3','subject7','subject4','subject5']
        v_file=['subject8']
        test_file=['subject9','subject6']
                
    elif dsname=='realdisp':
        dataset=rd_prepare_dataset(defaults['path'])
        t_file=['subject1', 'subject4', 'subject16', 'subject9', 'subject17', 'subject6', 'subject8', 'subject15', 'subject14', 'subject5', 'subject13', 'subject3']
        test_file=['subject11', 'subject12']
        v_file=['subject10', 'subject2']

    elif dsname=='skoda':
        dataset=sd_prepare_dataset(defaults['path'])
        dataset.get_data(defaults['split_ratio'])
        x_train=dataset.train_data
        y_train=dataset.train_label
        x_test=dataset.test_data
        y_test=dataset.test_label
        x_val=dataset.val_data
        y_val=dataset.val_label
        x_train['mean']= x_train.mean(axis=1)  
        x_train['std']= x_train.std(axis=1) 
        x_test['mean']= x_test.mean(axis=1)  
        x_test['std']= x_test.std(axis=1) 
        x_val['mean']= x_val.mean(axis=1)  
        x_val['std']= x_val.std(axis=1) 

    elif dsname=='gotov':
        dataset=go_prepare_dataset(defaults['path'])
        dataset.get_data()
        t_file=['GOTOV21','GOTOV11','GOTOV08','GOTOV33','GOTOV28','GOTOV7','GOTOV05','GOTOV10','GOTOV31']
        test_file = ['GOTOV20','GOTOV29','GOTOV16']
        v_file =['GOTOV17','GOTOV35', 'GOTOV06']

    if dsname!='skoda':
        dataset.get_data()
        print(f"\nDataset loading finished : {dsname} ....................................")
        train_x=dataset.train_data
        train_y=dataset.train_label
        x_train = pd.concat([train_x[i] for i,j in train_x.items() if i  in t_file],axis=0,ignore_index=True)
        y_train = pd.concat([train_y[i] for i,j in train_x.items() if i  in t_file],axis=0,ignore_index=True)
        x_test = pd.concat([train_x[i] for i,j in train_x.items() if i in test_file],axis=0,ignore_index=True)
        y_test = pd.concat([train_y[i] for i,j in train_y.items() if i in test_file],axis=0,ignore_index=True)
        x_val =  pd.concat([train_x[i] for i,j in train_x.items() if i in v_file],axis=0,ignore_index=True)
        y_val =  pd.concat([train_y[i] for i,j in train_y.items() if i in v_file],axis=0,ignore_index=True)
        x_train['mean']= x_train.mean(axis=1)  
        x_train['std']= x_train.std(axis=1) 
        x_test['mean']= x_test.mean(axis=1)  
        x_test['std']= x_test.std(axis=1) 
        x_val['mean']= x_val.mean(axis=1)  
        x_val['std']= x_val.std(axis=1) 

    print_details(x_train,y_train,x_val,y_val,x_test,y_test)
    dataset_new = SlidingWindowDataset(x_train, y_train, window_size=defaults['window_size'], step_size=defaults['step_size'])
    
    if dsname=='skoda':
        train_loader = DataLoader(dataset_new, batch_size=defaults['batch_size'], shuffle=True)
    else:
        train_loader = DataLoader(dataset_new, batch_size=defaults['batch_size'], shuffle=False)
    dataset_new = SlidingWindowDataset(x_val, y_val, window_size=defaults['window_size'], step_size=defaults['step_size'])
    val_loader = DataLoader(dataset_new, batch_size=defaults['batch_size'], shuffle=False)
    dataset_new = SlidingWindowDataset(x_test, y_test, window_size=defaults['window_size'], step_size=defaults['step_size'])
    test_loader = DataLoader(dataset_new, batch_size=defaults['batch_size'], shuffle=False)
    output=len(np.unique(y_train))
    parms_diam=x_train.shape[1]
    print("\n................................... Model Training started ....................................\n")
    model_edge = MLA(output, parms_diam, float(defaults['edge_drop']), defaults['heads_edge'], defaults['batch_size'],defaults['diam_internal'],float(defaults['weight_decay']),epoch,defaults['window_size'])
    model_edge,loss_acc_actual_pred = train(model_edge,train_loader,val_loader,test_loader,epoch,float(defaults['lr_edge']),float(defaults['weight_decay']))

    print("\n\n########################### Save Logs ##################################\n")
    save_logs(model_edge,loss_acc_actual_pred,defaults,defaults['log_path'],epoch,dsname,seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset based on the dataset name.")
    parser.add_argument("--dsname", type=str, help="Dataset name (choose from 'gotov', 'opportunity', 'realdisp', 'skoda', 'pamap2', 'mhealth')")
    parser.add_argument("--epoch", type=int, default=False, help=" Select epoch range (choose from 0-N)")
    parser.add_argument("--seed", type=int, default=42, help=" Set Seed value [42,56, etc]")
    args = parser.parse_args()
    seed=args.seed
    set_seed(seed)
    dsname=args.dsname
    epoch=args.epoch
    main(dsname,epoch,seed)

