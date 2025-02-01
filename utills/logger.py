import datetime
import pickle
import numpy as np
from utills.graph import summary_result
import os
def save_logs(model_edge,loss_acc_actual_pred,defaults,log_path,epoch,dsname,seed):
    os.makedirs(log_path, exist_ok=True)
    print(f"Directory '{log_path}'has been created.....")
    t_loss=loss_acc_actual_pred['train']['loss']
    t_acc=loss_acc_actual_pred['train']['acc']
    v_loss=loss_acc_actual_pred['val']['loss']
    v_acc=loss_acc_actual_pred['val']['acc']
    test_loss=loss_acc_actual_pred['test']['loss']
    test_acc=loss_acc_actual_pred['test']['acc']
    gt_labels=loss_acc_actual_pred['test']['actual']
    pred_labels=loss_acc_actual_pred['test']['predict']
    data=summary_result(t_acc,t_loss,v_acc,v_loss,test_loss, test_acc,None,epoch,gt_labels,pred_labels,log_path,dsname)
    print(data.report())
    data.accuracy_matrix()
    data.cm_matrix()
    data.visualize_data()
    
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    total_params_edge = sum(p.numel() for p in model_edge.parameters())
    model_file_GPU=os.path.join(log_path,f'{dsname}_Sensory_model_Linear_GPU_{round(test_acc[-1],3)}_{formatted_datetime}.pkl')
    model_file_CPU=os.path.join(log_path,f'{dsname}_Sensory_model_Linear_CPU_{round(test_acc[-1],3)}_{formatted_datetime}.pkl')
    stat_file=os.path.join(log_path,f'{dsname}_stats_Results_Linear_{round(test_acc[-1],3)}_{formatted_datetime}.txt')

    with open(model_file_GPU, 'wb') as file:
        pickle.dump(model_edge, file)     
    with open(model_file_CPU, 'wb') as file:
        pickle.dump(model_edge.to('cpu'), file)
        
    with open(stat_file, "w") as file:
        file.write("t_acc ## \n"+str(t_acc)+'\n\n')
        file.write("t_loss ## \n"+str(t_loss)+'\n\n')
        file.write("v_acc ## \n"+str(v_acc)+'\n\n')
        file.write("v_loss  ## \n"+str(v_loss)+'\n\n')
        file.write("test_acc  ##  \n"+str(test_acc)+'\n\n')
        file.write("test_loss  ##  \n"+str(test_loss)+'\n\n')
        file.write("gt_labels ##  \n"+str(list(np.concatenate(gt_labels).tolist()))+'\n\n')
        file.write("pred_labels  ##  \n"+str(list(np.concatenate(pred_labels).tolist()))+'\n\n')
        #file.write("gt_labels ##  \n"+str(list(np.concatenate(gt_labels)))+'\n\n')
        #file.write("pred_labels  ##  \n"+str(list(np.concatenate(pred_labels)))+'\n\n')
        hyper_paramter= str({'batch_size':defaults['batch_size'],'diam_internal':defaults['diam_internal'],'EPOCHS_edge':epoch,
                             'heads_edge':defaults['heads_edge'],'edge_drop':defaults['edge_drop'],
                             'lr_edge':defaults['lr_edge'],'weight_decay':defaults['weight_decay']})
        file.write("hyper_paramter ## "+hyper_paramter+'\n\n')
        file.write("window_size ## "+str(defaults['window_size'])+'\n\n')
        file.write("step_size  ## "+str(defaults['step_size'])+'\n\n')
        file.write("seed  ## "+str(seed)+'\n\n')

    print("############################################################################")
    for i in os.listdir(log_path)[-1:-6]:
        print(i)
    print("##############################################################################\n\n")
