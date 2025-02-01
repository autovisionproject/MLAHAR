from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime,os
class summary_result():

  def __init__(self,t_acc,t_loss,v_acc,v_loss,test_loss, test_acc,indx,epoc,actual,predict,log_path,dsname):
    self.t_acc=t_acc
    self.t_loss=t_loss
    self.v_acc=v_acc
    self.v_loss=v_loss
    self.test_loss=test_loss
    self.test_acc=test_acc
    self.indx=indx
    self.epoc=epoc
    self.dsname=dsname
    self.actual=np.concatenate(actual)
    self.predict=np.concatenate(predict)
    self.label=set(self.actual)
    self.log_path=log_path
    current_datetime = datetime.datetime.now()
    self.formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

  def visualize_data(self):
    epochs = range(0, self.epoc)
    indx = "All" if self.indx==None else self.indx+1
    print("\n\n")
    # Plot and label the training and validation loss values
    plt.plot(epochs, self.t_loss, label='Training Loss')
    plt.plot(epochs, self.v_loss, label='Validation Loss')
    plt.plot(epochs, self.test_loss, label='Testing Loss')
    # Add in a title and axes labels
    plt.title('Training, Validation and Testing Loss : Device-{}'.format(indx))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.xticks(arange(0, 21, 2))
    plt.legend(loc='best')
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_path,f'{self.dsname}_loss_graph_{self.formatted_datetime}.PNG'), dpi=300)  # Save the figure with 300 dpi resolution
    plt.clf()

    print("\n\n")
    plt.plot(epochs, self.t_acc, label='Training Accuracy')
    plt.plot(epochs, self.v_acc, label='Validation Accuracy')
    plt.plot(epochs, self.test_acc, label='Testing Accuracy')
    plt.title('Training, Validation and Testing Accuracy: Device-{}'.format(indx))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #Set the tick locations
    #plt.xticks(arange(0, 21, 2))
    # Display the plot
    plt.legend(loc='best')
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_path,f'{self.dsname}_accuracy_graph_{self.formatted_datetime}.PNG'), dpi=300)  # Save the figure with 300 dpi resolution



  def cm_matrix(self):
    cm = confusion_matrix(self.actual,self.predict,labels=list(self.label))
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.label, yticklabels=self.label)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_path,f'{self.dsname}_confusion_Matrix_{self.formatted_datetime}.PNG'), dpi=300)  # Save the figure with 300 dpi resolution

    #plt.show()

  def report(self):
    classification_rep = classification_report(self.actual, self.predict, labels=list(self.label), output_dict=True)
    metrics_df = pd.DataFrame(classification_rep).transpose()
    metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']]
    metrics_df.index = ['Class ' + str(i) for i in metrics_df.index]
    print("Classification Metrics:")
    print(metrics_df)


  def accuracy_matrix(self):
    cm = confusion_matrix(self.actual,self.predict,labels=list(self.label))
    class_wise_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    annot_array = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        if i == j:
            annot_array[i, j] = f"{class_wise_accuracy[i]:.2f}"
        else:
            annot_array[i, j] = ""

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=annot_array, fmt="", cmap="Blues", xticklabels=self.label, yticklabels=self.label)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix with Class-wise Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_path,f'{self.dsname}_confusion_Class-wiseAccuracy_{self.formatted_datetime}.PNG'), dpi=300)  # Save the figure with 300 dpi resolution

    #plt.show()
