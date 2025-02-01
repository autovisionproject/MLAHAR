
import torch
from sklearn.metrics import accuracy_score
from statistics import mean
import torch.nn as nn

#device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
device = f"cuda:1" if torch.cuda.is_available() else "cpu"

print(f"{device}" " is available.")

def train(model,train_dataloader, val_dataloader, test_dataloader, EPOCHS, learning_rate, weight_decay):

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.99),weight_decay=model.weight_decay)
 # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=weight_decay, patience=5)
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

  crossEntropyLoss=nn.CrossEntropyLoss()
  mse=nn.MSELoss()
  softmax = nn.Softmax(dim=1)
  loss_acc_actual_pred = {"train":{"loss":[],"acc":[],"actual":[],"predict":{}},
                          "val":{"loss":[],"acc":[],"actual":[],"predict":{}},
                          "test":{"loss":[],"acc":[],"actual":[],"predict":{}}}

  for epoch in range(1,EPOCHS+1):

    t_acc,t_loss=[],[]
    model=model.to(device)
    model.update_epoch(epoch)
    model.train()

    for i,(train_x, train_y) in enumerate(train_dataloader):
      train_x=train_x.float().to(device)
      train_y=train_y.long().to(device)

      # Model Forward pass
      yhat  = model(train_x,True)
      loss = crossEntropyLoss(yhat,train_y)
      #focal_loss =  1 * (1 - torch.exp(-ce_loss)) ** 2 * ce_loss
      #loss=focal_loss.mean()
      yhat = softmax(yhat)
      t_loss.append(loss.item())

      #l2_reg = torch.tensor(0., device=device)
      #for param in model.parameters():
       # l2_reg += torch.norm(param)
       # loss += model.weight_decay * l2_reg

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      yhat = yhat.detach().argmax(1)
      train_y = train_y.detach()
      t_acc.append(accuracy_score(train_y.cpu().numpy() ,yhat.cpu().numpy()))

    loss_acc_actual_pred["train"]["loss"] += [mean(t_loss)]
    loss_acc_actual_pred["train"]["acc"] += [mean(t_acc)]

    v_loss,v_acc,val_pre,val_act= model_eval(model,val_dataloader,EPOCHS,crossEntropyLoss,softmax)
    loss_acc_actual_pred["val"]["loss"] +=[v_loss]
    loss_acc_actual_pred["val"]["acc"] += [v_acc]

    test_loss,test_acc,test_pre,test_act= model_eval(model,test_dataloader,EPOCHS,crossEntropyLoss,softmax)
    loss_acc_actual_pred["test"]["loss"]+=[test_loss]
    loss_acc_actual_pred["test"]["acc"]+=[test_acc]

    if epoch==EPOCHS:
      loss_acc_actual_pred["test"]["predict"] = test_pre
      loss_acc_actual_pred["test"]["actual"] =  test_act
      loss_acc_actual_pred["val"]["actual"]  =  val_pre
      loss_acc_actual_pred["val"]["predict"] = val_act

    scheduler.step(v_loss)
    print("Epoch {:2d}/{:2d} Done, Train Loss: {:5f}  Train acc: {:5f}  Val Loss: {:5f}  Val Acc: {:5f}  Test Loss: {:5f} Test Acc: {:5f}\
    ".format(epoch,EPOCHS,mean(t_loss),mean(t_acc),v_loss,v_acc,test_loss,test_acc))

  return model,loss_acc_actual_pred


def model_eval(model, val_dataloader, EPOCHS, crossEntropyLoss, softmax):
    model.eval()
    with torch.no_grad():
        val_acc,val_loss=[],[]
        val_pre,val_act=[],[]
        for i, (val_x,val_y) in enumerate(val_dataloader):
          val_x = val_x.float().to(device)
          val_y = val_y.long().to(device)
          yhat = model(val_x)
          loss = crossEntropyLoss(yhat,val_y)
          yhat = softmax(yhat).squeeze(0)
          val_loss.append(loss.item())
          yhat=yhat.argmax(1)
          val_acc.append(accuracy_score(val_y.cpu().numpy(),yhat.cpu().numpy()))
          val_pre.append(yhat.cpu().numpy())
          val_act.append(val_y.cpu().numpy())
        return mean(val_loss),mean(val_acc),val_pre,val_act
