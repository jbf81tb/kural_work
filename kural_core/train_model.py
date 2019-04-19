import time
import math
import torch
import numpy as np
from IPython.display import clear_output
def train_model(model, train_dl, val_dl, opt, criterion, lr_start, lr_end=None, epochs=3, cycles=1, restarts=1, clip=50, print_epoch=True, is_classification=False, lr_cycle='sin'):
    # try:
        train_loss = []
        val_loss = []
        if is_classification: accuracy = []
        if lr_end is None: lr_end = lr_start
        for restart in range(restarts):
            # print(f'restart number {restart+1} out of {restarts}')
        #     if len(train_loss)>0: opt = optim.SGD(model.parameters(),lr=lr_start,momentum=0.9)
            time_list = []
            epoch_time_list = None
            cycle_print_str = 'Cycle: 0 ~ Remaining total: Unknown'
            # print(cycle_print_str)
            for cycle in range(cycles):
                ts = time.perf_counter()
        #         if cycle>0: epochs = epochs*2
                lr_start = lr_start*(0.99)
                epoch_print_str = f' ~ Epoch: 0 ~ Remaining in cycle: {(epochs*np.mean(epoch_time_list) if epoch_time_list is not None else 0):.3g}s'
                clear_output(wait=True)
                print(cycle_print_str + epoch_print_str)
                epoch_time_list = []
                for epoch in range(epochs):
                    ets = time.perf_counter()
                    if lr_cycle == 'cos': opt.param_groups[0]['lr'] = lr_end + 0.5*(lr_start-lr_end)*(1+math.cos(cycle/cycles*np.pi))
                    if lr_cycle == 'sin': opt.param_groups[0]['lr'] = lr_start + (lr_end-lr_start)*(math.sin(cycle/cycles*np.pi))
                    running_loss = 0.0
                    model.train()
                    for x_train, y_train in train_dl:
                        x_train, y_train = (a.cuda() for a in [x_train, y_train])
                        opt.zero_grad()
                        y_pred = model(x_train)
                        loss = criterion(y_pred, y_train)
                        loss.backward()
                        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                        opt.step()
                        running_loss += loss.item()
                    train_loss.append(running_loss/len(train_dl))

                    running_loss = 0.0
                    acc_list = []
                    model.eval()
                    with torch.no_grad():
                        for x_val, y_val in val_dl:
                            x_val, y_val = (a.cuda() for a in [x_val, y_val])
                            y_pred = model(x_val)
                            running_loss += criterion(y_pred, y_val).item()
                            if is_classification:
                                y_pred = y_pred.argmax(1)
                                acc_list.append((torch.sum(y_pred == y_val).float()/y_val.shape[0]).data.cpu().numpy())
                        val_loss.append(running_loss/len(val_dl))
                        if is_classification: accuracy.append(np.stack(acc_list).mean())
                    epoch_time_list.append(time.perf_counter()-ets)
                    if print_epoch:
                        epoch_time = (epochs-(epoch+1))*np.mean(epoch_time_list)
                        epoch_print_str = f' ~ Epoch: {epoch+1:2d}/{epochs} ~ Remaining in cycle: {int(epoch_time//3600)}h{int(epoch_time//60)-60*int(epoch_time//3600)}m{epoch_time%60:02.0f}s'
                        clear_output(wait=True)
                        print(cycle_print_str + epoch_print_str)
                time_list.append(time.perf_counter()-ts)
                if cycle<cycles-1:
                    remaining_time = (cycles-(cycle+1)-1)*np.mean(time_list)
                    cycle_print_str = f'Finished cycle {cycle+1:3d}/{cycles} ~ Remaining after current cycle: {int(remaining_time//3600)}h{int(remaining_time//60)-60*int(remaining_time//3600)}m{remaining_time%60:02.0f}s'
                    clear_output(wait=True)
                    print(cycle_print_str+epoch_print_str)
            clear_output(wait=True)
            final_print_str = f'Spent {int(sum(time_list)//60)}m {sum(time_list)%60:2.0f}s doing {cycles*epochs} total steps for an average of {sum(time_list)/cycles/epochs:3.1f}s per step.'
            lfps = len(final_print_str)
            print('><'*(lfps//2+lfps%2))
            print(final_print_str)
            print('><'*(lfps//2+lfps%2))
    # except Exception as e:
    #     print('')
    #     print(type(e))
    #     print(e)
    # finally:
        if is_classification:
            return (model, train_loss, val_loss, accuracy)
        else:
            return (model, train_loss, val_loss)

    # torch.save(model.state_dict(),'C:\\Users\\joshu\\Documents\\fastai\\courses\\kural_work\\kMeans_autoencoder_model.pth')
