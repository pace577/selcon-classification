import copy
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from utils.custom_dataset import load_std_regress_data, CustomDataset, load_dataset_custom
from utils.Create_Slices import get_slices
from model.classification import LogisticRegression
from model.SELCON import FindSubset_Vect
from model.facility_location import run_stochastic_Facloc

from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)

class Classification():
    def __init__(self, num_cls, criterion):
        self.num_cls = num_cls
        self.criterion = criterion
        self.select_every = 35
        self.reg_lambda = 1e-5
        self.val_loss = 0
        self.test_loss = 0.
        self.test_loss_std = 0
        self.batch_size = 4000
        self.learning_rate = 0.01
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.psuedo_length = 1.0
        self.subset_idx = None
        
    def weight_reset(self, m):
        '''
        Fills the input tensor using Glorot Initialisation and

        '''
        torch.manual_seed(42)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)
        
    def train_model(self, x_trn, y_trn, x_val, y_val, fraction,  delt = [0.3], x_tst = None, y_tst = None,\
                         num_epochs=2000, default = False, bud=None, fair=True):
        """Trains model. If fair=True then algorithm uses validation error as
        constraint, otherwise it will not."""

        sub_epoch = 3
        N, M = x_trn.shape
        bud = int(fraction * N)
        print("Budget, fraction and N:", bud, fraction, N)
        train_batch_size = min(bud,1000)
        print_every = 50

        deltas = torch.tensor(delt).to(self.device) 
        
        # initialise index
        rand_idxs = list(np.random.choice(N, size=bud, replace=False))
        sub_idxs = rand_idxs

        criterion = self.criterion()

        # Initialize model
        main_model = LogisticRegression(M, self.num_cls)
        main_model.apply(self.weight_reset)

        main_model = main_model.to(self.device)

        #criterion_sum = nn.MSELoss(reduction='sum')
        main_optimizer = torch.optim.Adam(main_model.parameters(), lr=self.learning_rate)
        #[{'params': main_model.parameters()}], lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(main_optimizer, step_size=1, gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(main_optimizer, milestones=change,\
        #     gamma=0.5) #[e*2 for e in change]
        cached_state_dict = copy.deepcopy(main_model.state_dict())

        if fair:
            alphas = torch.randn_like(deltas,device=self.device) #+ 5. #,requires_grad=True)
            alphas.requires_grad = True
            #print(alphas)
            #alphas = torch.ones_like(deltas,requires_grad=True)
            '''main_optimizer = optim.SGD([{'params': main_model.parameters()},
                        {'params': alphas}], lr=learning_rate) #'''

            dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.learning_rate) #{'params': alphas} #'''
            #delta_extend = torch.repeat_interleave(deltas,val_size, dim=0)
            #alphas.requires_grad = False
            alpha_orig = copy.deepcopy(alphas)


        if self.psuedo_length == 1.0:
            sub_rand_idxs = [s for s in range(N)]
            current_idxs = sub_idxs
        else:
            sub_rand_idxs = [s for s in range(N)]
            new_ele = set(sub_rand_idxs).difference(set(sub_idxs))
            sub_rand_idxs = list(np.random.choice(list(new_ele), size=int(self.psuedo_length*N), replace=False))

            sub_rand_idxs = sub_idxs + sub_rand_idxs

            current_idxs = [s for s in range(len(sub_idxs))]

        fsubset_d = FindSubset_Vect(x_trn[sub_rand_idxs], y_trn[sub_rand_idxs], x_val, y_val,main_model,\
                                    self.criterion,self.device,deltas,self.learning_rate,self.reg_lambda,self.batch_size, fair=fair)
        if fair:
            fsubset_d.precompute(int(num_epochs/4),sub_epoch,alpha_orig)
        else:
            fsubset_d.precompute(int(num_epochs/4),sub_epoch,torch.randn_like(deltas,device=self.device))


        main_model.load_state_dict(cached_state_dict)

        print("Starting Subset of size ",fraction," with fairness Run!")

        sub_idxs.sort()
        np.random.seed(42)
        np_sub_idxs = np.array(sub_idxs)
        np.random.shuffle(np_sub_idxs)
        loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
                transform=None),shuffle=False,batch_size=train_batch_size)

        loader_val = DataLoader(CustomDataset(x_val, y_val,transform=None),shuffle=False,\
            batch_size=self.batch_size)


        #for i in range(num_epochs):
        stop_count = 0
        prev_loss = 1000
        prev_loss2 = 1000
        i =0
        mul = 1
        lr_count = 0
        #while (True):

        ### Line 5 is here or inside the "FindSubset" function?
        for i in range(num_epochs):

            # inputs, targets = x_trn[sub_idxs].to(self.device), y_trn[idxs].to(self.device)
            #inputs, targets = x_trn[sub_idxs], y_trn[sub_idxs]

            temp_loss = 0.

            starting = time.process_time() 

            for batch_idx_t in list(loader_tr.batch_sampler):

                inputs_trn, targets_trn = loader_tr.dataset[batch_idx_t]
                inputs_trn, targets_trn = inputs_trn.to(self.device), targets_trn.to(self.device)

                main_optimizer.zero_grad()

                scores_trn = main_model(inputs_trn)

                l2_reg = 0
                for param in main_model.parameters():
                    l2_reg += torch.norm(param)

                if fair:
                    '''l = [torch.flatten(p) for p in main_model.parameters()]
                    flat = torch.cat(l)
                    l2_reg = torch.sum(flat*flat)'''

                    #state_orig = copy.deepcopy(main_optimizer.state)

                    '''alpha_extend = torch.repeat_interleave(alphas,val_size, dim=0)
                    val_scores = main_model(x_val_combined)
                    constraint = criterion(val_scores, y_val_combined) - delta_extend
                    multiplier = torch.dot(alpha_extend,constraint)'''

                    '''constraint = torch.zeros(len(x_val_list))
                    for j in range(len(x_val_list)):

                        inputs_j, targets_j = x_val_list[j], y_val_list[j]
                        scores_j = main_model(inputs_j)
                        constraint[j] = criterion(scores_j, targets_j) - deltas[j]'''

                    constraint = 0.
                    for batch_idx in list(loader_val.batch_sampler):

                        inputs, targets = loader_val.dataset[batch_idx]
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                        val_out = main_model(inputs)
                        '''if is_time:
                            val_out = sc_trans.inverse_transform(val_out.cpu().numpy())
                            val_out = torch.from_numpy(val_out).float()'''
                        constraint += criterion(val_out, targets)

                    constraint /= len(loader_val.batch_sampler)
                    constraint = constraint - deltas
                    multiplier = alphas*constraint*(float(constraint > 0)) #torch.dot(alphas,constraint)

                    loss = criterion(scores_trn, targets_trn) + self.reg_lambda*l2_reg*len(batch_idx_t) + \
                        multiplier #
                else:
                    loss = criterion(scores_trn, targets_trn) +  self.reg_lambda*l2_reg*len(batch_idx_t)
                temp_loss += loss.item()
                loss.backward()

                #if i % print_every == 0:  
                #    print(criterion(scores_trn, targets_trn) , reg_lambda*l2_reg*len(batch_idx_t) ,multiplier)

                # clamp gradients, just in case
                for p in filter(lambda p: p.grad is not None, main_model.parameters()):\
                     p.grad.data.clamp_(min=-.1, max=.1)

                main_optimizer.step()
                #scheduler.step()
                #main_optimizer.param_groups[1]['lr'] = learning_rate/2

                if fair:
                    '''for param in main_model.parameters():
                        param.requires_grad = False
                    alphas.requires_grad = True'''

                    dual_optimizer.zero_grad()

                    #if constraint > 0:
                    constraint = 0.
                    for batch_idx in list(loader_val.batch_sampler):

                        inputs, targets = loader_val.dataset[batch_idx]
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                        val_out = main_model(inputs)
                        '''if is_time:
                            val_out = sc_trans.inverse_transform(val_out.cpu().numpy())
                            val_out = torch.from_numpy(val_out).float()'''
                        constraint += criterion(val_out, targets)

                    constraint /= len(loader_val.batch_sampler)
                    constraint = constraint - deltas
                    multiplier = -1.0*alphas*constraint*(float(constraint > 0)) #torch.dot(-1.0*alphas ,constraint)

                    #print(alphas,constraint)

                    #main_optimizer.state = state_orig
                    multiplier.backward()
                    dual_optimizer.step()
                    #print(main_optimizer.param_groups)
                    #scheduler.step()`

                    alphas.requires_grad = False
                    alphas.clamp_(min=0.0)
                    alphas.requires_grad = True
                    #print(alphas)

                    '''for param in main_model.parameters():
                        param.requires_grad = True'''

                    #print(alphas,constraint)

            if i % print_every == 0:  # Print Training and Validation Loss
                print('Epoch:', i + 1, 'SubsetTrn', loss.item())
                print("Previous loss: ", prev_loss, "\n"\
                    ,"Temporary loss: ", temp_loss, "\n"\
                    ,"Mul: ", mul)
                #print(main_optimizer.state)#.keys())
                #print(main_optimizer.param_groups)#[0]['lr'])
                if fair:
                    print("Constraint: ", constraint, "\n"\
                        , "Alphas: ", alphas)
                    #print(alphas,constraint)
                    #print(criterion(scores, targets) , reg_lambda*l2_reg*len(idxs) ,multiplier)


            if ((i + 1) % self.select_every == 0):

                cached_state_dict = copy.deepcopy(main_model.state_dict())
                clone_dict = copy.deepcopy(cached_state_dict)

                if fair:
                    alpha_orig = alphas.detach().clone()#copy.deepcopy(alphas)

                    '''alpha_orig.requires_grad = False
                    alpha_orig = alpha_orig*((constraint >0).float())
                    alpha_orig.requires_grad = True'''


                fsubset_d.lr = main_optimizer.param_groups[0]['lr']*mul#,1e-4)

                state_values = list(main_optimizer.state.values())
                step = 0#state_values[0]['step']

                w_exp_avg = torch.zeros(x_trn.shape[1]+1,device=self.device)
                #torch.cat((state_values[0]['exp_avg'].view(-1),state_values[1]['exp_avg']))
                w_exp_avg_sq = torch.zeros(x_trn.shape[1]+1,device=self.device)
                #torch.cat((state_values[0]['exp_avg_sq'].view(-1),state_values[1]['exp_avg_sq']))

                if fair:
                    state_values = list(dual_optimizer.state.values())

                    a_exp_avg = torch.zeros(1,device=self.device)
                    #state_values[0]['exp_avg']
                    a_exp_avg_sq = torch.zeros(1,device=self.device)
                    #state_values[0]['exp_avg_sq']
                    #print(exp_avg,exp_avg_sq)
                    d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,current_idxs,alpha_orig,bud,\
                        train_batch_size,step,w_exp_avg,w_exp_avg_sq,a_exp_avg,a_exp_avg_sq)#,main_optimizer,dual_optimizer)
                else:
                    # d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,current_idxs,\
                    #     bud,train_batch_size,step,w_exp_avg,w_exp_avg_sq)
                    d_sub_idxs = fsubset_d.return_subset(clone_dict,sub_epoch,current_idxs,None,bud,\
                                                         train_batch_size,step,w_exp_avg,w_exp_avg_sq,None,None)#,main_optimizer,dual_optimizer)

                '''clone_dict = copy.deepcopy(cached_state_dict)
                alpha_orig = copy.deepcopy(alphas)

                sub_idxs = fsubset.return_subset(clone_dict,sub_epoch,sub_idxs,alpha_orig,bud,\
                    train_batch_size)
                print(sub_idxs[:10])'''

                current_idxs = d_sub_idxs

                d_sub_idxs = list(np.array(sub_rand_idxs)[d_sub_idxs])

                new_ele = set(d_sub_idxs).difference(set(sub_idxs))
                #print(len(new_ele),0.1*bud)

                if len(new_ele) > 0.1*bud:
                    main_optimizer = torch.optim.Adam([
                    {'params': main_model.parameters()}], lr=main_optimizer.param_groups[0]['lr'])
                    # {'params': main_model.parameters()}], lr=main_optimizer.param_groups[0]['lr']*mul)
                    #max(main_optimizer.param_groups[0]['lr'],0.001))

                    if fair:
                        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.learning_rate)

                    #mul=1
                    stop_count = 0
                    lr_count = 0

                sub_idxs = d_sub_idxs

                sub_idxs.sort()

                print("First 10 subset indices: ", sub_idxs[:10])
                np.random.seed(42)
                np_sub_idxs = np.array(sub_idxs)
                np.random.shuffle(np_sub_idxs)
                loader_tr = DataLoader(CustomDataset(x_trn[np_sub_idxs], y_trn[np_sub_idxs],\
                        transform=None),shuffle=False,batch_size=train_batch_size)

                main_model.load_state_dict(cached_state_dict)

            if abs(prev_loss - temp_loss) <= 1e-1*mul or abs(temp_loss - prev_loss2) <= 1e-1*mul:
                #print(main_optimizer.param_groups[0]['lr'])
                #print('lr',i)
                lr_count += 1
                if lr_count == 10:
                    # print(i,"Reduced",mul)
                    # print(prev_loss,temp_loss,alphas)
                    scheduler.step()
                    mul/=10
                    lr_count = 0
            else:
                lr_count = 0

            '''if (abs(prev_loss - temp_loss) <= 1e-3 or abs(temp_loss - prev_loss2) <= 1e-3) and\
                 stop_count >= 5:
                print(i,prev_loss,temp_loss,constraint)
                break 
            elif abs(prev_loss - temp_loss) <= 1e-3 or abs(temp_loss - prev_loss2) <= 1e-3:
                #print(prev_loss,temp_loss)
                stop_count += 1
            else:
                stop_count = 0'''

            '''if constraint <= 0 and (stop_count >= 2 or (i + 1) % select_every == 0): #10:
                print(i,constraint)
                break
            elif constraint <= 0:
                #print(alphas,constraint,stop_count)
                stop_count += 1
            else:
                stop_count = 0'''


            '''if i>=2000:
                break'''

            prev_loss2 = prev_loss
            prev_loss = temp_loss
            #i +=1

        self.subset_idx = sub_idxs
        #print(constraint)
        #print(alphas)
        no_red_error = self.criterion(reduction='none')

        main_model.eval()

        l = [torch.flatten(p) for p in main_model.parameters()]
        flat = torch.cat(l)

        # print(func_name,len(sub_idxs),file=modelfile)
        # print(flat,file=modelfile)

        with torch.no_grad():
            '''full_trn_out = main_model(x_trn)
            full_trn_loss = criterion(full_trn_out, y_trn)
            sub_trn_out = main_model(x_trn[idxs])
            sub_trn_loss = criterion(sub_trn_out, y_trn[idxs])
            print("\nFinal SubsetTrn and FullTrn Loss:", sub_trn_loss.item(),full_trn_loss.item(),file=logfile)'''

            #val_loss = 0.
            e_val_loss = self.eval_and_return_loss(main_model, loader_val, no_red_error)
            #val_loss /= len(loader_val.batch_sampler)
            self.val_loss = torch.mean(e_val_loss)
            # print(list(e_val_loss.cpu().numpy()),file=modelfile)

            if(default == True):
                loader_tst = DataLoader(CustomDataset(x_tst, y_tst,transform=None),shuffle=False,\
                                        batch_size=self.batch_size)
                e_tst_loss = self.eval_and_return_loss(main_model, loader_tst, no_red_error)

                #test_loss /= len(loader_tst.batch_sampler)    
                self.test_loss = torch.mean(e_tst_loss)
                self.test_loss_std = torch.std(e_tst_loss)
                # print(list(e_tst_loss.cpu().numpy()),file=modelfile)

    def eval_and_return_loss(self, model, dataloader, no_red_loss_fn):
        """Return a tensor of losses given dataloader"""
        for batch_idx in list(dataloader.batch_sampler):

            inputs, targets = dataloader.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = model(inputs)
            '''if is_time:
                outputs = sc_trans.inverse_transform(outputs.cpu().numpy())
                outputs = torch.from_numpy(outputs).float()'''
            #test_loss += criterion(outputs, targets)

            if batch_idx[0] == 0:
                epoch_loss = no_red_loss_fn(outputs, targets)

            else:
                batch_loss = no_red_loss_fn(outputs, targets)
                epoch_loss = torch.cat((epoch_loss, batch_loss),dim= 0)
        return epoch_loss
       
    def val_loss(self):
        return self.val_loss.cpu().numpy()
    
    def test_loss(self):
        return (self.test_loss.cpu().numpy(), self.test_loss_std.cpu().numpy())

    def return_subset(self):
        return np.array(self.subset_idx)
