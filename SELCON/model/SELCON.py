import math
import numpy as np
import time
import torch
import torch.nn as nn
import copy

from utils.custom_dataset import CustomDataset_WithId, CustomDataset
from torch.utils.data import DataLoader


class FindSubset_Vect(object):
    def __init__(self, x_trn, y_trn, x_val, y_val,model,loss,device,delta,lr,lam,batch,fair=True):
        
        self.x_trn = x_trn
        self.y_trn = y_trn
        #self.trn_batch = trn_batch

        self.x_val = x_val
        self.y_val = y_val

        self.model = model
        self.criterion = loss 
        self.device = device

        self.delta = delta
        self.lr = lr
        self.lam = lam
        #self.optimizer = optimizer
        self.batch_size = batch
        self.fair = fair

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

    def grad_logistic(self,pred,actual,X):
        X_t = torch.transpose(X, 0, 1)
        print("X_t shape: ", X_t.shape)
        print("pred ",pred.shape)
        print("actual ",pred.shape)
        return torch.matmul(X_t,pred-actual)

    def logistic(self,y):
        m=torch.nn.Sigmoid()
        return m(y)

    def precompute(self,f_pi_epoch,p_epoch,alphas):
        '''
        This function calculates the following:
        1. f(∅), where ∅ is the empty set
        2. f({i}), for all data points "i" in the training dataset
        These quantities are later used to compute f(̂^S\{i}) and
        f(i|∅), which are used to calculate the upper bound m[i].
        '''

        criterion = self.criterion()
        main_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}], lr=self.lr)
                
        dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        print("starting Pre compute")
        #alphas = torch.rand_like(self.delta,requires_grad=True) 

        #print(alphas)

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20,40,100],\
        #    gamma=0.5) #[e*2 for e in change]

        #alphas.requires_grad = False
        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val, device = self.device, transform=None),\
            shuffle=False, batch_size=self.batch_size)

        ### Compute F_phi
        #for i in range(f_pi_epoch):

        prev_loss = 1000
        stop_count = 0
        i=0
        
        while(True):
            
            main_optimizer.zero_grad()
            
            '''l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)'''

            #l = [torch.flatten(p) for p in main_model.parameters()]
            #flat = torch.cat(l)
            #l2_reg = torch.sum(flat*flat)
            
            constraint = 0. 
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                val_out = self.model(inputs)
                print('val_out shape ',val_out.shape)
                print('targets shape ',targets.shape)
                # print(val_out)
                # print(targets)
                constraint += criterion(val_out, targets)
                
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - self.delta
            multiplier = alphas*constraint #torch.dot(alphas,constraint)

            loss = multiplier
            self.F_phi = loss.item()
            loss.backward()
            
            main_optimizer.step()
            #scheduler.step()
            
            '''for param in self.model.parameters():
                param.requires_grad = False
            alphas.requires_grad = True'''

            dual_optimizer.zero_grad()

            constraint = 0.
            for batch_idx in list(loader_val.batch_sampler):
                    
                inputs, targets = loader_val.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                val_out = self.model(inputs)
                constraint += criterion(val_out, targets)
            
            constraint /= len(loader_val.batch_sampler)
            constraint = constraint - self.delta
            multiplier = -1.0*alphas*constraint #torch.dot(-1.0*alphas,constraint)
            
            multiplier.backward()
            
            dual_optimizer.step()

            alphas.requires_grad = False
            alphas.clamp_(min=0.0)
            alphas.requires_grad = True
            #print(alphas)

            '''for param in self.model.parameters():
                param.requires_grad = True'''

            if loss.item() <= 0.:
                break

            #if i>= f_pi_epoch:
            #    break

            if abs(prev_loss - loss.item()) <= 1e-1 and stop_count >= 5:
                break 
            elif abs(prev_loss - loss.item()) <= 1e-1:
                stop_count += 1
            else:
                stop_count = 0

            prev_loss = loss.item()
            i+=1

            #if i % 50 == 0:
            #    print(loss.item(),alphas,constraint)

        print("Finishing F phi")

        if loss.item() <= 0.:
            alphas = torch.zeros_like(alphas)

        print(loss.item())

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat = torch.cat(l).detach().clone()
        
        '''main_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        
        alphas = torch.ones(alphas.shape) * 1e-4
        x_val_ext = torch.cat((self.x_val,torch.ones(self.x_val.shape[0],device=self.device).view(-1,1))\
                ,dim=1)

        yTy = torch.dot(self.y_val,self.y_val)
        
        #yTX = alphas*torch.matmul(self.y_val.view(1,-1),x_val_ext)

        XTX = torch.inverse(alphas*torch.matmul(x_val_ext.T,x_val_ext))

        XTy = alphas*torch.matmul(x_val_ext.T,self.y_val.view(-1,1))

        accum = torch.matmul(XTX, XTy) #torch.matmul(XTy,XTX)

        flat = alphas*yTy - alphas*self.delta - accum'''
        
        self.F_values = torch.zeros(len(self.x_trn),device=self.device)

        device_new = self.device #"cuda:2"#self.device #
        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn, self.y_trn,\
            transform=None),shuffle=False,batch_size=self.batch_size*20)

        loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,device = self.device,transform=None),\
            shuffle=False,batch_size=self.batch_size*20)
        
        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, idxs = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ele_delta = self.delta.repeat(targets.shape[0]).to(self.device)
        
            weights = flat.view(1,-1).repeat(targets.shape[0], 1)
            ele_alphas = alphas.detach().repeat(targets.shape[0]).to(self.device)
            #print(weights.shape)

            exp_avg_w = torch.zeros_like(weights)
            exp_avg_sq_w = torch.zeros_like(weights)

            #exp_avg_a = torch.zeros_like(ele_alphas)
            #exp_avg_sq_a = torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1))\
                ,dim=1)

            bias_correction1 = 1.0 
            bias_correction2 = 1.0 

            for i in range(p_epoch):

                # trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                # fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]
                # print(torch.sum(exten_inp*weights,dim=1).shape)
                trn_loss_g = criterion(self.logistic(torch.sum(exten_inp*weights,dim=1)),targets)
                
                fin_trn_loss_g =self.grad_logistic(self.logistic(torch.sum(exten_inp*weights,dim=1)),targets,exten_inp)
                #no_bias = weights.clone()
                #no_bias[-1,:] = torch.zeros(weights.shape[0])
                
                weight_grad = fin_trn_loss_g+ 2*self.lam*\
                    torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)
                #+fin_val_loss_g*ele_alphas[:,None]

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = (self.lr) * math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
            val_losses = 0.
            for batch_idx_val in list(loader_val.batch_sampler):
                    
                inputs_val, targets_val = loader_val.dataset[batch_idx_val]
                inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                exten_val = torch.cat((inputs_val,torch.ones(inputs_val.shape[0],device=self.device).view(-1,1)),dim=1)
                
                exten_val_y = torch.mean(targets_val).repeat(min(self.batch_size*20,targets.shape[0]))

                # val_loss = torch.sum(weights*torch.mean(exten_val,dim=0),dim=1) - exten_val_y

                # val_losses+= val_loss*val_loss #torch.mean(val_loss*val_loss,dim=0)
                val_loss = criterion(self.logistic(torch.sum(exten_inp*weights,dim=1)),exten_val_y)
                val_losses += val_loss
                # fin_trn_loss_g =self.grad_logistic(self.logsitic(torch.sum(exten_inp*weights,dim=1)),exten_val_y,exten_inp)
            
            reg = torch.sum(weights[:,:-1]*weights[:,:-1],dim=1)
            # trn_loss = torch.sum(exten_inp*weights,dim=1) - targets
            trn_loss = criterion(self.logistic(torch.sum(exten_inp*weights,dim=1)),exten_val_y)
            # fin_trn_loss_g =self.grad_logistic(self.logsitic(torch.sum(exten_inp*weights,dim=1)),exten_val_y,exten_inp)

            self.F_values[idxs] = trn_loss+ self.lam*reg +torch.max(torch.zeros_like(ele_alphas),\
                (val_losses/len(loader_val.batch_sampler)-ele_delta)*ele_alphas)

        print(self.F_values[:10])

        #self.F_values = self.F_values - max(loss.item(),0.) 

        print("Finishing Element wise F")


    def return_subset(self,theta_init,p_epoch,curr_subset,alphas,budget,batch,\
        step,w_exp_avg,w_exp_avg_sq,a_exp_avg,a_exp_avg_sq):

        criterion = self.criterion()
        m_values = self.F_values.detach().clone() #torch.zeros(len(self.x_trn))
        
        self.model.load_state_dict(theta_init)

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=batch)

        if self.fair:
            loader_val = DataLoader(CustomDataset(self.x_val, self.y_val,device = self.device,transform=None),\
                shuffle=False,batch_size=batch)

        sum_error = torch.nn.BCELoss(reduction='sum')       

        with torch.no_grad():

            F_curr = 0.

            for batch_idx in list(loader_tr.batch_sampler):
            
                inputs, targets, _ = loader_tr.dataset[batch_idx]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                scores = self.model(inputs)
                #print(self.criterion(scores, targets).item())

                F_curr += sum_error(scores, targets).item() 

            #F_curr /= len(loader_tr.batch_sampler)
            #print(F_curr,end=",")

            l = [torch.flatten(p) for p in self.model.parameters()]
            flatt = torch.cat(l)
            l2_reg = torch.sum(flatt[:-1]*flatt[:-1])

            if self.fair:
                valloss = 0.
                for batch_idx in list(loader_val.batch_sampler):

                    inputs, targets = loader_val.dataset[batch_idx]
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    scores = self.model(inputs)
                    valloss += criterion(scores, targets).item()

                constraint = valloss/len(loader_val.batch_sampler) - self.delta
                multiplier = alphas*constraint #torch.dot(alphas,constraint)

                F_curr += (self.lam*l2_reg*len(curr_subset) + multiplier).item()
            else:
                F_curr += (self.lam*l2_reg*len(curr_subset)).item() #+ multiplier).item()

        # val_mul = multiplier.item()
        
        #print(self.lam*l2_reg*len(curr_subset), multiplier)
        #print(F_curr)

        main_optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        #dual_optimizer = torch.optim.Adam([{'params': alphas}], lr=self.lr)

        l = [torch.flatten(p) for p in self.model.state_dict().values()]
        flat = torch.cat(l).detach()

        loader_tr = DataLoader(CustomDataset_WithId(self.x_trn[curr_subset], self.y_trn[curr_subset],\
            transform=None),shuffle=False,batch_size=self.batch_size)

        #ele_delta = self.delta.repeat(min(self.batch_size,self.y_trn[curr_subset].shape[0])).to(self.device)

        beta1,beta2 = main_optimizer.param_groups[0]['betas']
        #main_optimizer.param_groups[0]['eps']

        rem_len = (len(curr_subset)-1)

        b_idxs = 0

        device_new = self.device #"cuda:2" #self.device #

        for batch_idx in list(loader_tr.batch_sampler):

            inputs, targets, _ = loader_tr.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            weights = flat.repeat(targets.shape[0], 1)
            exp_avg_w = w_exp_avg.repeat(targets.shape[0], 1)#torch.zeros_like(weights)
            exp_avg_sq_w = w_exp_avg_sq.repeat(targets.shape[0], 1) #torch.zeros_like(weights)

            if self.fair:
                ele_delta = self.delta.repeat(targets.shape[0]).to(self.device)
                ele_alphas = alphas.detach().repeat(targets.shape[0]).to(self.device)
                exp_avg_a = a_exp_avg.repeat(targets.shape[0])#torch.zeros_like(ele_alphas)
                exp_avg_sq_a = a_exp_avg_sq.repeat(targets.shape[0]) #torch.zeros_like(ele_alphas)

            exten_inp = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1)),dim=1)

            bias_correction1 = beta1**step#1.0 
            bias_correction2 = beta2**step#1.0 

            for i in range(p_epoch):
                sum_fin_trn_loss_g = self.eval_and_return_gradient(loader_tr, criterion, weights, train_set=True, clear_cache=True)
                #fin_trn_loss_g /= len(loader_tr.batch_sampler)
                sum_fin_trn_loss_g = sum_fin_trn_loss_g.to(self.device)

                # trn_loss_g = torch.sum(exten_inp*weights,dim=1) - targets
                # fin_trn_loss_g = exten_inp*2*trn_loss_g[:,None]

                fin_trn_loss_g = self.grad_logistic(torch.sum(exten_inp*weights,dim=1),targets,exten_inp)

                fin_trn_loss_g = (sum_fin_trn_loss_g - fin_trn_loss_g)/rem_len

                if self.fair:
                    fin_val_loss_g = self.eval_and_return_gradient(loader_val, criterion, weights, train_set=False, clear_cache=True)
                    fin_val_loss_g /= len(loader_val.batch_sampler)
                    fin_val_loss_g = fin_val_loss_g.to(self.device)
                    weight_grad = fin_trn_loss_g+ 2*rem_len*\
                        torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)+\
                            fin_val_loss_g*ele_alphas[:,None]
                else:
                    weight_grad = fin_trn_loss_g+ 2*rem_len*\
                        torch.cat((weights[:,:-1], torch.zeros((weights.shape[0],1),device=self.device)),dim=1)


                #print(weight_grad[0])

                exp_avg_w.mul_(beta1).add_(1.0 - beta1, weight_grad)
                exp_avg_sq_w.mul_(beta2).addcmul_(1.0 - beta2, weight_grad, weight_grad)
                denom = exp_avg_sq_w.sqrt().add_(main_optimizer.param_groups[0]['eps'])

                bias_correction1 *= beta1
                bias_correction2 *= beta2
                step_size = (self.lr)* math.sqrt(1.0-bias_correction2) / (1.0-bias_correction1)
                weights.addcdiv_(-step_size, exp_avg_w, denom)
                
                #weights = weights - self.lr*(weight_grad)

                '''print(self.lr)
                print((fin_trn_loss_g+ 2*self.lam*weights +fin_val_loss_g*ele_alphas[:,None])[0])'''

                #print(weights[0])

                if self.fair:
                    val_losses = torch.zeros_like(ele_delta).to(device_new)
                    val_loss = self.eval_and_return_loss(loader_val, criterion, weights,
                                                         train_set=False, clear_cache=True)

                    val_losses = (val_losses + val_loss).to(self.device)

                    alpha_grad = val_losses/len(loader_val.batch_sampler)-ele_delta

                    exp_avg_a.mul_(beta1).add_(1.0 - beta1, alpha_grad)
                    exp_avg_sq_a.mul_(beta2).addcmul_(1.0 - beta2, alpha_grad, alpha_grad)
                    denom = exp_avg_sq_a.sqrt().add_(main_optimizer.param_groups[0]['eps'])
                    ele_alphas.addcdiv_(step_size, exp_avg_a, denom)
                    ele_alphas[ele_alphas < 0] = 0
                    #print(ele_alphas[0])

                    #ele_alphas = ele_alphas + self.lr*(torch.mean(val_loss_p*val_loss_p,dim=0)-ele_delta)

            reg = torch.sum(weights[:,:-1]*weights[:,:-1],dim=1)

            trn_losses = self.eval_and_return_loss(loader_tr, criterion, weights, train_set=True)

            # trn_loss_ind = torch.sum(exten_inp*weights,dim=1) - targets
            # print("logistic shape ",self.logistic(exten_inp*weights).shape)
            # print("external val y ",exten_val_y.shape)
            # print("weights ",weights.shape)
            # print("external inpu ",exten_inp.shape)
            print("shape",targets.shape)
            trn_loss_ind = criterion(self.logistic(torch.sum(torch.matmul(exten_inp,weights.T),dim=1)),targets)

            trn_losses -= trn_loss_ind

            if self.fair:
                val_losses = self.eval_and_return_loss(loader_val, criterion, weights, train_set=False)
                abs_value = F_curr - (trn_losses + self.lam*reg*rem_len \
                    + (val_losses/len(loader_val.dataset)-ele_delta)*ele_alphas)
            else:
                abs_value = F_curr - (trn_losses + self.lam*reg*rem_len)

            neg_ind = ((abs_value ) < 0).nonzero().view(-1)

            abs_value [neg_ind] = torch.max(self.F_values)

            m_values[torch.tensor(curr_subset, dtype = torch.long)[b_idxs*self.batch_size:(b_idxs+1)*self.batch_size]]\
                 = abs_value

            b_idxs +=1

        values,indices =m_values.topk(budget,largest=False)

        return list(indices.cpu().numpy())

    def eval_and_return_loss(self, dataloader, criterion, weights,
                             train_set=True, clear_cache=False):
        total_loss = 0.
        for batch_idx_trn in list(dataloader.batch_sampler):

            if train_set:
                inputs, targets,_ = dataloader.dataset[batch_idx_trn]
            else:
                inputs, targets = dataloader.dataset[batch_idx_trn]

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            exten = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device).view(-1,1)),dim=1)
            exten_y = targets.view(-1,1).repeat(1,min(self.batch_size,targets.shape[0]))
            #print(exten_val_y[0])

            # trn_loss = torch.matmul(exten,torch.transpose(weights, 0, 1)) - exten_y
            loss = criterion(self.logistic(torch.matmul(exten,weights.T)),exten_y)

            total_loss += torch.sum(loss,dim=0)
            if clear_cache:
                del exten, exten_y, loss, inputs, targets
                torch.cuda.empty_cache()
        return total_loss

    def eval_and_return_gradient(self, dataloader, criterion, weights,
                                 train_set=True, clear_cache=True):
        final_grad = torch.zeros_like(weights).to(self.device)
        #val_losses = torch.zeros_like(ele_delta).to(device_new)
        for batch_idx in list(dataloader.batch_sampler):

            if train_set:
                inputs, targets, _ = dataloader.dataset[batch_idx]
            else:
                inputs, targets = dataloader.dataset[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            exten = torch.cat((inputs,torch.ones(inputs.shape[0],device=self.device)\
                .view(-1,1)),dim=1).to(self.device)
            #print(exten.shape)

            exten_y = targets.view(-1,1).repeat(1,min(self.batch_size,\
                targets.shape[0])).to(self.device)
            #print(exten_y[0])

            # val_loss_p = 2*(torch.matmul(exten,torch.transpose(weights, 0, 1).to(self.device))\
            #      - exten_y)
            #val_losses += torch.mean(val_loss_p*val_loss_p,dim=0)
            #val_loss_g = torch.unsqueeze(val_loss_p, dim=2).repeat(1,1,flat.shape[0])
            #print(val_loss_g[0][0])

            #mod_val = torch.unsqueeze(exten, dim=1).repeat(1,targets.shape[0],1)
            #print(mod_val[0])
            # final_grad += torch.mean(val_loss_p[:,:,None]*exten[:,None,:],dim=0)
            # print("")
            loss = criterion(self.logistic(torch.matmul(exten,torch.transpose(weights, 0, 1).to(self.device))),exten_y)
            print(final_grad.shape)
            final_grad +=self.grad_logistic(self.logistic(torch.matmul(exten,torch.transpose(weights, 0, 1).to(self.device))),exten_y,exten).T #changed

            if clear_cache:
                del exten, exten_y, loss, inputs, targets #mod_val,val_loss_g,
                torch.cuda.empty_cache()
        return final_grad
