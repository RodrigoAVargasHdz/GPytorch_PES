import os
import time
import datetime
import numpy as np
import argparse

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.utils import shuffle

ha2cm1 = 2.1947E5
def data_(Ntrain = 1000):
	X = np.loadtxt('Data/Train_X_original.dat')
	y = np.loadtxt('Data/Train_y_original.dat')
	X = X[:Ntrain,:]
	y = y[:Ntrain]
	print(X.shape)
	print(y.shape)
	
	X_test = np.loadtxt('Data/Test_X_original.dat')
	y_test = np.loadtxt('Data/Test_y_original.dat')
	print(X_test.shape)
	print(y_test.shape)
	
	return X, X_test, y, y_test
	
def check_inducing_potins(X):
	i0 = np.where(np.amin(X,axis=1, keepdims=True) > 0.9)[0]
	if i0.shape[0] > 0:
		n0 = X.shape[0]
		X = X[i0,:]
		n = X.shape[0]
		print('Real number of inducing points %i (%i)'%(n,n0))
	return X
	
def norm_data_c(x,y):
	x_mean = np.mean(x,axis = 0, keepdims=True)
	x_std = np.std(x,axis = 0, keepdims=True)
	y_mean = np.mean(y,axis = 0, keepdims=True)
	y_std = np.std(y,axis = 0, keepdims=True)
	return x_mean, x_std, y_mean, y_std

def norm_data(x_train,y_train, x_test):
	
	x_mean, x_std, y_mean, y_std = norm_data_c(x_train, y_train)
	print(x_mean, x_std, y_mean, y_std)
	
	norm_x_train = (x_train - x_mean)/ x_std
	norm_y_train = (y_train - y_mean)/ y_std

	norm_x_test = (x_test - x_mean)/ x_std	
	
	return norm_x_train, norm_y_train, norm_x_test, x_mean, x_std, y_mean, y_std

def norm_data_y(y_train,y_test):
	y_mean = np.mean(y_train,axis = 0, keepdims=True)
	y_std = np.std(y_train,axis = 0, keepdims=True)
	norm_y_train = (y_test - y_mean)/ y_std

	return norm_y_train
		

def denorm_data(x_norm,x_mean, x_std):
	
	x = (x_norm * x_std) + x_mean
	return x

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,d):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()#ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=d))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def main_GPytorch(train_x,train_y, test_x, test_y, n_inducing_points=None,l=None, files_ = None):
	
	start_time = time.time()
	gp_file,f_out = files_
	
	train_x = torch.from_numpy(train_x)
	train_x = train_x.double()	#float() #torch.tensor(train_x,dtype=torch.float32)
	train_y = torch.from_numpy(train_y)
	train_y = train_y.double()#.float()
	test_x_np = test_x
	test_x = torch.from_numpy(test_x)
	test_x = test_x.float() #torch.tensor(test_x,dtype=torch.float32)
	test_y = torch.from_numpy(test_y)
	test_y = test_y.float()	
	
	y_pred = test_y
	mse0 = 1E6
	
	d = train_x.size(1)
	f = open(f_out,'w')
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
	print('NI = %i, l = %i'%(n_inducing_points,l), file=f)
	print(gp_file, file=f)
	
# 	In all other cases, he suggests using a power of 2 as the mini-batch size. 
# 	So the minibatch should be 64, 128, 256, 512, or 1024 elements large.

	dummy_test_y = torch.full_like(test_x, dtype=torch.long, fill_value=0)
	test_dataset = TensorDataset(test_x,dummy_test_y)
	test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
		
	if os.path.isfile(gp_file):
		print('Restarting calculation from prev {} file!'.format(gp_file),file=f)
		state_dict = torch.load(gp_file)
		print(state_dict,file=f)
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		model = GPModel(train_x, train_y, likelihood,d)
		model.load_state_dict(state_dict)
		print('------------------------------------',file=f)
			
	else:
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		model = GPModel(train_x, train_y, likelihood,d)
	
	model = model.double()
	likelihood = likelihood.double()
# 		
	def test(model,likelihood):
		model.eval()
		likelihood.eval()

		means = torch.tensor([0.])
		with torch.no_grad():
			for x_batch,_ in test_loader:
				preds = likelihood(model(x_batch.double()))
				mean = preds.mean.cpu()
				means = torch.cat([means, mean])
				
		means = means[1:] 
		mean_SE = torch.mean((test_y.cpu() - means)**2)
		return mean_SE, means.detach() 	

# 	We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
	
	if l == 0:
		lr0 = 0.5	
		optimizer = torch.optim.SGD(model.parameters(), lr=lr0)	
	else:
		lr0 = 0.1
		optimizer = torch.optim.Adam([
    	{'params': model.parameters()},  # Includes GaussianLikelihood parameters
		], lr=lr0)
		
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.75)

# 	Our loss object. We're using the VariationalELBO
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)	
	
	if os.path.isfile(gp_file):
		output = model(train_x)
		loss = -mll(output, train_y)
		loss0 = loss.item()
		print('Loss function from prev file, loss  =  {}'.format(loss0) ,file=f)
	else:
		 loss0 = 1E6
		
	n_iterations = 2000
	lr_schedule_time = time.time()		
	for i in range(n_iterations+1):
		model.train()
		likelihood.train()	
		
		optimizer.zero_grad()
		output = model(train_x)
		loss = -mll(output, train_y)
		loss.backward()
		
		if loss.item() < loss0:
			loss0 = loss.item()
# 			print('* n = {}, loss0 = {}, loss.item() = {} '.format(i,loss0,loss.item()), file=f)
			mse, y_pred0 = test(model,likelihood)
			if mse < mse0:
				state_dict = model.state_dict()
				torch.save(state_dict, gp_file)
				y_pred = y_pred0 
				mese0 = mse				
			print('* n = {}, mean-SE = {}, loss.item() = {} '.format(i,mse,loss.item()), file=f)
# 			print(state_dict,file=f)
		
		if (i % 100) == 0:
			print(i,loss0,loss.item(), file=f)
			
		scheduler.step()
		optimizer.step()	
		
	print('---------------------------------', file=f)
	print('Training time =  %.6f hrs ---'% ((time.time() - start_time)), file=f)

#       ---------------------------------       
#       Parameters of the Trained model
	print('---------------------------------', file=f)
	ts_start_time = time.time()
	state_dict_file = torch.load(gp_file)
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	print(state_dict_file,file=f)

	mll = -loss0
	
	print('-----------------------------------',file=f)
	print('Total Test time =  %s seconds ---'% (time.time() - ts_start_time), file=f)
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
	f.close()
	return y_pred.numpy(), mll

def main_GP(n_points=100,l=0,files_=None):
# 	file
	gp_file ,res_file_name,  inducing_points_file_name,f_out = files_

# 	Data
	X_train, X_test, y_train, y_test = data_(n_points)

# 	TEST FOR normalize training data 
	norm_X, norm_y, norm_test_x, x_mean, x_std, y_mean, y_std = norm_data(X_train,y_train, X_test)
	norm_test_y = norm_data_y(y_train,y_test)
		
	norm_pred_y,mll = main_GPytorch(norm_X, norm_y, norm_test_x, norm_test_y, n_points,l,[gp_file,f_out])
	
	pred_y = denorm_data(norm_pred_y, y_mean, y_std)

# 	TEST FOR non-normalize training data 
# 	pred_y,mll = main_GPytorch(X_train, y_train, X_test, y_test, n_points,l,[gp_file,f_out])
		
	mean_SE = np.mean(np.square(y_test - pred_y))#mean_squared_error(y_test,pred_y)
	mean_AE = np.mean(np.abs(y_test - pred_y))# mean_absolute_error(y_test,pred_y)
	
	print('l = %i'%(l))
	print('Num inducing points = %i'%(n_points))
	print('mean SE = %.6f'%(mean_SE))
	print('mean AE = %.6f'%(mean_AE))
	print('mll = %.6f'%(mll))
	

	pred_D = np.column_stack((X_test,pred_y))
	np.savetxt(res_file_name,pred_D)
		
	print('-----------------------------------')
	
	r = np.append(l,n_points)
	r = np.append(r,mll)
	r = np.append(r,mean_SE)
	r = np.append(r,mean_AE)
	
	f_ = 'GP_Mat_l_NI_mll_meanSE_meanAE.txt'
	if os.path.isfile(f_):
		R = np.loadtxt(f_)
		R = np.vstack((R,r))
		np.savetxt(f_,R)
	else:
		np.savetxt(f_,np.atleast_2d(r).T)
	
	
def main():
	start_time = time.time()
	parser = argparse.ArgumentParser(description='GP ')
	parser.add_argument('--n', type=int, default=100, help='number of training points')
	parser.add_argument('--l', type=int, default=0, help='label')
#         parser.add_argument('--ymax', type=int, default=5000, help='max energy for training')	
	args = parser.parse_args()
	ni = args.n
	l = args.l
	gp_file_name = 'Results_GP_Mat/GP_Mat_NI_%i_l_%i.pth'%(ni,l)
	res_file_name = 'Results_GP_Mat/GP_Mat_y_NI_%i_l_%i.txt'%(ni,l)
	inducing_points_file_name = 'Results_GP/GP_Mat_training_points_NI_%i_l_%i.txt'%(ni,l)
	f_out = 'Results_GP_Mat/output_GP_Mat_NI_%i_l_%i.txt'%(ni,l)
	files_ = [gp_file_name,res_file_name,  inducing_points_file_name,f_out]
	print('-----------------------------------')
	print('N = %i, l = %i'%(ni,l))
	main_GP(ni,l,files_)        
	print('-----------------------------------')	
	print('Running time =  %s seconds ---'% (time.time() - start_time))
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
	print('-----------------------------------')



if __name__ == "__main__":
    main()

