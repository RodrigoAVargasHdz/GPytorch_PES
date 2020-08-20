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
def data_():
	X = np.loadtxt('Data/Train_X_original.dat')
	y = np.loadtxt('Data/Train_y_original.dat')
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
    def __init__(self,train_x, train_y,inducing_points, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()#ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=51))
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def main_spGPytorch(train_x,train_y, test_x, test_y, inducing_points_x,l, files_):
	
	start_time = time.time()
	gp_file,f_out = files_
		
	train_x = torch.from_numpy(train_x)
	train_x = train_x.double()#.float() #torch.tensor(train_x,dtype=torch.float32)
	train_y = torch.from_numpy(train_y)
	train_y = train_y.double()#.float()
	test_x_np = test_x
	test_x = torch.from_numpy(test_x)
	test_x = test_x.double()#.float() #torch.tensor(test_x,dtype=torch.float32)
	test_y = torch.from_numpy(test_y)
	test_y = test_y.double()#.float()		

	y_pred0 = test_y
	mse0 = 1E6
	
	d = train_x.size(1)

	
	if inducing_points_x.shape[0] == 1:
		n_inducing_points = inducing_points_x[0]
	else:
		n_inducing_points = inducing_points_x.shape[0]

	f = open(f_out,'w')	
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
	print('NI = %i, l = %i'%(n_inducing_points,l), file=f)
	print(gp_file, file=f)
	f.close()
	

# 	In all other cases, he suggests using a power of 2 as the mini-batch size. 
# 	So the minibatch should be 64, 128, 256, 512, or 1024 elements large.
	train_dataset = TensorDataset(train_x, train_y)
	train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

	dummy_test_y = torch.full_like(test_x, dtype=torch.long, fill_value=0)
	test_dataset = TensorDataset(test_x,dummy_test_y)
	test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
		
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
		
	def test(model,likelihood):
		model.eval()
		likelihood.eval()

		means = torch.tensor([0.])
		with gpytorch.settings.max_preconditioner_size(20), torch.no_grad(),gpytorch.settings.max_cg_iterations(7500):
			with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(200), gpytorch.settings.fast_pred_var():
				for x_batch,_ in test_loader:
					preds = likelihood(model(x_batch))
					mean = preds.mean.cpu()
					means = torch.cat([means, mean])
				
		means = means[1:] 
		mean_SE = torch.mean((test_y.cpu() - means)**2)
		return mean_SE, means.detach()	
	
		
	loss0 = 1E6
	epochs_iter = 1
	n_iterations = 1500
	
	for j in range(epochs_iter):
		f = open(f_out,'a')
		if os.path.isfile(gp_file) and j == 0:
			print('Restarting calculation from prev {} file!'.format(gp_file),file=f)
			if inducing_points_x.shape[0] == 1:
				print('Random {} inducing points!'.format(n_inducing_points),file=f)
				i0 = torch.randint(0, train_x.size(0), (n_inducing_points,))
				inducing_points = train_x[i0]	
			else:
				n_inducing_points = inducing_points_x.shape[0]
				inducing_points = torch.from_numpy(inducing_points_x)
				inducing_points = inducing_points.float()	
				print('{} of inducing points from prec. file!'.format(n_inducing_points),file=f)		
			
			state_dict = torch.load(gp_file)	
			model = GPModel(train_x, train_y, inducing_points,likelihood)
			model.load_state_dict(state_dict)
			
		else:
			n_inducing_points = inducing_points_x[0]
			print('Random {} inducing points and NO prev.  {} file!!'.format(n_inducing_points,gp_file),file=f)
			i0 = torch.randint(0, train_x.size(0), (n_inducing_points,))
			inducing_points = train_x[i0]	
			model = GPModel(train_x, train_y, inducing_points,likelihood)
		f.close()
		
		model = model.double()
		likelihood = likelihood.double()
		
# 		We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
		optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# 		optimizer = torch.optim.SGD(model.parameters(), lr=0.05)		
		
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.75)

# 		Our loss object. We're using the VariationalELBO
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)				

		for i in range(n_iterations+1):
			model.train()
			likelihood.train()	
		
			with gpytorch.settings.max_cg_iterations(7500):
				optimizer.zero_grad()
				output = model(train_x)
				loss = -mll(output, train_y)
					
				if loss.item() < loss0:
					loss0 = loss.item()
					f = open(f_out,'a')	 	 				
					print('* n = {}, loss.item() = {} '.format(i,loss.item()), file=f)				
					f.close()
# 					state_dict = model.state_dict()
# 					torch.save(state_dict, gp_file)
# 					inducing_points = model.covar_module.inducing_points.detach()				
					
					mse, y_pred = test(model,likelihood)
					if mse < mse0:
						mse0 = mse
						y_pred0 = y_pred
						f = open(f_out,'a')	 	 				
						print('* mse = {} '.format(mse), file=f)				
						f.close()
						state_dict = model.state_dict()
						torch.save(state_dict, gp_file)
						inducing_points = model.covar_module.inducing_points.detach()
							
				if (i % 100) == 0:
					f = open(f_out,'a')	 
					print(j,i,loss0,loss.item(), file=f)
					f.close()
				
				loss.backward()			
				optimizer.step()
				scheduler.step()	
	
	f = open(f_out,'a')	
	print('---------------------------------', file=f)
	print('Training time =  %.6f hrs ---'% ((time.time() - start_time)//3600), file=f)

#       ---------------------------------       
#       Parameters of the Trained model
	print('---------------------------------', file=f)
	ts_start_time = time.time()
	state_dict = torch.load(gp_file)
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model0 = GPModel(train_x, train_y, inducing_points,likelihood)
	model0.load_state_dict(state_dict)
	
	model0 = model0.double()
	likelihood = likelihood.double()
	
	output = model0(train_x)
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model0)
	loss = -mll(output, train_y)	
	print('Loss: %.8f' % (loss.item()), file=f)

	with torch.no_grad():
		print('Loss: %.8f' % (loss0), file=f)
		for param_name, param in model0.named_parameters():
			print(param_name, file=f)
			print(param.detach().numpy(), file=f)

# 	mse, y_pred = test(model0,likelihood)
	mse,y_ped = mse0,y_pred0
	print('MSE = {}'.format(mse),file=f)	
	means = y_pred.detach().numpy()
	inducing_points = inducing_points.detach().numpy()
	mll = -loss0
	
	print('-----------------------------------',file=f)
	print('Total Test time =  %s seconds ---'% (time.time() - ts_start_time), file=f)
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
	f.close()
	return means, inducing_points, mll

def main_spGP(n_inducing_points=25,l=0,files_=None):
# 	file
	gp_file ,res_file_name,  inducing_points_file_name,f_out = files_

# 	Data
	X_train, X_test, y_train, y_test = data_()
	
	norm_X, norm_y, norm_test_x, x_mean, x_std, y_mean, y_std = norm_data(X_train,y_train, X_test)	
	norm_test_y = norm_data_y(y_train, y_test)
	
	if os.path.isfile(inducing_points_file_name):
		X_inducing_points = np.loadtxt(inducing_points_file_name)
		print('Inducing points from prev. calculations!', X_inducing_points.shape)
		_, _, norm_x_inducing_points, _, _, _, _ = norm_data(X_train,y_train, X_inducing_points)	
	else:
		norm_x_inducing_points = np.array([n_inducing_points])

# 	spGPytorch
	norm_pred_y,norm_inducing_points,mll = main_spGPytorch(norm_X, norm_y, norm_test_x, norm_test_y,norm_x_inducing_points,l,[gp_file,f_out])
	pred_y = denorm_data(norm_pred_y, y_mean, y_std)
	inducing_points = denorm_data(norm_inducing_points, x_mean, x_std)
		
	mean_SE = np.mean(np.square(y_test - pred_y))#mean_squared_error(y_test,pred_y)
	mean_AE = np.mean(np.abs(y_test - pred_y))# mean_absolute_error(y_test,pred_y)
	
	print('l = %i'%(l))
	print('Num inducing points = %i'%(n_inducing_points))
	print('mean SE = %.6f'%(mean_SE))
	print('mean AE = %.6f'%(mean_AE))
	print('mll = %.6f'%(mll))
	

	pred_D = np.column_stack((X_test,pred_y))
	np.savetxt(res_file_name,pred_D)
	np.savetxt(inducing_points_file_name,inducing_points)
		
	print('-----------------------------------')
	
	r = np.append(l,n_inducing_points)
	r = np.append(r,mll)
	r = np.append(r,mean_SE)
	r = np.append(r,mean_AE)
	
	f_ = 'spGP_Mat_l_NI_mll_meanSE_meanAE.txt'
	if os.path.isfile(f_):
		R = np.loadtxt(f_)
		R = np.vstack((R,r))
		np.savetxt(f_,R)
	else:
		np.savetxt(f_,np.atleast_2d(r).T)
	
	
def main():
	start_time = time.time()
	parser = argparse.ArgumentParser(description='spGP')
	parser.add_argument('--ni', type=int, default=100, help='inducing points')
	parser.add_argument('--l', type=int, default=0, help='label')
#         parser.add_argument('--ymax', type=int, default=5000, help='max energy for training')	
	args = parser.parse_args()
	ni = args.ni
	l = args.l
	gp_file_name = 'Results_spGP_Mat/spGP_Mat_NI_%i_l_%i.pth'%(ni,l)
	res_file_name = 'Results_spGP_Mat/spGP_Mat_energy_NI_%i_l_%i.txt'%(ni,l)
	inducing_points_file_name = 'Results_spGP_Mat/spGP_Mat_inducing_points_NI_%i_l_%i.txt'%(ni,l)
	f_out = 'Results_spGP_Mat/output_spGP_Mat_NI_%i_l_%i.txt'%(ni,l)
	files_ = [gp_file_name,res_file_name,  inducing_points_file_name,f_out]
	print('-----------------------------------')
	print('N = %i, l = %i'%(ni,l))
	main_spGP(ni,l,files_)        
	print('-----------------------------------')	
	print('Running time =  %s seconds ---'% (time.time() - start_time))
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
	print('-----------------------------------')



if __name__ == "__main__":
    main()

