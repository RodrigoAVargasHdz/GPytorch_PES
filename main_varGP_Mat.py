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

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.utils import shuffle

ha2cm1 = 2.1947E5
def data_():
	X = np.loadtxt('Data/Train_polar_original.dat')
	y = np.loadtxt('Data/Train_ene_original.dat')
	print(X.shape)
	print(y.shape)
	
	X_test = np.loadtxt('Data/Test_polar_original.dat')
	y_test = np.loadtxt('Data/Test_ene_original.dat')
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

def denorm_data(x_norm,x_mean, x_std):
	
	x = (x_norm * x_std) + x_mean
	return x
	
'''
def norm_data(x_train,y_train, x_test):

	x_min = np.amin(x_test, axis=0, keepdims=True) 
	x_max = np.amax(x_test, axis=0, keepdims=True) 
	y_min = np.amin(y_train, axis=0, keepdims=True) 
	y_max = np.amax(y_train, axis=0, keepdims=True) 
	
	norm_x_train = (x_train - x_min)/ (x_max - x_min)
	norm_y_train = (y_train - y_min)/ (y_max - y_min)
	norm_x_test = (x_test - x_min)/ (x_max - x_min)
	
	return norm_x_train, norm_y_train, norm_x_test, x_min, x_max, y_min, y_max

def denorm_data(x_norm,x_min, x_max):
	x = (x_max - x_min)*x_norm + x_min
	return x
'''

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=51))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)	


def main_vargpytorch(train_x,train_y, test_x, n_inducing_points=None,l=None, files_ = None,x_inducing_points = np.array([])):
	
	start_time = time.time()
	gp_file,f_out = files_
	
	train_x = torch.from_numpy(train_x)
	train_x = torch.tensor(train_x,dtype=torch.float32)
	train_y = torch.from_numpy(train_y)
	test_x_np = test_x
	test_x = torch.from_numpy(test_x)
	test_x = torch.tensor(test_x,dtype=torch.float32)
	
	d = train_x.size(1)
	f = open(f_out,'w')
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
	print('NI = %i, l = %i'%(n_inducing_points,l), file=f)
	print(gp_file, file=f)
	

# 	In all other cases, he suggests using a power of 2 as the mini-batch size. 
# 	So the minibatch should be 64, 128, 256, 512, or 1024 elements large.
	train_dataset = TensorDataset(train_x, train_y)
	train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

	dummy_test_y = torch.full_like(test_x, dtype=torch.long, fill_value=0)
	test_dataset = TensorDataset(test_x,dummy_test_y)
	test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
	
	if x_inducing_points.size != 0:
		print('Using Inducing points from prev. calculation file!',file=f)
		inducing_points = torch.from_numpy(x_inducing_points)
		inducing_points = torch.tensor(inducing_points,dtype=torch.float32)
	else:
		print('Random Inducing points!',file=f)
		i0 = torch.randint(0, train_x.size(0), (n_inducing_points,))
		inducing_points = train_x[i0]
	
	if os.path.isfile(gp_file):
		print('Restarting calculation from prev {} file!'.format(gp_file),file=f)
		state_dict = torch.load(gp_file)	
		m_state_dict = torch.load(gp_file)
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		model = GPModel(inducing_points=inducing_points)
		model.load_state_dict(m_state_dict["model"])
		likelihood.load_state_dict(m_state_dict["likelihood"])
		
	else:
		model = GPModel(inducing_points=inducing_points)
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
	
	lr0 = 0.05
# 	We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
	optimizer = torch.optim.Adam([
	{'params': model.parameters()},
	{'params': likelihood.parameters()}], lr=lr0)

#	optimizer = torch.optim.SGD(model.parameters(), lr=lr0)
	
# 	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# 	Our loss object. We're using the VariationalELBO
	mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
	
	loss0 = 1E6
	model.train()
	likelihood.train()
	n_iterations = 5000
	lr_schedule_time = time.time()	
	for i in range(n_iterations+1):
		for x_batch, y_batch in train_loader:
			optimizer.zero_grad()
			output = model(x_batch)
			loss = -mll(output, y_batch)
			loss.backward()
		
		if loss.item() < loss0:
			loss0 = loss.item()
			state_dict = model.state_dict()
			likelihood_state_dict = likelihood.state_dict()
			torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, gp_file)
			inducing_points = model.variational_strategy.inducing_points.detach()
		
		if (i % 500) == 0:
			print(i,loss0,loss.item(), file=f)
		
		optimizer.step()
		scheduler.step()
		
	print('---------------------------------', file=f)
	print('Training time =  %.4f hrs ---'% ((time.time() - start_time)//3600), file=f)
	print('-----------------------------------',file=f)

#       ---------------------------------       
#       Parameters of the Trained model
	print('---------------------------------', file=f)
	ts_start_time = time.time()
	m_state_dict = torch.load(gp_file)
	model = GPModel(inducing_points=inducing_points)
	model.load_state_dict(m_state_dict["model"])
	likelihood.load_state_dict(m_state_dict["likelihood"])

	with torch.no_grad():
		print('Loss: %.8f' % (loss0), file=f)
		for param_name, param in model.named_parameters():
			print(param_name, file=f)
			print(param.detach().numpy(), file=f)


	model.eval()
	likelihood.eval()
	means = torch.tensor([0.])
# 	std_u = torch.tensor([0.])
# 	std_l = torch.tensor([0.])
	with torch.no_grad():
		for x_batch,_ in test_loader:
			preds = model(x_batch)
			means = torch.cat([means, preds.mean.cpu()])
#         	f_lower, f_upper = preds.confidence_region()
#         	std_l = torch.cat([std_l,f_lower])
#         	std_u = torch.cat([std_u,f_upper])

	means = means[1:]
	means = means.detach().numpy()
	mll = -loss0
	
	print('-----------------------------------',file=f)
	print('Total Test time =  %s seconds ---'% (time.time() - ts_start_time), file=f)
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
	f.close()
	return means, inducing_points, mll

def main_varGP(n_inducing_points=25,l=0,files_=None):
# 	file
	gp_file ,res_file_name,  inducing_points_file_name,f_out = files_

# 	Data
	X_train, X_test, y_train, y_test = data_()

	print('Normalization of the data is with respect to [Xtrain]')
	norm_X, norm_y, norm_test_x, x_mean, x_std, y_mean, y_std = norm_data(X_train,y_train, X_test)
				
	if os.path.isfile(inducing_points_file_name):
		print('Inducing points file exists!!')
		x_inducing_points = np.loadtxt(inducing_points_file_name)			
		_, _, norm_x_inducing_points, _, _, _, _ = norm_data(X_train,y_train, x_inducing_points)
			
		norm_pred_y,norm_inducing_points,mll = main_vargpytorch(norm_X, norm_y, norm_test_x, n_inducing_points,l,[gp_file,f_out],norm_x_inducing_points)
	
	else:	
		norm_pred_y,norm_inducing_points,mll = main_vargpytorch(X_train, y_train, X_test, n_inducing_points,l,[gp_file,f_out])			

	pred_y = denorm_data(norm_pred_y, y_mean, y_std)
	inducing_points = denorm_data(norm_inducing_points, x_mean, x_std)
		
	mean_SE = np.mean(np.square(y_test - pred_y))#mean_squared_error(y_test,pred_y)
	mean_AE = np.mean(np.abs(y_test - pred_y))# mean_absolute_error(y_test,pred_y)
	
	f = open(f_out,'a')
	print('l = %i'%(l),file=f)
	print('Num inducing points = %i'%(n_inducing_points),file=f)
	print('mean SE = %.12f'%(mean_SE),file=f)
	print('mean AE = %.12f'%(mean_AE),file=f)
	print('mll = %.12f'%(mll),file=f)
	print('-----------------------------------',file=f)
	f.close()
	

	pred_D = np.column_stack((X_test,pred_y))
	np.savetxt(res_file_name,pred_D)
	np.savetxt(inducing_points_file_name,inducing_points)
		
	print('-----------------------------------')
	
	r = np.append(l,n_inducing_points)
	r = np.append(r,mll)
	r = np.append(r,mean_SE)
	r = np.append(r,mean_AE)
	
	f_ = 'varGP_Mat_l_NI_mll_meanSE_meanAE.txt'
	if os.path.isfile(f_):
		R = np.loadtxt(f_)
		R = np.vstack((R,r))
		np.savetxt(f_,R)
	else:
		np.savetxt(f_,np.atleast_2d(r).T)
	
	
def main():
	start_time = time.time()
	parser = argparse.ArgumentParser(description='varGP imidazole')
	parser.add_argument('--ni', type=int, default=100, help='inducing points')
	parser.add_argument('--l', type=int, default=0, help='label')
#         parser.add_argument('--ymax', type=int, default=5000, help='max energy for training')	
	args = parser.parse_args()
	ni = args.ni
	l = args.l
	gp_file_name = 'Results_varGP_Mat/varGP_Mate_NI_%i_l_%i.pth'%(ni,l)
	res_file_name = 'Results_varGP_Mat/varGP_Mat_energy_NI_%i_l_%i.txt'%(ni,l)
	inducing_points_file_name = 'Results_varGP_Mat/varGP_inducing_points_NI_%i_l_%i.txt'%(ni,l)
	f_out = 'Results_varGP_Mat/output_varGP_Mat_NI_%i_l_%i.txt'%(ni,l)
	files_ = [gp_file_name,res_file_name,  inducing_points_file_name,f_out]
	print('-----------------------------------')
	print('N = %i, l = %i'%(ni,l))
	main_varGP(ni,l,files_)        
	print('-----------------------------------')	
	print('Running time =  %s seconds ---'% (time.time() - start_time))
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
	print('-----------------------------------')



if __name__ == "__main__":
    main()

