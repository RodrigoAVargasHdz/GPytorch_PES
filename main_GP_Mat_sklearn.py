import argparse
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ExpSineSquared, RationalQuadratic, DotProduct, ConstantKernel as C
# import matplotlib.pyplot as plt


#============= Imidazle Connection =============================================#
def GPR(Ntrain=5000):
	filename = 'simple_Niter_%i'%(Ntrain) 
	Direname = './Results_sklGP/'
	f = open( Direname + 'rmse_'+ filename + '.dat', mode='w' )

	X = np.loadtxt( 'Data/Train_X_original.dat', dtype='float')
	Y = np.loadtxt( 'Data/Train_y_original.dat', dtype='float').reshape(-1, 1)

	X = X[:Ntrain]
	Y = Y[:Ntrain]

	k = [C( 1.0, (1e-6, 1e+4) ),
     Matern( length_scale=np.ones(51),     length_scale_bounds=(1e-5, 1e+5), nu=2.5 ),
     RBF(    length_scale=np.ones(51),     length_scale_bounds=(1e-5, 1e+5) ),
     RBF(    length_scale=np.ones(51)*2.0, length_scale_bounds=(1e-5, 1e+5) ),
     Matern( length_scale=np.ones(51),     length_scale_bounds=(1e-5, 1e+5), nu=1.5 )
	]

	kernel = k[0]*k[1] #+ k[0]*k[2] + k[0]*k[3] + k[0]*k[4]
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10).fit(X, Y)
	np.savetxt( Direname + 'alpha_'+ filename +'.dat', gp.alpha_)
	f.write('Log-Marginal-Likelihood:\n %.6f \n'%(gp.log_marginal_likelihood(gp.kernel_.theta)))
	f.write( '%s \n'%gp.kernel_ )
	f.write( '%s \n'%gp.kernel_.get_params() )
	f.write('----------------------------------------------------------------\n')


	x = np.loadtxt( 'Data/Test_X_original.dat', dtype='float') 
	y = np.loadtxt( 'Data/Test_y_original.dat', dtype='float').reshape(-1, 1)
	y_pred_A, sigma = gp.predict(x, return_std=True)
	np.savetxt( Direname + 'Predict_Y_'+ filename +'.dat', y_pred_A)

	diff = y_pred_A - y
	RMSE = np.sqrt( ( np.square(diff) ).mean() )#/4.5563E-6
	f.write('RMSE(C):  %.9f\n'%RMSE )

	f.write('----------------------------------------------------------------\n')
	f.close()
	
def main():
	parser = argparse.ArgumentParser(description='skGP imidazole')
	parser.add_argument('--n', type=int, default=1000, help='number of points')
	args = parser.parse_args()
	Niter = args.n
	
	GPR(Niter)
	print('DONE')
	print('--------------')
	
if __name__ == "__main__":
    main()	