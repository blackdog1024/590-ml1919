# UNIVARIABLE REGRESSION EXAMPLE
#    -USING SciPy FOR OPTIMIZATION
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize
import func
from sklearn.model_selection import train_test_split
#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
model_type="logistic"
xcol=1; ycol=2;
NFIT=4

#HYPER-PARAM
OPT_ALGO='CG'


#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			#READ FILE
			with open(FILE_NAME) as f:
				self.input = json.load(f)  #read into dictionary

			#CONVERT INPUT INTO ONE LARGE MATRIX
				#SIMILAR TO PANDAS DF
			X=[];
			for key in self.input.keys():
				if(key in DATA_KEYS):
					X.append(self.input[key])

			#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
			self.X=np.transpose(np.array(X))

			#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
			self.XMEAN=np.mean(self.X,axis=0)
			self.XSTD=np.std(self.X,axis=0)

		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED");

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:", self.X.shape)
		print("X examples")
		print("X means:",np.mean(self.X,axis=0))
		print("X stds:",np.std(self.X,axis=0))

		#PRINT FIRST 5 SAMPLES
		for i in range(0,self.X.shape[1]):
			print("X column ",i,": ",self.X[0:5,i])

	def partition(self,f_train=0.825, f_val=0.15, f_test=0.025):
		#f_train=fraction of data to use for training

		#TRAINING: 	 DATA THE OPTIMIZER "SEES"
		#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
		#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)
		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0])
		CUT1=int(f_train*self.X.shape[0]);
		CUT2=int((f_train+f_val)*self.X.shape[0]);
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]

	def plot_xy(self,col1=1,col2=2,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			FS=18   #FONT SIZE
			ax.plot(self.X[:,col1], self.X[:,col2],'o') #,c=data['y'], cmap='gray')
			plt.xlabel(xla, fontsize=FS)
			plt.ylabel(yla, fontsize=FS)
			plt.show()

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD #/3.


#------------------------
#MAIN
#------------------------

#INITIALIZE DATA OBJECT
D=DataClass(INPUT_FILE)

#BASIC DATA PRESCREENING
D.report()
D.partition()
D.normalize()
D.report()

D.plot_xy(1,2,'age (years)','weight (lb)')
D.plot_xy(2,0,'weight (lb)','is_adult')


#------------------------
#DEFINE MODEL
#------------------------

def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#UN-NORMALIZE
def unnorm(x,col):
	return D.XSTD[col]*x+D.XMEAN[col]


#------------------------
#DEFINE LOSS FUNCTION
#------------------------
iteration=0;
iterationv=0;

#SAVE HISTORY FOR PLOTTING AT THE END
iterations=[];iterationvs=[]; loss_train=[];  loss_val=[]

def loss(p,xt,xv,yt,yv):
	global iteration,iterations,loss_train,loss_val

	#TRAINING LOSS
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration%25==0):
		print(iteration,training_loss,validation_loss) #,p)
	loss_train.append(training_loss)
	loss_val.append(validation_loss)
	iterations.append(iteration)

	iteration+=1

	return training_loss

def sto_losst(p,x,y):
	global iteration,iterations

	#LOSS
	yp=model(x,p) #model predictions for given parameterization p
	loss=(np.mean((yp-y)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration%25==0):
		print(iteration,loss) #,p)
	loss_train.append(loss)
	iterations.append(iteration)

	iteration+=1

	return loss
def sto_lossv(p,x,y):
	global iterationv,iterationvs

	#LOSS
	yp=model(x,p) #model predictions for given parameterization p
	loss=(np.mean((yp-y)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration%25==0):
		print(iteration,loss) #,p)
	loss_val.append(loss)
	iterationvs.append(iteration)
	iterationv += 1
	return loss
#------------------------
#FIT 1 MODEL
#------------------------


#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
p=np.random.uniform(0.5,1.,size=NFIT)

#SELECT DATA
#TRAINING
xt=D.X[:,xcol][D.train_idx]
yt=D.X[:,ycol][D.train_idx]
#VALIDATION
xv=D.X[:,xcol][D.val_idx]
yv=D.X[:,ycol][D.val_idx]
#TEST
xtest=D.X[:,xcol][D.test_idx]
ytest=D.X[:,ycol][D.test_idx]
#minibatch


#TRAIN MODEL USING different ways,I've annotated some of them,but all of them can use
algo = 'GD'
#algo = 'mom'
method = 'batch'
#method = 'mini-batch'
#method = 'stochastic'
if(method=='stochastic'):res = func.sto_optimizer(sto_losst,sto_lossv, p, algo,0.001,method,xt,xv,yt,yv)
else:res = func.optimizer(loss, p, algo,0.001,method,xt,xv,yt,yv)


#print('res:',res)
popt=res

print("OPTIMAL PARAM:",popt)

#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm(xt,xcol), unnorm(yt,ycol), 'o', label='Training set')
	ax.plot(unnorm(xv,xcol), unnorm(yv,ycol), 'x', label='Validation set')
	ax.plot(unnorm(xtest,xcol), unnorm(ytest,ycol), '*', label='Test set')
	ax.plot(unnorm(xm,xcol),unnorm(yp,ycol), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	if(algo == 'GD')&(method == 'batch'):plt.title("GD,batch,logistic regression")
	if (algo == 'GD') & (method == 'minibatch'): plt.title("GD,minibatch,logistic regression")
	if (algo == 'GD') & (method == 'stochastic'): plt.title("GD,stochastic,logistic regression")
	if (algo == 'mom') & (method == 'batch'): plt.title("mom,batch,logistic regression")
	if (algo == 'mom') & (method == 'minibatch'): plt.title("mom,minibatch,logistic regression")
	if (algo == 'mom') & (method == 'stochastic'): plt.title("mom,stochastic,logistic regression")
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	# ax.plot(yt, yt, '-', label='y_pred=y_data')

	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	if (algo == 'GD') & (method == 'batch'): plt.title("GD,batch,ypre-y")
	if (algo == 'GD') & (method == 'minibatch'): plt.title("GD,minibatch,ypre-y")
	if (algo == 'GD') & (method == 'stochastic'): plt.title("GD,stochastic,ypre-y")
	if (algo == 'mom') & (method == 'batch'): plt.title("mom,batch,ypre-y")
	if (algo == 'mom') & (method == 'minibatch'): plt.title("mom,minibatch,ypre-y")
	if (algo == 'mom') & (method == 'stochastic'): plt.title("mom,stochastic,lypre-y")
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS
if(method=='stochastic'):
	if(IPLOT):
		fig, ax = plt.subplots()
		#iterations,loss_train,loss_val
		ax.plot(iterations, loss_train, 'o', label='Training loss')
		plt.xlabel('optimizer iterations', fontsize=18)
		plt.ylabel('loss', fontsize=18)
		if (algo == 'GD'): plt.title("GD,stochastic,Training loss")
		if (algo == 'mom'): plt.title("mom,stochastic,Training loss")
		plt.legend()
		plt.show()
	if(IPLOT):
		fig, ax = plt.subplots()
		#iterations,loss_train,loss_val
		ax.plot(iterationvs, loss_val, 'o', label='Validation loss',color = 'orange')
		plt.xlabel('optimizer iterations', fontsize=18)
		plt.ylabel('loss', fontsize=18)
		if (algo == 'GD'): plt.title("GD,stochastic,Validation loss")
		if (algo == 'mom'): plt.title("mom,stochastic,Validation loss")
		plt.legend()
		plt.show()

else:
	if (IPLOT):
		fig, ax = plt.subplots()
		# iterations,loss_train,loss_val
		ax.plot(iterations, loss_train, 'o', label='Training loss')
		ax.plot(iterations, loss_val, 'o', label='Validation loss')
		plt.xlabel('optimizer iterations', fontsize=18)
		plt.ylabel('loss', fontsize=18)
		if (algo == 'GD') & (method == 'batch'): plt.title("GD,batch,Training loss and Validation loss")
		if (algo == 'GD') & (method == 'minibatch'): plt.title("GD,minibatch,Training loss and Validation loss")
		if (algo == 'mom') & (method == 'batch'): plt.title("mom,batch,Training loss and Validation loss")
		if (algo == 'mom') & (method == 'minibatch'): plt.title("mom,minibatch,Training loss and Validation loss")
		plt.legend()
		plt.show()
