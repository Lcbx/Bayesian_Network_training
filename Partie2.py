import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

digits=datasets.load_digits()

X=digits.data

y=digits.target
y_one_hot=np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1 #one hot target or shape NxK


X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42) #cuts dataset into trainingset (70%), rest (30%)

X_test, X_validation,y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42) #cuts rest into testset (15%), validationset(15%)


def softmax(x):
	 result = np.exp(x) / float(np.sum(np.exp(x)) + 1e-9)
	 return result
	 
def fprop(X,W):
	return softmax(np.dot(X, W.T))
	
	
def get_accuracy(X,y,W):
	y_pred = np.argmax(fprop(X,W), axis = 1)
	y_t = np.argmax(y, axis = 1)
	accuracy = np.mean(y_pred==y_t)
	return accuracy

def get_grads(y,y_pred,X):
	return np.dot((y - y_pred).T, X)

def get_loss(y,y_pred):
	return -1.0 * np.log(np.sum(y * y_pred))

def experiment(batch_size = 128, learning_rate = 0.0000003):
	
	W=np.random.normal(0,0.01,(len(np.unique(y)),X.shape[1])) # weights of shape KxL
	
	best_W = None
	best_accuracy = 0
	
	nb_epochs = 500
	minibatch_size = batch_size

	losses_validation=[]
	losses_train=[]

	for epoch in range(nb_epochs):

		for i in range (0 , X_train . shape [ 0 ] , minibatch_size ):
			
			minibatch_x = np.array(X_train[i:i+minibatch_size])
			minibatch_y = np.array(y_train[i:i+minibatch_size])

			y_pred = fprop(minibatch_x,W) 
			delta  = lr * get_grads(minibatch_y, y_pred, minibatch_x)
			W += delta
			
		# compute the losses
		
		y_pred = fprop(X_validation,W)
		loss_validation = get_loss(y_validation, y_pred)
		#print "loss_validation " + str(loss_validation)
		losses_validation.append(loss_validation) 
		
		y_pred = fprop(X_train,W)
		loss_train= get_loss(y_train, y_pred)
		#print "loss_train " + str(loss_train)
		losses_train.append(loss_train) 
		
		
		# compute the accuracy on the validation set
		accuracy=get_accuracy(X_validation, y_validation, W)
		#print "accuracy after batch : " + str(accuracy)
		
		# select the best parameters based on the validation accuracy
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_W = np.copy(W) 
				
	accuracy_on_unseen_data = get_accuracy(X_test,y_test, best_W)

	#print "accuracy on validation : " + str(accuracy_on_unseen_data) #0.897506925208

	#print "batch size  : " + str(minibatch_size) + " | learning rate : " + str(lr)

	plt.title("Batch size  : " + str(minibatch_size) + "  & Learning rate : " + str(lr) + " (best accuracy : " + str(best_accuracy)[:4] + ") ")
	plt.ylabel("Average negative log likelihood")
	plt.xlabel("Epoch")
	plt.plot(losses_train, label="train")
	plt.plot(losses_validation, label="validation")
	plt.legend()
	plt.savefig("bs"+str(minibatch_size)+"lr"+ str(lr) + ".png") #plt.show()
	plt.close()
	

batch_sizes = [32, 96, 256]
learning_rates = [0.00001, 0.000001, 0.0000001]	

for size in batch_sizes:
	for lr in learning_rates:
		experiment(size, lr)


	
	
	
	
	
	
	
	
	