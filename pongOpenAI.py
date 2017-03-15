import numpy as np
import gym



#parameteres

batchSize=10
gamma=0.99 #discount factor for reward
decayRate=0.99
learningRate=1e-4
inputDimensions=80*80   #input dimensions 80*80 grid
resume=False            #resume from a checkpoint
rewardSum=0				#initialize reward rewardSum
hiddenNeurons=200
images,hiddenLayers,rewards,gradients=[],[],[],[]
episodeNumber=0
runningReward=None

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

#function to preprocess the image
def preprocessing(inputObservation):
	inputObservation=inputObservation[35:195]        #crop the image
	inputObservation=inputObservation[::2,::2,0]	 # downsample the image by a factor of two
	inputObservation[inputObservation==144]=0		 #erase the background
	inputObservation[inputObservation==109]=0		 #erase the background
	inputObservation[inputObservation!=0]=1			 #set everything else to open
	return inputObservation.astype(np.float).ravel() #convert from 80*80 to 1600*1 matrix


def forwardPass(x,model):
	h=np.dot(model['W1'],x)     #dot product of layer 1 with image matrix
	h[h<0]=0					#apply ReLU 
	logp=np.dot(model['W2'],h)
	p=sigmoid(logp)				#convert logp into probability using sigmoid function
	return p,h 					#return probability of taking action and hidden layer

def action(prob):
	randomProb=np.random.uniform() #generate random probability
	if randomProb<prob:
		return 2				#go up
	else:
		return 3				#go down

def discountRewards(rewards):
	discountedRewards=np.zeros_like(rewards)
	runningAdd=0

	for t in reversed(xrange(0,size.rewards.size)):
		if rewards[t]!=0:
			runningAdd=0  #reset the sum at game boundary
		runningAdd=runningAdd*gamma+rewards[t]
		discountedRewards[t]=runningAdd
	return discountedRewards

def normalizeRewards(rewards):
	return (rewards-np.mean(rewards))/np.std(rewards)


def backProp(hiddenLayers,gradients,images,model):
	dCdW2=np.dot(hiddenLayers.T,gradients).ravel()
	deltal2=np.outer(gradients,model['W2'])
	deltal2[hiddenLayers<=0]=0
	dCdW1=np.dot(deltal2.T,images)
	return {'W1':dCdW1,'W2':dCdW2}

#RMSProp http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
def updateWeights(model,gradientBuffer):

	for key,value in model.iteritems():
		g=gradientBuffer[key]
		rmsProp[key]=decayRate*rmsProp[key]+(1-decayRage)*g**2
		model[key]+=learningRate*g/(np.sqrt(rmsProp[k])+1e-5)
		gradientBuffer[k]=np.zeros_like(v)



def init():

	#if we have a checkpoint load it
	if resume:
		model=pickle.load(open('model.p','rb'))
	#else create a new model
	else:
		model={}
		model['W1']=np.random.randn(hiddenNeurons,inputDimensions)/np.sqrt(inputDimensions)    # initialize hidden layer
		model['W2']=np.random.randn(hiddenNeurons)/np.sqrt(hiddenNeurons)					   # initialize output layer
	env=gym.make("Pong-v0")   																   # set up the environment for pong
	observation=env.reset()
	previousImage=None


	while True:
		if render:env.render()

		currentImage=preprocessing(observation)				#process the current observation

		if previousImage is not None:
			image=currentImage-previousImage				#difference between previous and current screen
		else:
			image=np.zeros(inputDimensions)
		previousImage=currentImage  						#update previous image

		prob,h=forwardPass(image,model)
		action=action(prob)

		images.append(image)
		hiddenLayers.append(h)


		#give action a label
		if action==2:
			y=1
		else:
			y=0

		#loss function gradient  see http://cs231n.github.io/neural-networks-2/#losses
		gradients.append(y-prob)

		# perform the action
		observation, reward, done, info = env.step(action)
		rewardSum += reward

		rewards.append(reward)


		if done:        #episode finished
			episodeNumber+=1

			#save input, hidden layer and rewards for this episode
			episodeImage=np.vstack(images)
			episodeHiddenLayer=np.vstack(hiddenLayers)
			episodeGradient=np.vstack(gradients)
			episodeRewards=np.vstack(rewards)
			images,hiddenLayers,rewards,gradients=[],[],[],[]

			#get discounted rewards
			discountedEpisodeRewards=discountedRewards(episodeRewards)

			#standardize the rewards
			discountedEpisodeRewards=normalizeRewards(discountedEpisodeRewards)

			#update the gradient with discounted rewards
			episodeGradient*=discountedEpisodeRewards


			backProgGradient=backProp(episodeHiddenLayer,episodeGradient,episodeImage,model)

			for k in model:
				backProgGradientBuffer[k]+=backProgGradient[k] #add gradients over batch

			if episodeNumber%batchSize==0:  #if we hit the batch size update the weights
				updateWeights(model,backProgGradientBuffer)

			if runningReward is None:
				runningReward=rewardSum
			else:
				runningReward=runningReward*0.99+rewardSum*0.01
				print("resetting env. episode total reward was %f. running mean %f".format(rewardSum,runningReward))

			if episodeNumber%100==0:
				pickle.dump(model,open('model.p','wb'))
			rewardSum=0
			observation=env.reset()
			previousImage=None
		if reward!=0:
			print('episode %d: game finished , reward %f'.format(episodeNumber,reward))




print("Starting to play")
init()