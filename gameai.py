"""
Huba Ferenc Benzar
201262833
"""
#Importing packages
import os                               #OS
import os.path                          #path from OS
import gym                              #OpenAI Gym
import random                           #initially let the agent run randomly
import tflearn                          #tensorflow learn
import numpy as np                      #matrix math
from tflearn.layers.estimator import regression #final layer
from tflearn.layers.core import input_data, dropout, fully_connected #input layer, drop out like 20%, fully connected layer instead of CNN
from sklearn.externals import joblib    #joblit from sklearn externals
from collections import Counter         #Counter from collections
from statistics import median, mean     #illustrate how random performed

#create data directory
log = 'log/'
if not os.path.exists(log):
        os.makedirs(log)
model = 'log/OpenAI_Model/'
if not os.path.exists(model):
        os.makedirs(model)

env = gym.make("CartPole-v0")           #import game "CartPole-v0"

learning_rate = 1e-3                    #LearningRate        
score_requirement = 70                  #learn from all random games that have a score of this number or greater
initial_games = 10000                   #if this is too high it will brute force the methods
episodes = 200                          #every frame the pole is balanced +1
env.reset()                             #starts environment


#Creating Neural Network
def neural_network(input_size):
        print("Creating Model...")
        network = input_data(shape=[None, input_size, 1], name = "input") #input size is 4 in this case but better if i wanted to switch games
        #if there is an error for failed to do something with memory, make the nodes smaller
        #Tree layer
        #1 layer
        network = fully_connected(network, 128, activation = "relu") #takes input as network, 128 nodes on the layer, actication layer is rectufied linear
        network = dropout(network, 0.8) #0.8 is keeprate not dropout rate
        #2 layer
        network = fully_connected(network, 256, activation = "relu") #takes input as network, 256 nodes on the layer, actication layer is rectufied linear
        network = dropout(network, 0.8) #0.8 is keeprate not dropout rate
        #3 layer
        network = fully_connected(network, 512, activation = "relu") #takes input as network, 512 nodes on the layer, actication layer is rectufied linear
        network = dropout(network, 0.8) #0.8 is keeprate not dropout rate
        #4 layer
        network = fully_connected(network, 256, activation = "relu") #takes input as network, 256 nodes on the layer, actication layer is rectufied linear
        network = dropout(network, 0.8) #0.8 is keeprate not dropout rate
        #5 layer
        network = fully_connected(network, 128, activation = "relu") #takes input as network, 128 nodes on the layer, actication layer is rectufied linear
        network = dropout(network, 0.8) #0.8 is keeprate not dropout rate
        #output layer takes 2 unique outputs, change this number
        network = fully_connected(network, 2, activation="softmax")
        #network regression with the optimizer Adam, learning rate is the default, loss is categorical crosscentropy
        network = regression(network, optimizer="Adam", learning_rate = learning_rate,
                             loss = "categorical_crossentropy", name="targets")
        #model is tflearn deep neural network on the network creating a new directory called log
        model = tflearn.DNN(network, tensorboard_dir=log)
        
        #simple model not trained
        print("Model Created")
        return model

#Training Model
def train_model(training_data, model=False):
        print("Training Model...")
        #error if there is no training data
        if (len(training_data) < 1):
                print("Error no training data!")
        else:
                #x data
                #numpy array i 0th for i in training data which contains observations, actions. Reshape to -1 length of training data and to whatever shape it is
                x = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
                #y data
                #same as x  data but it's 1st in the list comprehension
                y = [i[1] for i in training_data]

                #if there is no model create a new model of the length of x
                if not model:
                        model = neural_network(input_size = len(x[0]))
                #model fit takes input which is our x and y data, number of epochs too many will overfit too many is a problem.
                #snapshot step is 500 show metric is true, run id creates a folder where this model is strored
                model.fit({'input': x}, {'targets': y}, n_epoch=3, snapshot_step=500 , show_metric=True, run_id="OpenAI_Model")
                
                #return trained model
                print("Trained Model")
                return model

def population():
        print("Creating Training Data...")
        #actual data to train on observation and move made append data if it's above requested #Observation, Moves
        training_data = []              
        total_reward = []               #all scores
        accepted_scores = []            #total_reward that met the threshold
        #iterate through game
        for i in range(initial_games):
                score = 0               #score is set to 0
                game_memory = []        #moves specifically from this environment
                prev_obs = []           #previous observation that we saw
                #game
                for j in range(episodes):
                        #env.render() # Comment out if i want it to go faster, renders game
                        action = env.action_space.sample() #takes environment and takes a random action #easy to switch games with
                        obs, reward, done, info = env.step(action)
                        #pixel data (pole position, cart position), reward 1 or 0 info and other info #step takes action
            
                        if len(prev_obs) > 0:
                                game_memory.append([prev_obs, action])
                        prev_obs = obs
                        score += reward
                        if done: break
            
                if score >= score_requirement:                          #if score is equal or bigger than threshold proceeds
                        accepted_scores.append(score)                   #appends score let through
                        for data in game_memory:
                                #convert to one-hot output layer for the neural network
                                if data[1] == 1:
                                        output = [0,1]
                                elif data[1] == 0:
                                        output = [1,0]
                    
                                training_data.append([data[0], output]) #save training data
                                #print([data[0], output])               #testing output
                
                env.reset()                                             #reset environment to play again
                total_reward.append(score)                              #save total score for reward
        training_data_save = np.array(training_data)                    #later referencing
    
        if (len(accepted_scores) < 1):
                print("Error no number above",score_requirement,"in", initial_games, "initiations and",episodes ,"games.")
        else:
                print("Here are some details")
                print("Average accepted total_reward: ", mean(accepted_scores))
                print("Median accepted total_reward: ", median(accepted_scores))
                print(Counter(accepted_scores))
        
        #LOGGING
        fhand = open(log+'Training-Data.txt', 'w')
        fhand.write('Training Data\n')
        fhand = open(log+'Training-Data.txt', 'a')
        fhand.write(str(accepted_scores)+'\n')
        fhand.write('\nAverage accepted score: '+(str(mean(accepted_scores)))+'\nMedian accepted score: '+(str(median(accepted_scores))))
        fhand.close()
        np.save(log+"Training-Data", training_data_save)
        print("Training Data Created. Check the logs directory.")

        return training_data

#attempt to load in model, FAILED ATTEMPT
def load():
        path = ('.'+'/log/OpenAI_Model/')
        files = os.listdir(path)
        if (len(files)>0):
                decision = input("Please enter (y/Y) if you wish to load a file: ")    
                if decision.lower() == "y":
                        print("Hello, would you like to load a model from the following?")
                        for name in files:
                                print("\t"+name+"\t")
                        data = input("Please enter the file you wish to load:\n")
                        exist = (path+data)
                        if os.path.exists(exist) and os.access(exist, os.R_OK):
                                neural_network(input_size)
                                model = dnn.load(data)
                                trainain_model()
                                model=True
                                print("Loading: " , data)
                        else:
                               print("Error this file does not exist!\n")
                else:
                        print("Skipping to next step!\n")
        else:
           print()

#Playing Game
def game():
        print("Playing Game with Model...")
        #LOGGING
        fhand = open(log+'Testing-Trained-Model.txt', 'w')
        fhand.write('Iteration\tReward\n')
        fhand.close()
        #total_reward and choices are empty lists
        total_reward = []
        choices = []
        counter = 0
        #number of games we want to play
        for each_game in range(100):
                counter+=1
                score = 0 #total reward starts at 0
                game_memory = [] #creating empty lists
                prev_obs = []
                env.reset() #resets environment
                #iterate through the number of steps we want to make
                for i in range(episodes):
                        #env.render() #renders game
                        #if there is no previous observation we do a random move
                        if len(prev_obs) == 0:
                                action = env.action_space.sample()
                        else: #otherwise when a frame is seen the action
                                #argmax of model, we preduct the previous observation, which we reshape just like before and take 0th
                                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
                        choices.append(action) #we append all of the actions we make shows ratio of our network on what it oes
                    
                        new_obs, reward, done, info = env.step(action)  #new observation, reward, done, info is environemt step in action
                        prev_obs = new_obs #previous observation is new observation
                        game_memory.append([new_obs, action])   #for retraining the network
                        score += reward #reward added up
                        #ends
                        if done: break
                total_reward.append(score) #appends total_reward

                #LOGGING
                fhand = open(log+'Testing-Trained-Model.txt', 'a')
                fhand.write(str(counter) + '\t\t' + str(score) + '\n')
                fhand.close()

        #LOGGING AND PRINTING   
        print("Requirment to pass:\nOver 100 consecutive trials get over 195 reward.")
        print("Average reward over {} games:".format(len(total_reward)), sum(total_reward)/len(total_reward))
        print("Choice 1: {}, Choice 2: {}".format(choices.count(1)/len(choices), choices.count(0)/len(choices)))

        fhand = open(log+'Testing-Trained-Model.txt', 'a')
        fhand.write(str("\nAverage reward over {} games:".format(len(total_reward))+ str(sum(total_reward)/len(total_reward)))+
                    (str("\n\nChoice 1: {}\nChoice 2: {}".format(choices.count(1)/len(choices), choices.count(0)/len(choices)))+
                     (str("\n\nPass\t\tFail") + '\n')))

        if(sum(total_reward)/len(total_reward))> 195:
                print("Pass")
                fhand.write(str("Pass\t"))
                fhand.close()
        else:
                print("Fail")
                fhand.write(str("\t\tFail"))
                fhand.close()
    

#Executing
training_data = population()
model = train_model(training_data)
game()
