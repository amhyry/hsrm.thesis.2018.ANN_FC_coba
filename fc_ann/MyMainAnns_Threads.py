# coding=utf-8
from datetime import datetime
import logging
#tensorboard --logdir DIRECTORY_PATH --debug
# tensorboard --logdir=/tmp/histogram_of_ann
# tensorboard --logdir=C:/Users/goose/eclipse-workspace/PP_ANN/version7/20180417 00-04-35/ANN_3/train --debug
#tensorboard --inspect --logdir="C:/Users/goose/eclipse-workspace/PP_ANN/version7/20180417 00-04-35/ANN_3/train"

from tensorflow.python import debug as tf_debug # pip3 install pyreadline
import tensorflow as tf
import numpy as np
import argparse
import sys
import math
import matplotlib.pyplot as plt
import datetime
import copy

import os

import version12.Modeller as mdl
import version12.Library as mylib
from version12.Dialogbox import Processing
#from thread import start_new_thread


np.set_printoptions(suppress=True)

'''

from MyRawData import MyInputData
from MyRawData import Maske
import MyRawData as myrd
import MyLibrary as mylib
from MyLibrary import VektorComparisionTF
'''
tf.logging.set_verbosity(tf.logging.INFO)

pathToTheSourceData = r'Rohdata\\'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def DifferenceSum(vektor):
    return vektor.sum()
    
def aboluteDifferenceMean(vektor):
    return vektor.mean()


class MyArtificialNeuralNetwork(object):
    """
        Class for initiating a ArtificialNeuralNetwork
    """
    def __init__(self, monthToRead, data, pfad=None, tensorboardlogging=False):
        tf.reset_default_graph()
        self.forced_button = False
        globaleschritte = tf.Variable(0,dtype=tf.int32, name='Globalsteps')
        self.monthToRead = monthToRead
        self.tensorboardlogging = tensorboardlogging
        print(self.monthToRead)
        print("Reading the CSV input files.")
        self.input = data.initializeMonths(monthToRead)
        print(self.input.train.x.shape)
        
        print("Start Normalizing")
        self.input.normalize_all(mylib.NormLNFunction)
        print("Input Data Created.")
        
        print("Setting up variables and model...")
        self.x = tf.placeholder(tf.float64, [None, self.input.train.x[1].size] )
        self.y = tf.placeholder(tf.float64, [None, 1])
        self.y_ = tf.placeholder(tf.float64, [None, 1])
        self.a = tf.placeholder(tf.float64, [None, 1])
        self.b = tf.placeholder(tf.float64, [None, 1])        
        
        #offset = tf.fill(model.shape, 500)
        
        #Basis logger
        #logging.basicConfig(filename='example.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S ')
        os.makedirs(os.path.join(pfad, "ANN_" + str(monthToRead)))
        logging.basicConfig(filename=pfad + '/example.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S ')
        self.pfad = pfad + "/ANN_" + str(monthToRead)
        
        #set up model        
        with tf.name_scope('Model'):
            #model, w1, b1 = add_layer(x, self.input.train.x[1].size, 2, tf.tanh)#ohne Maske
            
            self.model, self.w1, self.b1 = mdl.add_layer(self.x, self.input.train.x[1].size, 4, tf.tanh, "Layer1")
            #self.model, self.w1, self.b1 = mdl.add_layer(self.x, self.input.train.x[1].size, 4, tf.tanh, "Layer1", self.input.maske)#mit Maske
            #keep_prob = tf.placeholder("float")
            #model = tf.nn.dropout(model, keep_prob)
            self.model, self.w2, self.b2 = mdl.add_layer(self.model, 4, 1, layer_name="Layer2")
            #model, w3, b3 = mdl.add_layer(model, 2, 1, tf.tanh, layer_name="Layer3")            
        beta = 1.0
        
        
        
        with tf.name_scope('loss'):
            #normalized_model = tf.multiply(tf.log(tf.add(model,  tf.constant( 500, dtype=tf.float32) ) ), tf.constant( 1/10, dtype=tf.float32))
            self.y_denorm = tf.multiply(tf.div(tf.subtract(self.b,self.a), 11), tf.exp(10*self.y)-10 )
            
            #y_denorm = (tf.exp(10*y)-500)*((b-a)/11)
            self.model_denorm = tf.multiply((tf.exp(10*self.model)-10),tf.div(tf.subtract(self.b,self.a),11))
            #denorm (np.exp(10*elem)-offset)*((b-a)/11) 
            #norm (1/10)*np.log(elem*((b-a)/11) + offset)
               
            #normalized_model = model
            self.loss = tf.reduce_sum(tf.square(self.model_denorm - self.y_denorm))
            #regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)# + tf.nn.l2_loss(w3)
            #loss = tf.reduce_sum(loss + beta * regularizer)
            mdl.variable_summaries(self.loss)
        
        #train_step is the parameter which has to be called for an optimizing step of the weights specified in the neural network, it contains the optimizer and the activationfunction
        with tf.name_scope('Learningrate'):
            self.learning_rate = tf.train.exponential_decay(0.01,                # Base learning rate.
                                                       globaleschritte,  # Current index into the dataset.
                                                       10000,            # Decay step.
                                                       0.95,                # Decay rate.
                                                       staircase=True)
            mdl.variable_summaries(self.learning_rate)
        with tf.name_scope('Trainingssteps'):
            self.train_step = tf.train.AdamOptimizer(1e-4,name='Adam').minimize(self.loss, globaleschritte) 
        with tf.name_scope('Measures'):
            self.y__denorm = tf.multiply(tf.div(tf.subtract(self.b,self.a), 11), tf.exp(10*self.y_)-10 )
            self.mse = tf.metrics.mean_squared_error(labels=self.y__denorm, predictions=self.model_denorm, name="MSE")
            self.mae = tf.metrics.mean_absolute_error(labels=self.y__denorm, predictions=self.model_denorm, name="MAE") 
        
        #Create a modelsaver for saving a model at a specific point
        self.saver = tf.train.Saver()
        
        print("Ann initialized.")
    
        self.sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.pfad + '/train',
                                      self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.pfad + '/test')
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        
#==============================================================================
        print("Start of training.")
        
        self.errors_trainingdata = []
        self.errors_validationdata = []
        self.logger_array_training = []
        self.logger_array_test = []
        self.patience_array = []
        
        self.loss_delta_counter = 500
        
        self.loss_delta_minimum = 9999999
        
        self.patience = 20
        self.min_delta = 0.0001
        self.i = 0
        self.patience_cnt = 0
        self.counter = 20
        self.errors_validationdata.append(0.0000000000)
        #, a: self.input.validation.a, b: self.input.validation.b
        self.absoluteMinimum = self.loss.eval(session=self.sess, feed_dict={self.x: self.input.validation.x, self.y: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b})
        self.loss_valid_absoluteMinimum = self.loss.eval(session=self.sess, feed_dict={self.x: self.input.validation.x, self.y: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b})
        """ 
=================================================================================================================
Start:
Training des Neuronalen Netzes
=================================================================================================================
        """    
        #for epoch in range(100000):
        
        '''
        forced_stopping = False
        param = Processing()
        param.edit()
        if param.type == 0:
            forced_stopping = False
        else:
            forced_stopping = True          
        '''
        #forced_button = ForcedButton()
        
        
    def startTraining(self):
        
        print(self.sess.run(self.w1))
        while True:
            self.i += 1
            summary, curr_trainstep, self.loss_train, curr_w1, curr_w2, curr_b1, curr_b2 = self.sess.run([self.merged, self.train_step, self.loss, self.w1, self.w2, self.b1, self.b2], feed_dict={self.x: self.input.train.x, self.y: self.input.train.y, self.a: self.input.train.a, self.b: self.input.train.b})
            
            self.loss_valid = self.sess.run(self.loss, feed_dict={self.x: self.input.validation.x, self.y: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b})
                
            if self.tensorboardlogging:
                self.train_writer.add_summary(self.summary, self.i)
            
            if self.loss_delta_minimum < (self.loss_valid - self.loss_train):
                pass
                self.loss_delta_minimum = self.loss_valid - self.loss_train
            else:
                pass
                self.loss_delta_counter -= 1
            
            self.errors_trainingdata.append(self.loss_train)
            self.patience_array.append(self.patience_cnt)
            #logger_array_training.append([curr_loss, curr_loss_valid, curr_trainstep, curr_w1, curr_w2, curr_b1, curr_b2])
            
            
            #finde absolutes Minimum, f�hre saverfunktion durch.
            if self.loss_valid_absoluteMinimum > self.loss_valid:# and loss_delta_minimum > (loss_valid - loss_train):
            #if absoluteMinimum > loss_valid and loss_delta_minimum > (loss_valid - loss_train):
                self.loss_valid_absoluteMinimum = self.loss_valid
                storedw1 = curr_w1
                storedw2 = curr_w2
                storedb1 = curr_b1
                storedb2 = curr_b2
                storedAtStep = self.i
                      
            #falls konstanter Anstieg oder loss 10% h�her als absolutes Minimum, -> Early Stopping
            #elif patience_cnt >= patience or (i > 5000 and (curr_loss_valid*1.20) > absoluteMinimum):
            elif (self.i > 5000 and (self.loss_valid) > self.loss_valid_absoluteMinimum*1.50) or self.i >= 1000000:# or loss_delta_counter == 0:
            #if i > 5000:
                pass
                self.patience_cnt += 1
                
            else:
                self.patience_cnt = 0
                
                #print("early stopping...")
                #lade Model f�r Testing mit den werten beim besten Optimum
                
                
            if self.patience_cnt > self.patience or self.forced_button == True:
                print(storedw1)
                #self.logger_array_test.append([storedAtStep, self.absoluteMinimum, storedw1, storedw2, storedb1, storedb2]) 
                
                 
                updatew1 = tf.assign(self.w1, storedw1)
                updatew2 = tf.assign(self.w2, storedw2)
                updateb1 = tf.assign(self.b1, storedb1)
                updateb2 = tf.assign(self.b2, storedb2)                
                
                #updatew1 = w1.assign(storedw1)
                #updatew2 = w2.assign(storedw2)
                #updateb1 = b1.assign(storedb1)
                #updateb2 = b2.assign(storedb2)
                self.sess.run([updatew1, updatew2])
                #self.sess.run([updatew1, updatew2, updateb1, updateb2])
                print(self.sess.run(self.w1))
                
                break                
            '''
            #falls der Validationloss anf�ngt zu steigen, oder konstant gr��er als das absolute minimum bleibt, fange an den Patience Wert hochzuz�hlen
            if loss_valid >= errors_validationdata[-1]:
                pass
                patience_cnt += 1
            else:
                patience_cnt = 0
            '''
            self.errors_validationdata.append(self.loss_valid)   

                #save_path = saver.save(sess, "/tmp/model.ckpt")
                #print("Model saved in path: %s" % save_path)
            
            if self.i % 1000 == 0:
                print(self.w1) #Tensor("Model/Layer1/Mul:0", shape=(12, 2), dtype=float64)
                          #<tf.Variable 'Model/Layer1/weights/Variable:0' shape=(12, 2) dtype=float64_ref>
                #print(curr_w1)
                logging.info('step %d, Current Validationloss %g > validationloss-1 %g Patience: %g ' % (self.i, self.loss_valid, self.errors_validationdata[-1], self.patience_cnt))
                logging.info('step %d, evaluation training accuracy: %g real training loss: %g Patience: %g ' % (self.i, self.loss_valid, self.loss_train, self.patience_cnt))
                print('step %d, evaluation training accuracy: %g real training loss: %g Patience: %g ' % (self.i, self.loss_valid, self.loss_train, self.patience_cnt))
                #print(sess.run(model_denorm, feed_dict={x: self.input.validation.x, a: self.input.validation.a, b: self.input.validation.b}))
                #print(sess.run(y_denorm, feed_dict={y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b}))
                




                mse_train = self.sess.run(self.mse, feed_dict={self.x: self.input.train.x, self.y_: self.input.train.y, self.a: self.input.train.a, self.b: self.input.train.b})

                mse_valid = self.sess.run(self.mse, feed_dict={self.x: self.input.validation.x, self.y_: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b})
                
                mse_test = self.sess.run(self.mse, feed_dict={self.x: self.input.test.x, self.y_: self.input.test.y, self.a: self.input.test.a, self.b: self.input.test.b})
                
                mae_train = self.sess.run(self.mse, feed_dict={self.x: self.input.train.x, self.y_: self.input.train.y, self.a: self.input.train.a, self.b: self.input.train.b})

                mae_valid = self.sess.run(self.mse, feed_dict={self.x: self.input.validation.x, self.y_: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b})
                
                mae_test = self.sess.run(self.mse, feed_dict={self.x: self.input.test.x, self.y_: self.input.test.y, self.a: self.input.test.a, self.b: self.input.test.b})
                
                print(mse_train)
                print(mse_valid)
                print(mse_test)
                print(mae_train)
                print(mae_valid)
                print(mae_test)
                
                self.logger_array_training.append([self.i, self.loss_train, curr_w1, curr_w2])
                #self.logger_array_training.append([self.i, self.loss_train, curr_w1, curr_w2, curr_b1, curr_b2])
                
        #logging.info('step %d, evaluation training accuracy: %g real training loss: %g Patience: %g ' % (i, train_accuracy, curr_loss, patience_cnt))
        print("Training completed.")
        ''' 
=================================================================================================================
Ende:
Training des Neuronalen Netzes


Start: 
Loggingfunktionen, Evaluierungsfunktionen
=================================================================================================================
        '''          
        #sess.run(tf.global_variables_initializer())
        save_path = self.saver.save(self.sess, self.pfad + "/Saved/model.ckpt")
        print("Model saved in path: %s" % save_path)
        
        '''
=================================================================================================================
Ende:
Loggingfunktionen, Evaluierungsfunktionen


Start: 
Creating Output and Measure for perfomance measurement of the trained model
=================================================================================================================
        '''        
        self.output = copy.deepcopy(self.input)
        
        self.output.train.y_ = self.sess.run(self.model_denorm, feed_dict={self.x:  self.output.train.x, self.a: self.input.train.a, self.b: self.input.train.b})
        self.output.validation.y_ = self.sess.run(self.model_denorm, feed_dict={self.x:  self.output.validation.x, self.a: self.input.validation.a, self.b: self.input.validation.b})
        self.output.test.y_ = self.sess.run(self.model_denorm, feed_dict={self.x:  self.output.test.x, self.a: self.input.test.a, self.b: self.input.test.b})
              
        self.output.denormalize_all(mylib.DeNormLNFunction)

        self.output.deleteLast()
        '''
=================================================================================================================
Ende:
Measures


Start: 
Logging the training into files, setting ANN Variables for 
=================================================================================================================        
        '''
        
        print(self.output.train.y.shape)
        print(self.output.train.y_.shape)
        print(self.output.train.y_)

        np.savetxt(self.pfad + "/errors_los.csv", np.asarray(self.errors_trainingdata), delimiter=";",fmt='%10.5f')
        np.savetxt(self.pfad + "/errors_valid.csv", np.asarray(self.errors_validationdata), delimiter=";",fmt='%10.5f')
        
        np.savetxt(self.pfad + "/W1.csv", np.asarray(curr_w1), delimiter=";",fmt='%10.5f')
        np.savetxt(self.pfad + "/W2.csv", np.asarray(curr_w2), delimiter=";",fmt='%10.5f')
        np.savetxt(self.pfad + "/B1.csv", np.asarray(curr_b1), delimiter=";",fmt='%10.5f')
        np.savetxt(self.pfad + "/B1.csv", np.asarray(curr_b2), delimiter=";",fmt='%10.5f')        
        
        vctf_train, vctf_validation, vctf_test = mylib.writeInFile(self.pfad, self.output, self.errors_trainingdata, self.errors_validationdata)
        
        logging.info('------------------------------')
        logging.info('------------------------------------------------------------------------------------------')
        logging.info('------------------------------')        
        
        logging.info("Ann mit %g Monaten als Input " %(self.monthToRead))
        logging.info('Steps: %g Traings_Loss: %g Validierungs_Loss: %g ' % (self.i, self.loss_train, self.loss_valid))
        logging.info('Mittelwertabweichung - Train: %g Eval: %g Test: %g ' % (vctf_train.aboluteDifferenceMean(),vctf_validation.aboluteDifferenceMean(),vctf_test.aboluteDifferenceMean()))
        logging.info('Gesamtabweichung - Train: %g Eval: %g Test: %g ' % (vctf_train.aboluteDifferenceSum(),vctf_validation.aboluteDifferenceSum(),vctf_test.aboluteDifferenceSum()))
        
        logging.info('Portfoliowert Train: %g ANNS: %g ' % (DifferenceSum(self.output.train.y),DifferenceSum(self.output.train.y_)))
        logging.info('Portfoliowert Valid: %g ANNS: %g ' % (DifferenceSum(self.output.validation.y),DifferenceSum(self.output.validation.y_)))
        logging.info('Portfoliowert Test: %g  ANNS: %g ' % (DifferenceSum(self.output.test.y),DifferenceSum(self.output.test.y_)))
        
        logging.info("Die Losswerte der Optimalen Funktion:")
        logging.info(self.loss.eval(session=self.sess, feed_dict={self.x: self.input.train.x, self.y: self.input.train.y, self.a: self.input.train.a, self.b: self.input.train.b}))
        logging.info(self.loss.eval(session=self.sess, feed_dict={self.x: self.input.validation.x, self.y: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b}))
        logging.info(self.loss.eval(session=self.sess, feed_dict={self.x: self.input.test.x, self.y: self.input.test.y, self.a: self.input.test.a, self.b: self.input.test.b}))
        
        logging.info("Die Losswerte der Optimalen Funktion:")
        self.train_loss = self.loss.eval(session=self.sess, feed_dict={self.x: self.input.train.x, self.y: self.input.train.y, self.a: self.input.train.a, self.b: self.input.train.b})
        self.valid_loss = self.loss.eval(session=self.sess, feed_dict={self.x: self.input.validation.x, self.y: self.input.validation.y, self.a: self.input.validation.a, self.b: self.input.validation.b})
        self.test_loss = self.loss.eval(session=self.sess, feed_dict={self.x: self.input.test.x, self.y: self.input.test.y, self.a: self.input.test.a, self.b: self.input.test.b})
        
        
        logging.info('----------------------------- ')
        logging.info('Trainingshistory: %s ' % (self.logger_array_training))
        logging.info('------------------------------')
        logging.info('------------------------------------------------------------------------------------------')
        logging.info('------------------------------')
        logging.info('Optimum: %s ' % (self.logger_array_test))
        
        logging.info('-----------ENDE---------------')
        print("Logfiles successfully created.")
        #gr�n - Daten in den CSV Dateien / Rot - Erechnete Werte
        '''
        self.train_meandifference = vctf_train.aboluteDifferenceMean()
        self.valid_meandifference = vctf_validation.aboluteDifferenceMean()
        self.test_meandifference = vctf_test.aboluteDifferenceMean()
    
        self.train_sumdifference = vctf_train.aboluteDifferenceSum()
        self.valid_sumdifference = vctf_validation.aboluteDifferenceSum()
        self.test_sumdifference = vctf_test.aboluteDifferenceSum()
        
        self.train_annSummed = DifferenceSum(self.output.train.y_)
        self.valid_annSummed = DifferenceSum(self.output.validation.y_)
        self.test_annSummed = DifferenceSum(self.output.test.y_)
        
        self.train_realSummed = DifferenceSum(self.output.train.y)
        self.valid_realSummed = DifferenceSum(self.output.validation.y)
        self.test_realSummed = DifferenceSum(self.output.test.y)
        '''
        mylib.plotTheLoss(self.errors_trainingdata,self.errors_validationdata, self.pfad + '/loss.pdf')
        mylib.plotTheLoss(np.log(self.errors_trainingdata),np.log(self.errors_validationdata), self.pfad + '/logn_loss.pdf')
        mylib.plotThePatience(self.patience_array, self.pfad + '/patience.pdf')
        
        mylib.plotTheOutput(self.output.train.y, self.output.train.y_, self.pfad + '/train.pdf')
        mylib.histogram(vctf_train.difference, self.pfad + '/train_hist.pdf')
        
        mylib.plotTheOutput(self.output.validation.y, self.output.validation.y_, self.pfad + '/validation.pdf')
        mylib.histogram(vctf_validation.difference, self.pfad + '/validation_hist.pdf')
        
        mylib.plotTheOutput(self.output.test.y, self.output.test.y_, self.pfad + '/test.pdf')
        mylib.histogram(vctf_test.difference, self.pfad + '/test_hist.pdf')
        
        tf.Session.close(self.sess)
        print("End of ANN reached")
        print("tensorboard --logdir=\"" + os.path.dirname(os.path.realpath(__file__)) + "\\" + self.pfad.replace("/","\\") + '\\train\"')
