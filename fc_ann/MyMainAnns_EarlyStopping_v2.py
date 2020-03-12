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
        globaleschritte = tf.Variable(0,dtype=tf.int32, name='Globalsteps')
        print(monthToRead)
        print("Reading the CSV input files.")
        self.input = data.initializeMonths(monthToRead)
        print(self.input.train.x.shape)
        
        print("Start Normalizing")
        self.input.normalize_all(mylib.NormLNFunction)
        print("Input Data Created.")
        
        print("Setting up variables and model...")
        x = tf.placeholder(tf.float32, [None, self.input.train.x[1].size] )
        y = tf.placeholder(tf.float32, [None, 1])
        a = tf.placeholder(tf.float32)
        b = tf.placeholder(tf.float32)        
        
        #offset = tf.fill(model.shape, 500)
        
        #Basis logger
        #logging.basicConfig(filename='example.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S ')
        os.makedirs(os.path.join(pfad, "ANN_" + str(monthToRead)))
        logging.basicConfig(filename=pfad + '/example.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S ')
        pfad = pfad + "/ANN_" + str(monthToRead)
        
        
        #set up model        
        with tf.name_scope('Model'):
            #model, w1, b1 = add_layer(x, self.input.train.x[1].size, 2, tf.tanh)#ohne Maske
            model, w1, b1 = mdl.add_layer(x, self.input.train.x[1].size, 2, tf.tanh, "Layer1")#mit Maske
            #keep_prob = tf.placeholder("float")
            #model = tf.nn.dropout(model, keep_prob)
            model, w2, b2 = mdl.add_layer(model, 2, 1, layer_name="Layer2")
            #model, w3, b3 = mdl.add_layer(model, 2, 1, tf.tanh, layer_name="Layer3")            
        beta = 1.0
        
        with tf.name_scope('loss'):
            
            #normalized_model = tf.multiply(tf.log(tf.add(model,  tf.constant( 500, dtype=tf.float32) ) ), tf.constant( 1/10, dtype=tf.float32))
            y_denorm = (tf.exp(10*y)-500)*((b-a)/11)
            model_denorm = (tf.exp(10*model)-500)*((b-a)/11)
            #denorm (np.exp(10*elem)-offset)*((b-a)/11) 
            #norm (1/10)*np.log(elem*((b-a)/11) + offset)
               
            #normalized_model = model
            loss = tf.reduce_sum(tf.square(model_denorm - y_denorm))
            #regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)# + tf.nn.l2_loss(w3)
            #loss = tf.reduce_sum(loss + beta * regularizer)
            mdl.variable_summaries(loss)
        #train_step is the parameter which has to be called for an optimizing step of the weights specified in the neural network, it contains the optimizer and the activationfunction
        with tf.name_scope('Learningrate'):
            learning_rate = tf.train.exponential_decay(0.01,                # Base learning rate.
                                                       globaleschritte,  # Current index into the dataset.
                                                       10000,            # Decay step.
                                                       0.95,                # Decay rate.
                                                       staircase=True)
            mdl.variable_summaries(learning_rate)
        with tf.name_scope('Trainingssteps'):
            train_step = tf.train.AdamOptimizer(1e-4,name='Adam').minimize(loss, globaleschritte)
        #Create a modelsaver for saving a model at a specific point
        saver = tf.train.Saver()
        
        print("Ann initialized.")
    
        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(pfad + '/train',
                                      sess.graph)
        test_writer = tf.summary.FileWriter(pfad + '/test')
        sess.run(tf.global_variables_initializer())
        
        
#==============================================================================
        
        print(x.get_shape())
        print("Start of training.")
        
        errors_trainingdata = []
        errors_validationdata = []
        logger_array_training = []
        logger_array_test = []
        patience_array = []
        
        patience = 20
        min_delta = 0.0001
        i = 0
        patience_cnt = 0
        counter = 20
        errors_validationdata.append(0.0000000000)
        #, a: self.input.validation.a, b: self.input.validation.b
        absoluteMinimum = loss.eval(session=sess, feed_dict={x: self.input.validation.x, y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b})
        
        """ 
=================================================================================================================
Start:
Training des Neuronalen Netzes
=================================================================================================================
        """    
        #for epoch in range(100000):
        while True:
            i += 1
            summary, curr_trainstep, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2 = sess.run([merged, train_step, loss, w1, w2, b1, b2], feed_dict={x: self.input.train.x, y: self.input.train.y, a: self.input.train.a, b: self.input.train.b})
            #curr_trainstep, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2 = sess.run([train_step, loss, w1, w2, b1, b2], feed_dict={x: self.input.train.x, y: self.input.train.y})
            if tensorboardlogging:
                train_writer.add_summary(summary, i)
            
            curr_loss_valid = loss.eval(session=sess, feed_dict={x: self.input.validation.x, y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b})
            
            errors_trainingdata.append(curr_loss)
            patience_array.append(patience_cnt)
            #logger_array_training.append([curr_loss, curr_loss_valid, curr_trainstep, curr_w1, curr_w2, curr_b1, curr_b2])
            
            
            #finde absolutes Minimum, f�hre saverfunktion durch.
            if absoluteMinimum > curr_loss_valid+curr_loss:
                absoluteMinimum = curr_loss_valid+curr_loss
                storedw1 = curr_w1
                storedw2 = curr_w2
                storedb1 = curr_b1
                storedb2 = curr_b2
                storedAtStep = i
                      
            #falls konstanter Anstieg oder loss 10% h�her als absolutes Minimum, -> Early Stopping
            #elif patience_cnt >= patience or (i > 5000 and (curr_loss_valid*1.20) > absoluteMinimum):
            
            if (i > 5000 and (curr_loss_valid) > absoluteMinimum*1.20):
            #if i > 100000:
                pass
                #print("early stopping...")
                #lade Model f�r Testing mit den werten beim besten Optimum
                logger_array_test.append([storedAtStep, absoluteMinimum, storedw1, storedw2, storedb1, storedb2]) 
                model_testing, w1_test, b1_test = mdl.add_layer_fixed(x, storedw1, storedb1, tf.tanh)
                model_testing, w2_test, b2_test = mdl.add_layer_fixed(model_testing, storedw2, storedb2)
                y_denorm = (tf.exp(10*y)-500)*((b-a)/11)
                model_testing_denorm = (tf.exp(10*model_testing)-500)*((b-a)/11)                
                loss_test = tf.reduce_sum(tf.square(model_testing_denorm - y_denorm))
                break                

            #falls der Validationloss anf�ngt zu steigen, oder konstant gr��er als das absolute minimum bleibt, fange an den Patience Wert hochzuz�hlen
            if curr_loss_valid >= errors_validationdata[-1]:
                pass
                patience_cnt += 1
            else:
                patience_cnt = 0
            
            errors_validationdata.append(curr_loss_valid)   

                #save_path = saver.save(sess, "/tmp/model.ckpt")
                #print("Model saved in path: %s" % save_path)
            
            if i % 1000 == 0:
                #print(curr_w1)
                logging.info('step %d, Current Validationloss %g > validationloss-1 %g Patience: %g ' % (i, curr_loss_valid, errors_validationdata[-1], patience_cnt))
                logging.info('step %d, evaluation training accuracy: %g real training loss: %g Patience: %g ' % (i, curr_loss_valid, curr_loss, patience_cnt))
                print('step %d, evaluation training accuracy: %g real training loss: %g Patience: %g ' % (i, curr_loss_valid, curr_loss, patience_cnt))
                
                logger_array_training.append([i, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2])
                
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
        save_path = saver.save(sess, pfad + "/Saved/model.ckpt")
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
        
        self.output.train.y_ = sess.run(model_testing, feed_dict={x:  self.output.train.x})
        self.output.validation.y_ = sess.run(model_testing, feed_dict={x:  self.output.validation.x})
        self.output.test.y_ = sess.run(model_testing, feed_dict={x:  self.output.test.x})
              
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
        
        
        
        vctf_train, vctf_validation, vctf_test = mylib.writeInFile(pfad, self.output, errors_trainingdata, errors_validationdata)
        
        logging.info('------------------------------')
        logging.info('------------------------------------------------------------------------------------------')
        logging.info('------------------------------')        
        
        logging.info("Ann mit %g Monaten als Input " %(monthToRead))
        logging.info('Steps: %g Traings_Loss: %g Validierungs_Loss: %g ' % (i, curr_loss, curr_loss_valid))
        logging.info('Mittelwertabweichung - Train: %g Eval: %g Test: %g ' % (vctf_train.aboluteDifferenceMean(),vctf_validation.aboluteDifferenceMean(),vctf_test.aboluteDifferenceMean()))
        logging.info('Gesamtabweichung - Train: %g Eval: %g Test: %g ' % (vctf_train.aboluteDifferenceSum(),vctf_validation.aboluteDifferenceSum(),vctf_test.aboluteDifferenceSum()))
        
        logging.info('Portfoliowert Train: %g ANNS: %g ' % (DifferenceSum(self.output.train.y),DifferenceSum(self.output.train.y_)))
        logging.info('Portfoliowert Valid: %g ANNS: %g ' % (DifferenceSum(self.output.validation.y),DifferenceSum(self.output.validation.y_)))
        logging.info('Portfoliowert Test: %g  ANNS: %g ' % (DifferenceSum(self.output.test.y),DifferenceSum(self.output.test.y_)))
        
        logging.info("Die Losswerte der Optimalen Funktion:")
        logging.info(loss_test.eval(session=sess, feed_dict={x: self.input.train.x, y: self.input.train.y, a: self.input.train.a, b: self.input.train.b}))
        logging.info(loss_test.eval(session=sess, feed_dict={x: self.input.validation.x, y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b}))
        logging.info(loss_test.eval(session=sess, feed_dict={x: self.input.test.x, y: self.input.test.y, a: self.input.test.a, b: self.input.test.b}))
        
        logging.info('----------------------------- ')
        logging.info('Trainingshistory: %s ' % (logger_array_training))
        logging.info('------------------------------')
        logging.info('------------------------------------------------------------------------------------------')
        logging.info('------------------------------')
        logging.info('Optimum: %s ' % (logger_array_test))
        
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
        mylib.plotTheLoss(errors_trainingdata,errors_validationdata, pfad + '/loss.pdf')
        mylib.plotTheLoss(np.log(errors_trainingdata),np.log(errors_validationdata), pfad + '/logn_loss.pdf')
        mylib.plotThePatience(patience_array, pfad + '/patience.pdf')
        
        mylib.plotTheOutput(self.output.train.y, self.output.train.y_, pfad + '/train.pdf')
        mylib.histogram(vctf_train.difference, pfad + '/train_hist.pdf')
        
        mylib.plotTheOutput(self.output.validation.y, self.output.validation.y_, pfad + '/validation.pdf')
        mylib.histogram(vctf_validation.difference, pfad + '/validation_hist.pdf')
        
        mylib.plotTheOutput(self.output.test.y, self.output.test.y_, pfad + '/test.pdf')
        mylib.histogram(vctf_test.difference, pfad + '/test_hist.pdf')
        
        tf.Session.close(sess)
        print("End of ANN reached")
        print("tensorboard --logdir=\"" + os.path.dirname(os.path.realpath(__file__)) + "\\" + pfad.replace("/","\\") + '\\train\"')
