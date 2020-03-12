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
        x = tf.placeholder(tf.float64, [None, self.input.train.x[1].size] )
        y = tf.placeholder(tf.float64, [None, 1])
        y_ = tf.placeholder(tf.float64, [None, 1])
        a = tf.placeholder(tf.float64)
        b = tf.placeholder(tf.float64)        
        
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
            keep_prob = tf.placeholder("float64")
            model = tf.nn.dropout(model, keep_prob)
            model, w2, b2 = mdl.add_layer(model, 2, 1, layer_name="Layer2")
            #model, w3, b3 = mdl.add_layer(model, 2, 1, tf.tanh, layer_name="Layer3")            
        beta = 0.01
        
        with tf.name_scope('loss'):
            
            #normalized_model = tf.multiply(tf.log(tf.add(model,  tf.constant( 500, dtype=tf.float32) ) ), tf.constant( 1/10, dtype=tf.float32))
            #y_denorm = (tf.exp(10*y)-500)*((b-a)/11)
            #model_denorm = (tf.exp(10*model)-500)*((b-a)/11)
            y_denorm = tf.multiply(tf.div(tf.subtract(b,a), 11), tf.exp(10*y)-10 )
            model_denorm = tf.multiply((tf.exp(10*model)-10),tf.div(tf.subtract(b,a),11))
            #denorm (np.exp(10*elem)-offset)*((b-a)/11) 
            #norm (1/10)*np.log(elem*((b-a)/11) + offset)
               
            #normalized_model = model
            loss = tf.reduce_sum(tf.square(model_denorm - y_denorm))
            regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)# + tf.nn.l2_loss(w3)
            loss = tf.reduce_sum(loss + beta * regularizer)
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
        
        
        with tf.name_scope('Measures'):
            mse = tf.metrics.mean_squared_error(labels=y_, predictions=model_denorm, name="MSE")
        
        print("Ann initialized.")
    
        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(pfad + '/train',
                                      sess.graph)
        test_writer = tf.summary.FileWriter(pfad + '/test')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        
#==============================================================================
        
        print(x.get_shape())
        print("Start of training.")
        
        errors_trainingdata = []
        errors_validationdata = []
        logger_array_training = [] # np.asarray([])
        logger_array_test = []
        patience_array = []
        
        patience = 20
        min_delta = 0.0001
        i = 0
        patience_cnt = 0
        counter = 20
        errors_validationdata.append(0.0000000000)
        #, a: self.input.validation.a, b: self.input.validation.b
        absoluteMinimum = loss.eval(session=sess, feed_dict={x: self.input.validation.x, y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b, keep_prob:1})
        
        """ 
=================================================================================================================
Start:
Training des Neuronalen Netzes
=================================================================================================================
        """    
        for epoch in range(100000):
        #while True:
            i += 1
            summary, curr_trainstep, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2 = sess.run([merged, train_step, loss, w1, w2, b1, b2], feed_dict={x: self.input.train.x, y: self.input.train.y, a: self.input.train.a, b: self.input.train.b, keep_prob:0.75})
            #curr_trainstep, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2 = sess.run([train_step, loss, w1, w2, b1, b2], feed_dict={x: self.input.train.x, y: self.input.train.y})
            if tensorboardlogging:
                train_writer.add_summary(summary, i)
            
            curr_loss_valid = loss.eval(session=sess, feed_dict={x: self.input.validation.x, y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b, keep_prob:1})
            
            
            
            errors_trainingdata.append(curr_loss)
            patience_array.append(patience_cnt)

         

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
                
                #logger_array_training = np.append(logger_array_training, [i, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2])
                

                mse_train = sess.run(mse, feed_dict={x: self.input.train.x, y_: self.input.train.y, a: self.input.train.a, b: self.input.train.b, keep_prob:1})

                mse_valid = sess.run(mse, feed_dict={x: self.input.validation.x, y_: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b, keep_prob:1})
                
                mse_test = sess.run(mse, feed_dict={x: self.input.test.x, y_: self.input.test.y, a: self.input.test.a, b: self.input.test.b, keep_prob:1})
                
                print(mse_train)
                print(mse_valid)
                print(mse_valid)
                #print('step %d, Train Accuracy MSE: %e MAE: %f ' % (i, mse_train, mae_train))    
                #print('step %d, Train Accuracy MSE: %f MAE: %f ' % (i, mse_validation, mae_validation))
                #print('step %d, Train Accuracy MSE: %f MAE: %f ' % (i, mse_test, mae_test)) 
                
                
                logger_array_training.append([i, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2])
        
                #logger_array_test.append([storedAtStep, absoluteMinimum, storedw1, storedw2, storedb1, storedb2]) 

        logger_array_training.append([i, curr_loss, curr_w1, curr_w2, curr_b1, curr_b2])
        
        
        #updatew1 = w1.assign(curr_w1 * 0.75)
        #updatew2 = w2.assign(curr_w2 * 0.75)
        #updateb1 = b1.assign(curr_b1 * 0.75)
        #updateb2 = b2.assign(curr_b2 * 0.75)
        #sess.run([updatew1, updatew2, updateb1, updateb2])

        
        
    
        
        
        #mse = tf.metrics.mean_squared_error(labels=Y_test, predictions=Y_pred)        
        
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
        
        self.output.train.y_ = sess.run(model_denorm, feed_dict={x:  self.output.train.x, a: self.output.train.a, b: self.output.train.b, keep_prob:1})
        self.output.validation.y_ = sess.run(model_denorm, feed_dict={x:  self.output.validation.x, a: self.output.validation.a, b: self.output.validation.b, keep_prob:1})
        self.output.test.y_ = sess.run(model_denorm, feed_dict={x:  self.output.test.x, a: self.output.test.a, b: self.output.test.b, keep_prob:1})             
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
        
        vctf_train, vctf_validation, vctf_test = mylib.writeInFile(pfad, self.output, errors_trainingdata, errors_validationdata)
        
        np.savetxt(pfad + "/W1.csv", np.asarray(curr_w1), delimiter=";",fmt='%10.5f')
        np.savetxt(pfad + "/W2.csv", np.asarray(curr_w2), delimiter=";",fmt='%10.5f')
        np.savetxt(pfad + "/B1.csv", np.asarray(curr_b1), delimiter=";",fmt='%10.5f')
        np.savetxt(pfad + "/B1.csv", np.asarray(curr_b2), delimiter=";",fmt='%10.5f')
        
        #curr_w1.tofile(pfad + "/Gewichte.csv",sep=',',format='%10.5f')
        
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
        logging.info(loss.eval(session=sess, feed_dict={x: self.input.train.x, y: self.input.train.y, a: self.input.train.a, b: self.input.train.b, keep_prob:1}))
        logging.info(loss.eval(session=sess, feed_dict={x: self.input.validation.x, y: self.input.validation.y, a: self.input.validation.a, b: self.input.validation.b, keep_prob:1}))
        logging.info(loss.eval(session=sess, feed_dict={x: self.input.test.x, y: self.input.test.y, a: self.input.test.a, b: self.input.test.b, keep_prob:1}))
        
        logging.info('----------------------------- ')
        #logging.info('Trainingshistory: %s ' % (logger_array_training))
        
        
        logging.info('------------------------------')
        logging.info('------------------------------------------------------------------------------------------')
        logging.info('------------------------------')
        #logging.info('Optimum: %s ' % (logger_array_test))
        
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
