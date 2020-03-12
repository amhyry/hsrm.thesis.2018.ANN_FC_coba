# coding=utf-8
import matplotlib.pyplot as plt
import os
import datetime


#Individual Imports
#from version7.MyMainAnns_Regularizer import MyArtificialNeuralNetwork
#import version7.Data as dta
#import version7.Library as mylib

#from version7.MyMainAnns_EarlyStopping import MyArtificialNeuralNetwork
from version14.MyMainAnns_Vers3 import MyArtificialNeuralNetwork

import version14.RawData as dta
import version14.Library as mylib
import threading
import numpy as np

'''
Created on 02.04.2018

@author: Arnold Riemer
'''

if __name__ == '__main__':
    pass
    eingabe = input("Geben Sie einen Titel ein\n")
    print(eingabe)
    pfad = datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")
    pfad = pfad + "_" + eingabe 
    os.makedirs(pfad)
    
    tensorboardlogging = False
    input = dta.MyInputData()
    
    #python -m tensorboard.main --logdir="H:\My Documents\_Thesis\Workspace\PKP_ANN\version7\20180418 11-31-48\ANN_4\train"
    
    
    myann3 = MyArtificialNeuralNetwork(3,input, pfad, tensorboardlogging)
    myann3.startTraining()
    myann4 = MyArtificialNeuralNetwork(4,input, pfad, tensorboardlogging)
    myann4.startTraining()
    
    print("fertig")
    #th = threading.Thread(target = myann3.startTraining())
    #th2 = threading.Thread(target = myann4.startTraining())
    
    #th.start()    
    #th2.start()
    '''
    param = Processing()
    param.edit()
    if param.type == 1:
        print("beenden")
        myann3.forced_button = True
    
    
    #th = threading.Thread(target = myann3)
    #th.start()
    '''
       
    myann5 = MyArtificialNeuralNetwork(5,input, pfad, tensorboardlogging)
    myann5.startTraining()
    myann6 = MyArtificialNeuralNetwork(6,input, pfad, tensorboardlogging)
    myann6.startTraining()
    myann7 = MyArtificialNeuralNetwork(7,input, pfad, tensorboardlogging)
    myann7.startTraining()
    myann8 = MyArtificialNeuralNetwork(8,input, pfad, tensorboardlogging)
    myann8.startTraining()
    myann9 = MyArtificialNeuralNetwork(9,input, pfad, tensorboardlogging)
    myann9.startTraining()
    myann10 = MyArtificialNeuralNetwork(10,input, pfad, tensorboardlogging)
    myann10.startTraining()
    myann11 = MyArtificialNeuralNetwork(11,input, pfad, tensorboardlogging)
    myann11.startTraining()
    print(myann3.output.train.y_.sum())
    print("Finished all ANNs, compiling output...")
    
    train_annSummed_line = [myann3.output.train.y_.sum(), myann4.output.train.y_.sum(), myann5.output.train.y_.sum(), myann6.output.train.y_.sum(), myann7.output.train.y_.sum(), myann8.output.train.y_.sum(), myann9.output.train.y_.sum(), myann10.output.train.y_.sum(), myann11.output.train.y_.sum()]
    train_realSummed_line = [myann3.output.train.y.sum(), myann4.output.train.y.sum(), myann5.output.train.y.sum(), myann6.output.train.y.sum(), myann7.output.train.y.sum(), myann8.output.train.y.sum(), myann9.output.train.y.sum(), myann10.output.train.y.sum(), myann11.output.train.y.sum()]
    
    valid_annSummed_line = [myann3.output.validation.y_.sum(), myann4.output.validation.y_.sum(), myann5.output.validation.y_.sum(), myann6.output.validation.y_.sum(), myann7.output.validation.y_.sum(), myann8.output.validation.y_.sum(), myann9.output.validation.y_.sum(), myann10.output.validation.y_.sum(), myann11.output.validation.y_.sum()]
    valid_realSummed_line = [myann3.output.validation.y.sum(), myann4.output.validation.y.sum(), myann5.output.validation.y.sum(), myann6.output.validation.y.sum(), myann7.output.validation.y.sum(), myann8.output.validation.y.sum(), myann9.output.validation.y.sum(), myann10.output.validation.y.sum(), myann11.output.validation.y.sum()]
    
    test_annSummed_line = [myann3.output.test.y_.sum(), myann4.output.test.y_.sum(), myann5.output.test.y_.sum(), myann6.output.test.y_.sum(), myann7.output.test.y_.sum(), myann8.output.test.y_.sum(), myann9.output.test.y_.sum(), myann10.output.test.y_.sum(), myann11.output.test.y_.sum()]
    test_realSummed_line = [myann3.output.test.y.sum(), myann4.output.test.y.sum(), myann5.output.test.y.sum(), myann6.output.test.y.sum(), myann7.output.test.y.sum(), myann8.output.test.y.sum(), myann9.output.test.y.sum(), myann10.output.test.y.sum(), myann11.output.test.y.sum()]
    
    mylib.plotPortfolioCompare( 'Trainingskurve', pfad+"/portfoliosicht.pdf" , [3,4,5,6,7,8,9,10,11],
                            train_annSummed_line, 'ANN_Trainingsdata', 
                            train_realSummed_line, 'IST_Trainingsdata',
                            valid_annSummed_line, 'ANN_Validierungsdata',
                            valid_realSummed_line, 'IST_Validierungsdata',
                            test_annSummed_line, 'ANN_Testsdata',
                            test_realSummed_line, 'IST_Testdata')
    
    mylib.plotPortfolio( 'Trainingskurve Trainingsdaten', pfad+"/portfoliosicht_Trainingsdaten.pdf" , [3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7,8,9,10,11,12],
                            train_annSummed_line, 'ANN', 
                            train_realSummed_line, 'Manipulated IST',
                            input.TrainLRSummed[26:38], 'Forecast',
                            input.TrainLRSummed[14:26], 'IST YTD',
                            input.TrainLRSummed[2:14], 'Genehmigt')
    
    mylib.plotPortfolio( 'Trainingskurve Validierungsdaten', pfad+"/portfoliosicht_Validierungsdaten.pdf" , [3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7,8,9,10,11,12],
                            valid_annSummed_line, 'ANN', 
                            valid_realSummed_line, 'Manipulated IST',
                            input.ValidLRSummed[26:38], 'Forecast',
                            input.ValidLRSummed[14:26], 'IST YTD',
                            input.ValidLRSummed[2:14], 'Genehmigt')
    
    mylib.plotPortfolio( 'Trainingskurve Testdaten', pfad+"/portfoliosicht_Testdaten.pdf" , [3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7,8,9,10,11,12],
                            test_annSummed_line, 'ANN', 
                            test_realSummed_line, 'Manipulated IST',
                            input.TestLRSummed[26:38], 'Forecast',
                            input.TestLRSummed[14:26], 'IST YTD',
                            input.TestLRSummed[2:14], 'Genehmigt')
    
    losses = []
    #losses.append(np.array( ["Monat", "Trainingsloss", "Validierungslos"," SummeMinima" ,  "Testlos" ] ))
    losses.append(np.array( [3, myann3.train_loss, myann3.valid_loss, myann3.train_loss + myann3.valid_loss , myann3.test_loss ] ))
    losses.append(np.array( [4, myann4.train_loss, myann4.valid_loss, myann4.train_loss + myann4.valid_loss , myann4.test_loss ] ))
    losses.append(np.array( [5, myann5.train_loss, myann5.valid_loss, myann5.train_loss + myann5.valid_loss , myann5.test_loss ] ))
    losses.append(np.array( [6, myann6.train_loss, myann6.valid_loss, myann6.train_loss + myann6.valid_loss , myann6.test_loss ] ))
    losses.append(np.array( [7, myann7.train_loss, myann7.valid_loss, myann7.train_loss + myann7.valid_loss , myann7.test_loss ] ))
    losses.append(np.array( [8, myann8.train_loss, myann8.valid_loss, myann8.train_loss + myann8.valid_loss , myann8.test_loss ] ))
    losses.append(np.array( [9, myann9.train_loss, myann9.valid_loss, myann9.train_loss + myann9.valid_loss , myann9.test_loss ] ))
    losses.append(np.array( [10, myann10.train_loss, myann10.valid_loss, myann10.train_loss + myann10.valid_loss , myann10.test_loss ] ))
    losses.append(np.array( [11, myann11.train_loss, myann11.valid_loss, myann11.train_loss + myann11.valid_loss , myann11.test_loss ] ))
     
    
    np.savetxt(pfad + '/all_losses.csv', mylib.convertForCSV(np.array(losses)), header="Monat;Trainingsloss;Validierungsloss;SummeMinima; TestLoss",  delimiter=';', fmt='%s')
 
    loss_sse = []
    #losses.append(np.array( ["Monat", "Trainingsloss", "Validierungslos"," SummeMinima" ,  "Testlos" ] ))
    loss_sse.append(np.array( [3, myann3.train_loss_sse, myann3.valid_loss_sse, myann3.train_loss_sse + myann3.valid_loss_sse , myann3.test_loss_sse ] ))
    loss_sse.append(np.array( [4, myann4.train_loss_sse, myann4.valid_loss_sse, myann4.train_loss_sse + myann4.valid_loss_sse , myann4.test_loss_sse ] ))
    loss_sse.append(np.array( [5, myann5.train_loss_sse, myann5.valid_loss_sse, myann5.train_loss_sse + myann5.valid_loss_sse , myann5.test_loss_sse ] ))
    loss_sse.append(np.array( [6, myann6.train_loss_sse, myann6.valid_loss_sse, myann6.train_loss_sse + myann6.valid_loss_sse , myann6.test_loss_sse ] ))
    loss_sse.append(np.array( [7, myann7.train_loss_sse, myann7.valid_loss_sse, myann7.train_loss_sse + myann7.valid_loss_sse , myann7.test_loss_sse ] ))
    loss_sse.append(np.array( [8, myann8.train_loss_sse, myann8.valid_loss_sse, myann8.train_loss_sse + myann8.valid_loss_sse , myann8.test_loss_sse ] ))
    loss_sse.append(np.array( [9, myann9.train_loss_sse, myann9.valid_loss_sse, myann9.train_loss_sse + myann9.valid_loss_sse , myann9.test_loss_sse ] ))
    loss_sse.append(np.array( [10, myann10.train_loss_sse, myann10.valid_loss_sse, myann10.train_loss_sse + myann10.valid_loss_sse , myann10.test_loss_sse ] ))
    loss_sse.append(np.array( [11, myann11.train_loss_sse, myann11.valid_loss_sse, myann11.train_loss_sse + myann11.valid_loss_sse , myann11.test_loss_sse ] ))
     
    
    np.savetxt(pfad + '/all_loss_sse.csv', mylib.convertForCSV(np.array(loss_sse)), header="Monat;Trainingsloss;Validierungsloss;SummeMinima; TestLoss",  delimiter=';', fmt='%s')
       
    
    
    '''
    plt.clf()
    plt.figure(figsize=(8, 5), dpi=200)
    line_1, = plt.plot([3,4,5,6,7,8,9,10,11], train_annSummed_line, color='green', markerfacecolor='blue', markersize=7, label='ANN_Trainingsdata')
    line_2, = plt.plot([3,4,5,6,7,8,9,10,11], train_realSummed_line, color='red', markerfacecolor='blue', markersize=7, label='IST_Trainingsdata')
    line_3, = plt.plot([3,4,5,6,7,8,9,10,11], valid_annSummed_line, color='blue', markerfacecolor='blue', markersize=7, label='ANN_Validierungsdata')
    line_4, = plt.plot([3,4,5,6,7,8,9,10,11], valid_realSummed_line , color='yellow', markerfacecolor='blue', markersize=7, label='IST_Validierungsdata')
    line_5, = plt.plot([3,4,5,6,7,8,9,10,11], test_annSummed_line, color='pink', markerfacecolor='blue', markersize=7, label='ANN_Testsdata')
    line_6, = plt.plot([3,4,5,6,7,8,9,10,11], test_realSummed_line , color='orange', markerfacecolor='blue', markersize=7, label='IST_Testdata')
    plt.legend(handles=[line_1, line_2, line_3, line_4, line_5, line_6])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.axis([3, 12, 0, 100000])
    
      #plt.plot(a, 'g_', label="A", linewidth=2)
    #plt.plot(b, 'r_', label="B", linewidth=2)
    plt.ylabel(' Trainingskurve ')
    plt.grid(True)
    plt.savefig("portfoliosicht.pdf", bbox_inches='tight', format='pdf')
    plt.show()
    
    
    plt.clf()
    plt.figure(figsize=(8, 5), dpi=200)
    line_1, = plt.plot([3,4,5,6,7,8,9,10,11], train_annSummed_line, color='green', markerfacecolor='blue', markersize=7, label='ANN')
    line_2, = plt.plot([3,4,5,6,7,8,9,10,11], train_realSummed_line, color='red', markerfacecolor='blue', markersize=7, label='Manipulated IST')
    line_3, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvTrain.csvLRSummed[26:38], color='orange', markerfacecolor='blue', markersize=7, label='Forecast')
    line_4, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvTrain.csvLRSummed[14:26], color='yellow', markerfacecolor='blue', markersize=7, label='IST YTD')
    line_5, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvTrain.csvLRSummed[2:14] , color='pink', markerfacecolor='blue', markersize=7, label='Genehmigt' )
    plt.legend(handles=[line_1, line_2, line_3, line_4, line_5])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.ylabel(' Trainingskurve Trainingsdaten')
    plt.grid(True)
    plt.savefig("portfoliosicht_Trainingsdaten.pdf", bbox_inches='tight', format='pdf')
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(8, 5), dpi=200)
    line_1, = plt.plot([3,4,5,6,7,8,9,10,11], valid_annSummed_line, color='green', markerfacecolor='blue', markersize=7, label='ANN')
    line_2, = plt.plot([3,4,5,6,7,8,9,10,11], valid_realSummed_line, color='red', markerfacecolor='blue', markersize=7, label='Manipulated IST')
    line_3, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvValidierung.csvLRSummed[26:38], color='orange', markerfacecolor='blue', markersize=7, label='Forecast')
    line_4, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvValidierung.csvLRSummed[14:26], color='yellow', markerfacecolor='blue', markersize=7, label='IST YTD')
    line_5, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvValidierung.csvLRSummed[2:14] , color='pink', markerfacecolor='blue', markersize=7, label='Genehmigt')
    
    plt.legend(handles=[line_1, line_2, line_3, line_4, line_5])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(' Trainingskurve Validierungsdaten')
    plt.grid(True)
    plt.savefig("portfoliosicht_Validierungsdaten.pdf", bbox_inches='tight', format='pdf')
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(8, 5), dpi=200)
    line_1, = plt.plot([3,4,5,6,7,8,9,10,11], test_annSummed_line, color='green', markerfacecolor='blue', markersize=7, label='ANN')
    line_2, = plt.plot([3,4,5,6,7,8,9,10,11], test_realSummed_line, color='red', markerfacecolor='blue', markersize=7, label='Manipulated IST')
    line_3, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvTest.csvLRSummed[26:38], color='orange', markerfacecolor='blue', markersize=7, label='Forecast') # Forecast
    line_4, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvTest.csvLRSummed[14:26], color='yellow', markerfacecolor='blue', markersize=7, label='IST YTD') # IST
    line_5, = plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], myann3.input.csvTest.csvLRSummed[2:14], color='pink', markerfacecolor='blue', markersize=7, label='Genehmigt') #Genehmigt
    plt.legend(handles=[line_1, line_2, line_3, line_4, line_5])
    plt.ylabel(' Trainingskurve Testdaten')
    plt.grid(True)
    plt.savefig("portfoliosicht_Testdaten.pdf", bbox_inches='tight', format='pdf')
    plt.show()
    '''

    print("End of programm reached")
