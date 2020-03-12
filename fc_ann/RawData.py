import csv
import numpy as np
import copy
'''
Created on 15.04.2018

@author: Arnold Riemer
'''

pathToTheSourceData = r'Rohdata\\'


class MyData(object):
    def __init__(self, x, y, a, b):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.y_ = None
        temp = []
        temp2 = []
        temp3 = []
        temp.append(y)
        temp2.append(a)
        temp3.append(b)
        self.y = np.array(temp).T
        self.a = np.array(temp2).T
        self.b = np.array(temp3).T
        

    def normalize(self, NormFunction=None):
        self.x = normalize(self.x, self.a, self.b, NormFunction)
        self.y = normalize(self.y, self.a, self.b, NormFunction)
        #if not self.y_ is None:
            #self.y_ = normalize(self.y_, self.a, self.b, NormFunction)
        
    def denormalize(self, NormFunction=None):
        self.x = denormalize(self.x, self.a, self.b, NormFunction)
        self.y = denormalize(self.y, self.a, self.b, NormFunction)
        #if not self.y_ is None:
            #self.y_ = denormalize(self.y_, self.a, self.b, NormFunction)
            
    def denormalizeOut(self, NormFunction=None):
        self.y_ = denormalize(self.y_, self.a, self.b, NormFunction)

    def deleteLast(self):
        #print(self.y)
        self.x = np.delete(self.x, self.x.T[0].size-1, 0)
        self.y = np.delete(self.y, self.y.T[0].size-1, 0)
        if not self.y_ is None:
            self.y_ = np.delete(self.y_, self.y_.T[0].size-1, 0)     

class MyInputData(object):
    '''
    Creates an Object Input with the Variables csvTrain, csvValidierung and csvTest, which contains sample DATA from given Path to source CSV Files
    This Object inherits additional Variables TrainLRSummed, ValidLRSummed and TestLRSummed, 
    which contains the last row in each CSV File, the last row should be the sum of each refering column.
    
    After initialization of an object of this class, the method initializeMonths should be called, otherwise the Variables train, validation and test will not be created.
    These Variables are not optional, because they contain the CSV Informations trimmed to the relevant Parameters.
    '''
    def __init__(self, source_path=pathToTheSourceData):
        self.source_path = source_path
        self.csvTrain       = convertCSV_Content(readCSV(source_path + "training.csv"))
        self.csvValidierung = convertCSV_Content(readCSV(source_path + "validierung.csv"))
        self.csvTest        = convertCSV_Content(readCSV(source_path + "test.csv"))    
        self.TrainLRSummed = self.csvTrain.T[:,-1]
        self.ValidLRSummed = self.csvValidierung.T[:,-1]
        self.TestLRSummed = self.csvTest.T[:,-1]
             
    def initializeMonths(self, monate):    
        x,y, a, b = self.createInput(self.csvTrain, monate)
        self.train =        copy.deepcopy(MyData(x,y,a,b))   

        x,y, a, b = self.createInput(self.csvValidierung, monate)
        self.validation =   copy.deepcopy(MyData(x,y,a,b))
        
        x,y, a, b = self.createInput(self.csvTest, monate)
        self.test =         copy.deepcopy(MyData(x,y,a,b))
        
        self.maske = returnContentAsArray(pathToTheSourceData + "maske_" + str(monate) +".csv")
        return copy.deepcopy(self)
    
    def createInput(self, csvContent, monthToRead=11):
        '''
        Returns a ndarray of given Month Parameters to read as X and array of IST Y
        '''
        csvContentforInput = self.deleteIrrelevantMonths(csvContent, monthToRead)
        a = csvContentforInput[:,0]
        b = csvContentforInput[:,1]
        
        x = np.delete(csvContentforInput, 0, 1)
        x = np.delete(x, 0, 1)
        #y = x.T
        y = x[:,23]
        deletedrows = 0
        for j in range(0,4):
            for i in range(monthToRead,12):
                index = i + j*12 #löschfunktion
                index = index - deletedrows
                x = np.delete(x, index, 1)
                deletedrows = deletedrows + 1
                #print(index)
        return x, y, a, b
    
    def deleteLast(self):
        self.train.deleteLast()
        self.validation.deleteLast()
        self.test.deleteLast()

    def normalize_all(self, NormFunction=None):
        self.train.normalize(NormFunction)
        self.validation.normalize(NormFunction)
        self.test.normalize(NormFunction)
       
    def denormalize_all(self, NormFunction=None):
        self.train.denormalize(NormFunction)
        self.validation.denormalize(NormFunction)
        self.test.denormalize(NormFunction)

    def denormalize_all_out(self, NormFunction=None):
        self.train.denormalizeOut(NormFunction)
        self.validation.denormalizeOut(NormFunction)
        self.test.denormalizeOut(NormFunction)


    def deleteIrrelevantMonths(self, x, monthToRead):
        newArr = []
        for i in x:
            if(i[0] <= monthToRead):
                newArr.append(i)
        return np.array(newArr)   
    
    def normalize(self, params, a, b, NormFunction=None):
        index = 0
        for elem in params:
            elem[...] = self.normalizeInRow(elem, a[index], b[index], NormFunction)
            index =+ 1
        return params
 
    def normalizeInRow(self, params, a, b, NormFunction):
        for elem in np.nditer(params, op_flags=['readwrite']):
            #elem[...] = np.tanh(elem/1000)
            elem[...] = NormFunction(elem, a, b)
            #elem[...] = np.tanh(elem/1000*((b-a)/11))
        return params


    def denormalize(self, params, a, b, NormFunction=None):
        index = 0
        for elem in params:
            elem[...] = self.denormalizeInRow(elem, a[index], b[index], NormFunction)
            index =+ 1
        return params
 
    def denormalizeInRow(self, params, a, b, NormFunction=None):
        for elem in np.nditer(params, op_flags=['readwrite']):
            #elem[...]=1000*np.arctanh(elem)
            elem[...] = NormFunction(elem, a, b)
            #print(elem)
            #elem[...] = 1000*np.arctanh(elem)/((b-a)/11)
        return params        
 
def readCSV(params):
    file = open(params, "r")
    csv_reader = csv.reader(file, delimiter=";")
    final_list = []
    for row in csv_reader:
        final_list.append(row)

    file.close() 
       
    newlist = []
    finallist = []
    for temp in final_list:
        newlist = [elems.replace(',','.') for elems in temp]
        finallist.append(newlist)
    return finallist

def convertCSV_Content(final_list):
    '''
    Creates a readable ndArray of type Float from the Content of given List.
    List Should look like Example in Projectlist.
    '''
    A = np.delete(final_list, 0, 0) # Delete first (title) Row
    A = np.delete(A, 0, 1) # Delete first (title) Column
    A = np.delete(A, 2, 0) # Delete Vert. Genehmigt Row
    A = np.delete(A, 14, 0) # Delete Ist Row
    A = np.delete(A, 26, 0) # Delete Forecast Row
    A = np.delete(A, 38, 0) # Delete Plan Row  
    listContent = final_list       
    for x in np.nditer(A.T, op_flags=['readwrite']):
        x[...] = 0.0 if x == '' else x        
    csvContent = A.T.astype(np.float)
    csvLRSummed = csvContent.T[:,-1]
    return csvContent
       
def normalize(params, a, b, NormFunction=None):
    index = 0
    for elem in params:
        elem[...] = normalizeInRow(elem, a[index], b[index], NormFunction)
        index =+ 1
    return params
  
def normalizeInRow(params, a, b, NormFunction=None):
    for elem in np.nditer(params, op_flags=['readwrite']):
        #elem[...] = np.tanh(elem/1000)
        elem[...] = NormFunction(elem, a, b)
        #elem[...] = np.tanh(elem/10000*((b-a)/11))
    return params

def denormalize(params, a, b, NormFunction=None):
    index = 0
    for elem in params:
        elem[...] = denormalizeInRow(elem, a[index], b[index], NormFunction)
        index =+ 1
    return params
 
def denormalizeInRow(params, a, b, NormFunction=None):
    for elem in np.nditer(params, op_flags=['readwrite']):
        #elem[...]=1000*np.arctanh(elem)
        elem[...] = NormFunction(elem, a, b)
        #elem[...] = 10000*np.arctanh(elem)/((b-a)/11)
    return params

def returnContentAsArray(params):
    file = open(params, "r")
    csv_reader = csv.reader(file, delimiter=";")
    final_list = []
    for row in csv_reader:
        final_list.append(row)
    file.close()    
    newlist = []
    finallist = []
    for temp in final_list:
        newlist = [elems.replace(',','.') for elems in temp]
        finallist.append(newlist)        
    return np.asarray(finallist, dtype=np.float)

class Maske(object):
    '''
    classdocs
    '''
    def __init__(self):
        self.mo = { 3 : returnContentAsArray(pathToTheSourceData + "maske_3.csv") , 
                    4 : returnContentAsArray(pathToTheSourceData + "maske_4.csv") ,
                    5 : returnContentAsArray(pathToTheSourceData + "maske_5.csv") ,
                    6 : returnContentAsArray(pathToTheSourceData + "maske_6.csv") ,
                    7 : returnContentAsArray(pathToTheSourceData + "maske_7.csv") ,
                    8 : returnContentAsArray(pathToTheSourceData + "maske_8.csv") ,
                    9 : returnContentAsArray(pathToTheSourceData + "maske_9.csv") ,
                    10 : returnContentAsArray(pathToTheSourceData + "maske_10.csv") ,
                    11 : returnContentAsArray(pathToTheSourceData + "maske_11.csv") }