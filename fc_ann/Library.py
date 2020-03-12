# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
'''
Created on 16.04.2018

@author: Arnold Riemer
'''


def convertForCSV(array):
    pass
    if True:
        temp = np.array(["%.2f" % w for w in array.reshape(array.size)])
        temp = np.array([s.replace('.' , ',') for s in temp])
        temp = temp.reshape(array.shape)
        return temp
    
    temp = ["%.2f" % x for x in array]
    return [s.replace('.' , ',') for s in temp]

def writeInFile(pfad, Data, train_loss, valid_loss):
    np.savetxt(pfad + '/train_loss.csv', convertForCSV(np.array(train_loss)), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/validation_loss.csv', convertForCSV(np.array(valid_loss)), delimiter=';', fmt='%s')

    np.savetxt(pfad + '/input_train_x.csv', convertForCSV(Data.train.x.T), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/input_validation_x.csv', convertForCSV(Data.validation.x.T), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/input_test_x.csv', convertForCSV(Data.test.x.T), delimiter=';', fmt='%s')
    
    np.savetxt(pfad + '/input_train_y.csv', convertForCSV(Data.train.y), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/input_validation_y.csv', convertForCSV(Data.validation.y), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/input_test_y.csv', convertForCSV(Data.test.y), delimiter=';', fmt='%s')
  
    np.savetxt(pfad + '/ann_train.csv', convertForCSV(Data.train.y_), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/ann_validation.csv', convertForCSV(Data.validation.y_), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/ann_test.csv', convertForCSV(Data.test.y_), delimiter=';', fmt='%s')

    vctf_train = VektorComparisionTF(Data.train.y_, Data.train.y)
    vctf_validation = VektorComparisionTF(Data.validation.y_, Data.validation.y)
    vctf_test = VektorComparisionTF(Data.test.y_, Data.test.y)
    np.savetxt(pfad + '/train_difference.csv', convertForCSV(vctf_train.difference), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/validation_difference.csv', convertForCSV(vctf_validation.difference), delimiter=';', fmt='%s')
    np.savetxt(pfad + '/test_difference.csv', convertForCSV(vctf_test.difference), delimiter=';', fmt='%s')    
    
    
    return vctf_train, vctf_validation, vctf_test

def plotPortfolioCompare( yname, pfad , 
                    monthsA,
                    a, label1, 
                    b, label2,
                    c, label3,
                    d, label4,
                    e, label5,
                    f, label6):
    pass
    plt.clf()
    plt.figure(figsize=(8, 5), dpi=200)
    
    line_1, = plt.plot(monthsA, a, color='red', linestyle=':', markerfacecolor='blue', markersize=7, label=label1)
    line_2, = plt.plot(monthsA, b, color='red', markerfacecolor='blue', markersize=7, label=label2)
    line_3, = plt.plot(monthsA, c, color='green', linestyle=':', markerfacecolor='blue', markersize=7, label=label3)
    line_4, = plt.plot(monthsA, d , color='green', markerfacecolor='blue', markersize=7, label=label4)
    line_5, = plt.plot(monthsA, e, color='blue', linestyle=':', markerfacecolor='blue', markersize=7, label=label5)
    line_6, = plt.plot(monthsA, f , color='blue', markerfacecolor='blue', markersize=7, label=label6)
    plt.legend(handles=[line_1, line_2, line_3, line_4, line_5, line_6])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(pfad)
    
    #plt.axis([3, 12, 0, 100000])
    
      #plt.plot(a, 'g_', label="A", linewidth=2)
    #plt.plot(b, 'r_', label="B", linewidth=2)
    plt.ylim((0, 100000))
    plt.ylabel(yname)
    plt.grid(True)
    plt.savefig(pfad, bbox_inches='tight', format='pdf')
    plt.clf()
    #plt.show()


def plotPortfolio( yname, pfad , 
                    monthsA, monthsB,
                    a, label1, 
                    b, label2,
                    c, label3,
                    d, label4,
                    e, label5,
                    f=None, label6=None):
    pass
    plt.clf()
    plt.figure(figsize=(8, 5), dpi=200)
    plt.ylim((0, 100000))
    line_1, = plt.plot(monthsA, a, color='green', markerfacecolor='blue', markersize=7, label=label1)
    line_2, = plt.plot(monthsA, b, color='red', markerfacecolor='blue', markersize=7, label=label2)
    line_3, = plt.plot(monthsB, c, color='blue', markerfacecolor='blue', markersize=7, label=label3)
    line_4, = plt.plot(monthsB, d , color='yellow', markerfacecolor='blue', markersize=7, label=label4)
    line_5, = plt.plot(monthsB, e, color='pink', markerfacecolor='blue', markersize=7, label=label5)
    if not f is None:
        line_6, = plt.plot(monthsB, f , color='orange', markerfacecolor='blue', markersize=7, label=label6)
        plt.legend(handles=[line_1, line_2, line_3, line_4, line_5, line_6])
    else:
        plt.legend(handles=[line_1, line_2, line_3, line_4, line_5])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(pfad)
    
    #plt.axis([3, 12, 0, 100000])
    plt.ylim((0, 100000))
      #plt.plot(a, 'g_', label="A", linewidth=2)
    #plt.plot(b, 'r_', label="B", linewidth=2)
    plt.ylabel(yname)
    plt.grid(True)
    plt.savefig(pfad, bbox_inches='tight', format='pdf')
    plt.clf()
    #plt.show()



def plotTheLoss(a,b, name="foo.pdf"):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8, 5), dpi=200)
    #line_1, = plt.plot(a, color='green', linestyle='None', marker='|', markerfacecolor='blue', markersize=7, label='Loss')
    line_1, = plt.plot(a, color='green', linestyle='-', markerfacecolor='blue', markersize=7, label='Trainings_Loss')
    line_2, = plt.plot(b, color='red', linestyle='-', markerfacecolor='blue', markersize=7, label='Validation_Loss')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.legend(handles=[line_1, line_2])
    plt.grid(True)
    plt.savefig(name, bbox_inches='tight', format='pdf')
    #plt.show()

def plotThePatience(a, name="foo.pdf"):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8, 5), dpi=200)
    #line_1, = plt.plot(a, color='green', linestyle='None', marker='|', markerfacecolor='blue', markersize=7, label='Loss')
    line_1, = plt.plot(a, color='green', linestyle='-', markerfacecolor='blue', markersize=7, label='Patience_Count')

    plt.ylabel('Patience_Count')
    plt.xlabel('Step')
    plt.grid(True)
    plt.savefig(name, bbox_inches='tight', format='pdf')
    #plt.show()
    plt.close('all')



def plotter(a, b=None,name="foo.pdf"):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8, 5), dpi=200)
    #line_1, = plt.plot(a, color='green', linestyle='None', marker='|', markerfacecolor='blue', markersize=7, label='IST')
    #plt.plot(a, 'g_', label="A", linewidth=2)
    #line_2, = plt.plot(b, 'r_', label="ANN", linewidth=2)
    plt.scatter(a, b, edgecolor='b', s=20, label="Samples")
    plt.ylabel('Ist in T€')
    plt.xlabel('Projekte')
    plt.legend(loc="best")
    #plt.legend(handles=[line_1, line_2])
    
    
    plt.grid(True)
    #plt.savefig(name, bbox_inches='tight', format='pdf')
    plt.show()
    #plt.close('all')



def plotTheOutput(a, b=None,name="foo.pdf"):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8, 5), dpi=200)
    line_1, = plt.plot(a, color='green', linestyle='None', marker='|', markerfacecolor='blue', markersize=7, label='IST')
    #plt.plot(a, 'g_', label="A", linewidth=2)
    line_2, = plt.plot(b, 'r_', label="ANN", linewidth=2)
    plt.ylabel('Ist in T€')
    plt.xlabel('Projekte')
    
    plt.legend(handles=[line_1, line_2])
    
    
    plt.grid(True)
    plt.savefig(name, bbox_inches='tight', format='pdf')
    #plt.show()
    plt.close('all')
    
def histogram(a, name="foo.pdf"):
    plt.clf()
    plt.cla()
    plt.hist(a)

    plt.ylabel('Projekte')
    plt.grid(True)
    plt.savefig(name, bbox_inches='tight', format='pdf')
    #plt.show()
    plt.close('all')
    
class VektorComparisionTF(object):
    '''
    classdocs
    '''
    def __init__(self, first, second):
        if first.shape != second.shape:
            raise Exception('Shapes of given Vector do not Match',)
        self.first = first
        self.second = second
        
        x = tf.placeholder(tf.float32, first.shape, name = "first")
        y = tf.placeholder(tf.float32, second.shape, name = "second")        
        model = tf.abs(y) - tf.abs(x)
        #mae = tf.metrics.mean_absolute_error(y, x)
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        #tf.local_variables_initializer()
        #self.mae = sess.run(mae, feed_dict={x: first, y: second})
        self.difference = sess.run(model, feed_dict={x: first, y: second})
    
    def aboluteDifferenceSum(self):
        return np.abs(self.difference).sum()
    
    def aboluteDifferenceMean(self):
        return np.abs(self.difference).mean()
    
    def plot(self):
        plotTheOutput(self.difference)
    
    def calculate(self, interval):
        final = np.array([[0,0],[5,0],[10,0],[20,0],[50,0]])
        for x in np.nditer(self.difference):
            t = np.abs(100/self.aboluteDifferenceSum()*x)
            #print(t)
            for i in range(4, 0,-1):
                if final[i][0] < t:
                    final[i][1] = final[i][1] + 1
                    break
                if t < 5:
                    final[0][1] = final[0][1] + 1
                    break       
        return final
    
def NormFunction(elem, a, b):
    return np.tanh(elem/10000*((b-a)/11))

def DeNormFunction(elem, a, b):
    return 10000*np.arctanh(elem)/((b-a)/11)

def NormLNFunction(elem, a, b):
    offset = 10
    return (1/10)*np.log(elem*((b-a)/11) + offset)

def DeNormLNFunction(elem, a, b):
    offset = 10
    return (np.exp(10*elem)-offset)*((b-a)/11)          

        