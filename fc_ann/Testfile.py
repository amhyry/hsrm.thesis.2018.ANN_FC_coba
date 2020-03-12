import numpy as np
import version11.RawData as dta
import tensorflow as tf
#import matplotlib.pyplot as plt
#from openpyxl import Workbook
#from openpyxl import load_workbook
import version12.Library as mylib
import copy

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import datetime
import os

def NormFunction(elem, a, b):
    return np.tanh(elem/10000*((b-a)/11))

def DeNormFunction(elem, a, b):
    return 10000*np.arctanh(elem)/((b-a)/11)

def NormLNFunction(elem, a, b):
    offset = 500
    return (1/10)*np.log(elem*((b-a)/11) + offset)

def DeNormLNFunction(elem, a, b):
    offset = 500
    return (np.exp(10*elem)-offset)*((b-a)/11)
    
    
   
   
testin = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1]
testout = [ 3.310495144, 5.483507116, 1.562337958, -0.446876282, -1.964565367, 4.560250282, 6.615645457, 3.339307831, 8.277912613, 4.335197312, 6.959006533, 9.295076414, 6.642989048, 10.75504772, 11.96324678, 5.3879286, 11.54742528, 12.90082178, 10.67735827, 12.82429698, 9.287021913] 
input = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ]
output = [ -0.513889482, -0.001974795, 9.032928673, 22.54443634, 43.80018726, 62.5807605, 92.54620798, 119.3979175, 151.2234205, 192.7037543, 238.0400569, 282.7144774, 337.4306555, 390.7141295, 456.8147672, 517.8982388, 586.3615213, 663.3670698, 743.804661, 825.7102814, 908.6767076, 1005.148491, 1101.522158, 1194.483008, 1300.69826, 1408.22934, 1523.873138, 1642.126992, 1764.677075, 1894.461758, 2028.05539 ]
   
   
   
    
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

array =[[[ 0.14676282,  0.03455082,  0.15042725,  0.09115881],
       [ 0.07697837,  0.15317628,  0.13299353,  0.09383005],
       [-0.12656879,  0.06329668, -0.08818144,  0.23814206],
       [ 0.32686982,  0.43481466,  0.31170693,  0.23387818],
       [-0.19430029, -0.14942734, -0.02931355, -0.1172968 ],
       [ 0.06869261, -0.09599274, -0.21006078, -0.19037364],
       [ 0.11253987, -0.09674381,  0.05533104,  0.01243536],
       [ 0.28010482,  0.23945457,  0.16195647,  0.16404192],
       [-0.20475684, -0.07866593, -0.28041133, -0.11833722],
       [ 0.04465827, -0.06401879, -0.06219077, -0.07466368],
       [ 0.03253097,  0.06020303,  0.04192312, -0.01394873],
       [-0.13635154, -0.14074853, -0.05921587, -0.02280422],
       [-0.03233989, -0.0457528 ,  0.08430453, -0.08888004],
       [-0.08176839,  0.19329354,  0.14503767,  0.23257022],
       [ 0.03321674,  0.07596249,  0.03221212,  0.08117042],
       [ 0.37640968,  0.10883583,  0.39064074,  0.22635151]] , 
       [[ 0.13692768,  0.02432723,  0.14115588,  0.08118642],
       [ 0.076222  ,  0.15201959,  0.13294417,  0.09323406],
       [-0.12905577,  0.05997248, -0.09004493,  0.23560031],
       [ 0.3349975 ,  0.44271362,  0.32079807,  0.24320717],
       [-0.18537749, -0.1403179 , -0.01869434, -0.10650811],
       [ 0.08041453, -0.08415245, -0.19682524, -0.17682751],
       [ 0.11288051, -0.09678334,  0.05669284,  0.01370499],
       [ 0.28082263,  0.23975731,  0.16363169,  0.16556787],
       [-0.20093584, -0.07481243, -0.27559817, -0.11363415],
       [ 0.04957693, -0.05912588, -0.05626181, -0.06893329],
       [ 0.02953515,  0.05667157,  0.03964429, -0.01675089],
       [-0.14419542, -0.14881589, -0.06627857, -0.03008296],
       [-0.03141846, -0.04499136,  0.08637741, -0.08700906],
       [-0.08839051,  0.18641162,  0.13917077,  0.22623099],
       [ 0.02179042,  0.0637736 ,  0.02119896,  0.06939965],
       [ 0.37382126,  0.10612756,  0.38875768,  0.22465895]] ]

def convertForCSV(array):
    pass
    if True:
        temp = np.array(["%.2f" % w for w in array.reshape(array.size)])
        temp = np.array([s.replace('.' , ',') for s in temp])
        temp = temp.reshape(array.shape)
        return temp
    
    temp = ["%.2f" % x for x in array]
    return [s.replace('.' , ',') for s in temp]

if __name__ == '__main__':
    pass
    mylib.plotter(testin, testout, "Testdatei.pdf")
    
    #data = dta.MyInputData()
    #input = data.initializeMonths(3)
    
    #print(input.train.x)
    #input.normalize_all(mylib.NormLNFunction)
    #print(input.train.x)
    #input.denormalize_all(mylib.DeNormLNFunction) 
    #print(input.train.x)
    #pfad = datetime.datetime.now().strftime("%Y%m%d %H-%M-%S")
    #os.makedirs(pfad)    
    #print(convertForCSV(input.train.y))

    
    #print(a_strings)
    
    #np.savetxt(pfad + '/testestest.csv', convertForCSV(input.train.y), delimiter=';', fmt='%s')
    #print(convertForCSV(input.train.y))
    
    #print("done")  
    
    
    #print(np.asarray(array))
    #a = 80
    #b = 22
    #c = (a+b)/b
    #print(c)
    
    #array = np.asarray(array)
    
    '''
    #x,y,z = array.nonzero()
    
    #print(array)
    #print(x)
    #print(y)
    #print(z)

    #x = np.linspace(-6, 6, 30)
    #y = np.linspace(-6, 6, 30)

    #X, Y = np.meshgrid(x, y)
    #Z = f(X, Y)

    #fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');
    
    ax.view_init(60, 35)

    plt.show()
    
    
    
    #input.initializeMonths(3)
  
    
    x = [90000000,50000000,10000000,100000,10000,9999,8000,5000,5500,5000,4500,4000,3800,3700,3600,3550,3525,3520,3519,3518,3517,3516,3515,3515,3515,3515,3515,3515]
    
    a = []
    a.append([x, np.log(x), [np.mean(x[i-5:i]) for i in range(len(x))]])
    
    print(a)
    
    
    #mylib.plotTheLoss(x, "foo2.pdf")
    #mylib.plotTheLoss(np.log(x))
    
    #mylib.plotTheLoss([np.mean(x[i-5:i]) for i in range(len(x))], "foo3.pdf")
    
    print("Done")
    #print(input.csvTrain)
    # Model parameters
    # W = tf.Variable([.3], dtype=tf.float32)
    # b = tf.Variable([1.], dtype=tf.float32)
    
    
    
    a_0 = tf.Variable([1.], dtype=tf.float32)
    a_1 = tf.Variable([.3], dtype=tf.float32)
    a_2 = tf.Variable([.3], dtype=tf.float32)
    a_3 = tf.Variable([.3], dtype=tf.float32)
    a_4 = tf.Variable([.3], dtype=tf.float32)
    a_5 = tf.Variable([.3], dtype=tf.float32)
    a_6 = tf.Variable([.3], dtype=tf.float32)
    a_7 = tf.Variable([.3], dtype=tf.float32)
    a_8 = tf.Variable([.3], dtype=tf.float32)
    a_9 = tf.Variable([.3], dtype=tf.float32)
    a_10 = tf.Variable([.3], dtype=tf.float32)
    a_11 = tf.Variable([.3], dtype=tf.float32)
    a_12 = tf.Variable([.3], dtype=tf.float32)
    a_13 = tf.Variable([.3], dtype=tf.float32)
    
    
    # Model input and output
    x = tf.placeholder(tf.float32)
    # linear_model = W*x + b
    y = tf.placeholder(tf.float32)
    
    polynom_model = a_0 + a_1*x + a_2*x**2 + a_3*x**3 + a_4*x**4 + a_5*x**5 + a_6*x**6# + a_7*x**7 + a_8*x**8# +a_9*x**9+a_10*x**10+a_11*x**11#+a_12*x**12+a_13*x**13
    
    
    # loss
    loss = tf.reduce_sum(tf.square(polynom_model - y)) # sum of the squares
    # optimizer
    
    optimizer = tf.train.AdamOptimizer(0.000001)
  

    wb = load_workbook(r'H:\My Documents\_Thesis\Archiv\Beispiel_Polynomfit.xlsx', data_only=True) 
    ws = wb["Tabelle1"]
    cell_range_x = ws['B45':'B50'] 
    cell_range_y = ws['C45':'C50']  
  
    x_train = []
    for cellObj in cell_range_x: 
        x_train.append( cellObj[0].value )
        
    y_train = []
    for cellObj in cell_range_y:
        y_train.append( cellObj[0].value )
    
    print (x_train)
    print (y_train)
    #print(x_test)
    #print(y_test)  
  
  
    
    #optimizer = tf.train.GradientDescentOptimizer(0.00001)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(10000):
        #if i%100 == 0:
        #    print(W)
        #    print(b)
        # print(sess.run([W, b]))
        sess.run(train, {x: x_train, y: y_train})
        print(sess.run(loss, {x: x_train, y: y_train}))
    
    print(sess.run(loss, {x: x_train, y: y_train}))
    #print(sess.run(loss, {x: testin, y: testout})    )
    
    result = sess.run(polynom_model, {x: x_train})
    
    
    # evaluate training accuracy
    #curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    curr_a_0, curr_a_1, curr_a_2, curr_a_3, curr_a_4, curr_loss = sess.run([a_0, a_1, a_2, a_3, a_4, loss], {x: x_train, y: y_train})
    print("a_0: %s a_1: %s a_2: %s a_3: %s a_4: %s loss: %s"%(curr_a_0, curr_a_1, curr_a_2, curr_a_3, curr_a_4, curr_loss))
    
    plt.figure(figsize=(8, 5), dpi=200)
    line_1, = plt.plot(x_train, y_train, color='green', linestyle='-', marker='x', markerfacecolor='blue', markersize=7, label='Loss')
    line_2, = plt.plot(x_train, result, color='red', linestyle='-', marker='x', markerfacecolor='blue', markersize=7, label='Loss')
    plt.show()

#print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

    
    #x = np.random.randint(2, size=(200, 1))
    #print("hello world")
    #print(x)
    
    '''