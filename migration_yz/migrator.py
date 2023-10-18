import numpy as np
import random

class migrator:

    def get_loc_history(self):
        return self.history
    
    def get_loc_current(self):
        return self.history[:,-1]
    
    def get_prob_history(self):
        return self.probablity
    
    def get_prob_current(self):
        return self.probablity[:,-1]
    
    def migrate(self):
        np.random.seed(0)
        array_size = (self.n,)
        lottery = np.random.rand(*array_size)
        if_Switch = np.zeros(array_size, dtype = np.bool)
        if_Switch[self.get_prob_current() > lottery] = True
        new_Location = np.logical_xor(if_Switch, self.get_loc_current())
        print(self.history.shape)
        
        new_Location = new_Location.reshape(-1,1)
        print(new_Location.shape)
        print(if_Switch[0:10])
        self.history = np.append(self.history,new_Location, axis = 1)
        print(self.history.shape)


    def __init__(self, prev_p, prev_c):
        polar_minus_c_dif = prev_p - prev_c
        self.n = polar_minus_c_dif.shape[0]
        self.history = np.zeros([self.n,1], dtype = np.bool)
        self.probablity = np.ones([self.n,1],dtype = np.float) * 0.5
        threshold = np.median(polar_minus_c_dif)#take the median for even split
        i = 0
        for item in polar_minus_c_dif:#now the decision is based on the positivity of difference, which is based on the fact that about half and half split
            if item > threshold:
                self.history[i] = True #True is for Polar
            else:
                self.history[i] = False
            i += 1
            
                
        
    def __str__(self):
        count = 0
        result = ''
        for row in self.filematrix:
            result += str(count) + '.tif' + str(row) + '\n'
            count += 1
        return result
