import numpy as np

class migrator:

    def get_loc_history(self):
        return self.history
    
    
    def __init__(self, prev_p, prev_c):
        polar_minus_c_dif = prev_p - prev_c
        n = polar_minus_c_dif.shape[0]
        self.history = np.zeros([n,1], dtype = np.bool)
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
