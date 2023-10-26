import numpy as np
class migrator:

    def get_loc_history(self):
        return self.history
    
    def get_loc_current(self):
        return self.history[:,-1]
    
    def get_prob_history(self):
        return self.probablity
    
    def get_prob_current(self):
        return self.probablity[:,-1]
    
    def decide_and_mod_prob(self, scorematrix):
        current_loc = self.get_loc_current()
        self.probablity = np.append(self.probablity,np.zeros([self.n,1]),1)
        for i in range(self.n):
            max_score = np.max(scorematrix[i])
            max_arg = np.argmax(scorematrix[i])
            if current_loc[i]:#if it is polar dominant
                avg_carte = np.average(scorematrix[i][self.K:])
                if max_arg >= self.K:
                    dif = avg_carte - np.max(scorematrix[i][0:self.K])
                    self.mod_prob(i, True, dif)
                else:
                    self.mod_prob(i, False, max_score - avg_carte)
            else:
                avg_polar = np.average(scorematrix[i][:self.K])
                if max_arg < self.K:
                    dif = avg_polar - np.max(scorematrix[i][self.K:])
                    self.mod_prob(i, True, dif)
                else:
                    self.mod_prob(i, False, max_score - avg_polar)
                    

    def mod_prob(self, index, shouldMove, acce):
        if shouldMove:
            self.probablity[index][-1] = self.probablity[index][-2]*(1+acce)
            if self.probablity[index][-1] > 1:
                self.probablity[index][-1] = 1
        else:
            self.probablity[index][-1] = self.probablity[index][-2]*(1-acce)

    def migrate(self):
        array_size = (self.n,)
        lottery = np.random.rand(*array_size)
        if_Switch = np.zeros(array_size, dtype = np.bool)
        if_Switch[self.get_prob_current() > lottery] = True
        new_Location = np.logical_xor(if_Switch, self.get_loc_current())
        print(self.history.shape)
        number_of_move = np.count_nonzero(if_Switch)
        new_Location = new_Location.reshape(-1,1)
        print(new_Location.shape)
        print(if_Switch[0:10])
        self.history = np.append(self.history,new_Location, axis = 1)
        print(self.history.shape)
        print('round of migration completed')
        print('Report\n Number moved:', number_of_move)


    def __init__(self, prev_p, prev_c, K):
        self.K = K
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