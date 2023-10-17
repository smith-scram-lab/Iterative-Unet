import numpy as np

class migrator:

    def get_loc_history(self):
        print(self.history.sum())
        print(self.history)
        return self.history
    
    
    def __init__(self, prev_p, prev_c):
        self.filematrix = filematrix
        n = filematrix.shape[0]
        self.history = np.zeros([n,1], dtype = np.bool)
        for row_f, row_h in zip(self.filematrix, self.history):
            if np.argmax(row_f) > 4:
                row_h = [True]
                
        
    def __str__(self):
        count = 0
        result = ''
        for row in self.filematrix:
            result += str(count) + '.tif' + str(row) + '\n'
            count += 1
        return result
