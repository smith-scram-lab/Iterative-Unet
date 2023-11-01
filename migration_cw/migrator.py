import numpy as np

class migrator:

    def __init__(self, prev_p, prev_c, K):
        self.K = K
        polar_minus_c_dif = prev_p - prev_c
        self.n = polar_minus_c_dif.shape[0]
        self.history = np.zeros([self.n,1], dtype = np.bool)
        threshold = np.median(polar_minus_c_dif)#take the median for even split
        i = 0
        for item in polar_minus_c_dif:#now the decision is based on the positivity of difference, which is based on the fact that about half and half split
            if item > threshold:
                self.history[i] = True #True is for Polar
            else:
                self.history[i] = False
            i += 1
                
    def get_loc_current(self):
        return self.history[:,-1]
    
    def get_loc_history(self):
        return self.history
    
    def median_of_medians(self, arr, k):
        if len(arr) <= 5:
            return np.partition(arr, k)[k]

        sublists = [arr[i:i+5] for i in range(0, len(arr), 5)]
        medians = [np.median(sublist) for sublist in sublists]
    
        pivot = self.median_of_medians(np.array(medians), len(medians) // 2)
    
        low = arr[arr < pivot]
        high = arr[arr > pivot]
        equal = arr[arr == pivot]
    
        if k < len(low):
            return self.median_of_medians(low, k)
        elif k < len(low) + len(equal):
            return pivot
        else:
            return self.median_of_medians(high, k - len(low) - len(equal))
        
    def migrate(self,scorematrix):
        loc = self.get_loc_current()
        curr = np.zeros(len(loc))
        for i in range(len(scorematrix)):
            if loc[i]:
                average = np.average(scorematrix[i][self.K:])
                local = np.max(scorematrix[i][:self.K])
                curr[i] = local - average
            else:
                average = np.average(scorematrix[i][:self.K])
                local = np.max(scorematrix[i][self.K:])
                curr[i] = local - average
        k = len(curr)//2
        median = self.median_of_medians(curr, k)
        if median < 0:
            median = 0
        polar = 0
        cartesian = 0
        for i in range(len(curr)):
            if loc[i] and curr[i]< median:
                loc[i] = False
                polar += 1
            elif curr[i]< median:
                loc[i] = True
                cartesian += 1
            if polar > k//2 or cartesian > k//2:
                i = len(curr)
        self.history = np.append(self.history,loc[:, np.newaxis], axis = 1)
        


# rows = 7404
# cols = 10

# # Create an empty array filled with zeros
# array = np.zeros((rows, cols))

# # First 7404//2 rows, only one of the first five positions has a random value between 0 and 1
# random_indices = np.random.randint(0, 5, rows//2)
# array[:rows//2, random_indices] = np.random.uniform(0, 1, rows//2)

# # First 7404//2 rows, last five positions have random values between 0 and 1
# array[:rows//2, 5:] = np.random.uniform(0, 1, (rows//2, 5))

# # Last 7404//2 rows, only one of the last five positions has a random value between 0 and 1
# random_indices = np.random.randint(5, 10, rows//2)
# array[rows//2:, random_indices] = np.random.uniform(0, 1, rows//2)

# # Last 7404//2 rows, first five positions have random values between 0 and 1
# array[rows//2:, :5] = np.random.uniform(0, 1, (rows//2, 5))
        
# input = np.zeros([7404], dtype=bool) #testing code
# test =  migrator(input,5)
# random_array = array
# test.migrate(random_array)
# print(test.get_loc_current())
# print(test.get_loc_history())
            
                
            
            
# def migrate(self,scorematrix):
    #     loc = self.get_loc_current()
    #     polar = np.empty(np.sum(loc))
    #     cartesian = np.empty(len(loc)-np.sum(loc))
    #     for i in range(len(scorematrix)):
    #         if loc[i]:
    #             average = np.average(scorematrix[i][self.K:])
    #             local = np.max(scorematrix[i][:self.K])
    #             polar[i] = local - average
    #         else:
    #             average = np.average(scorematrix[i][:self.K])
    #             local = np.max(scorematrix[i][self.K:])
    #             cartesian[i] = local - average
    #     polar = np.argsort(polar)
    #     cartesian = np.argsort(cartesian)
        
    #     for i in range(len(scorematrix)//4):
    #         if(len(polar)>i):
    #             loc[polar[i]] = False
    #         if(len(cartesian)>i):
    #             loc[cartesian[i]] = True
    #     self.history = np.append(self.history,loc[:, np.newaxis], axis = 1)