import numpy as np

class Migrator:

    def get_loc_history(self):
        return self.history
    
    def get_loc_current(self):
        return self.history[:,-1]

    def get_decision(self, K, scorematrix):
        current_loc = self.get_loc_current()
        decision = np.zeros([self.n,1])
        diff = np.zeros([self.n,1])

        for i in range(self.n):
            max_score = np.max(scorematrix[i])
            max_arg = np.argmax(scorematrix[i])
            if current_loc[i]: #if it is polar dominant
                print('p d')
                if max_arg >= self.K: #need to move to cartesian
                    print('need to move to cartesian')
                    decision[i] = 1
                    diff[i] = np.max(scorematrix[i][0:self.K]) - np.max(scorematrix[i]) #negative
            else: #if it is cartesian dominant
                print('c d')
                if max_arg < self.K: #need to move to polar
                    print('need to move to polar')
                    decision[i] = -1
                    diff[i] = np.max(scorematrix[i]) - np.max(scorematrix[i][self.K:]) #positive

        print(decision)
        number_of_ones_decision = np.count_nonzero(decision == 1)
        print('The number of moving: polar -> cartesian: ', number_of_ones_decision)
        number_of_minusones_decision = np.count_nonzero(decision == -1)
        print('The number of moving: cartesian -> polar: ', number_of_minusones_decision)
        return diff, decision

    def decide_move(self, num_max_move, diff, decision):
        current_loc = self.get_loc_current()

        num_polarToCarte = np.count_nonzero(decision == 1)
        num_carteToPolar = np.count_nonzero(decision == -1)
        num_should_move = max(num_polarToCarte, num_carteToPolar)

        if num_max_move > num_should_move:
            num_max_move = num_should_move

        sorted_diff = sorted(range(len(diff)), key=lambda index: diff[index])
        polarToCarte = sorted_diff[:num_max_move]
        carteToPolar = sorted_diff[-num_max_move:]

        new_Location = current_loc.copy()

        count_moveto_polar = 0
        count_moveto_carte = 0
        for index in polarToCarte:
            new_Location[index] = False
            count_moveto_polar+=1
        for index in carteToPolar:
            new_Location[index] = True
            count_moveto_carte+=1

        new_Location = new_Location.reshape(-1,1)
        self.history = np.append(self.history, new_Location, axis = 1)
        print(self.history.shape)
        print('round of migration completed')
        print('Report\nNumber moved:', count_moveto_polar+count_moveto_carte, 'in total')
        print('The number of moving: polar -> cartesian: ', count_moveto_carte)
        print('The number of moving: cartesian -> polar: ', count_moveto_polar)


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





### For test only:

#     def __init__(self, K, history):
#         self.K = K
#         self.history = history
#         self.n = history.shape[0]

# K = 5
# scorematrix = np.load('../results/scorematrix/scorematrix_round_0.npy')
# history = np.load('../results/history/history_round_0.npy')

# test = Migrator(K, history)
# diff, decision = test.get_decision(K, scorematrix)
# test.decide_move(500, diff, decision)