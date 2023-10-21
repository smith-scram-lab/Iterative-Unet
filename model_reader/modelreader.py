class modelreader:
    
    def update_score(self, score_matrix):
        print('updated')
        print(score_matrix[0])
        self.score_matrix = score_matrix

    def update_polar_history(self, polar_history):
        self.polar_history = polar_history
    
    def update_carte_history(self, carte_history):
        self.carte_history = carte_history

    def get_score(self):
        #print(self.score_matrix[0])
        return self.score_matrix
    
    def __init__(self):
        self.polar_history = None
        self.carte_history = None
        self.score_matrix = None