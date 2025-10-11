import json
import numpy as np
import math as m
class agent_:
    def __init__(self,nb_state,nb_action,alpha=0.1,epsilon=0.1,gamma=0.9,decay=0.99,path=None,lam_bda=0,threshold=10**-4):  # gama = 0 simple TD(0) , != 0 mean td(lambda)
        self.path = path
        self.nb_state = nb_state
        self.nb_action = nb_action
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = np.zeros((nb_state, nb_action))
        self.decay = decay
        self.q_table= np.load(self.path) if self.path!= None else np.zeros((nb_state, nb_action))
        self.elegibility =np.zeros((nb_state, nb_action))
        self.lam_bda= lam_bda
        self.map=set()
        self.threshold=threshold
    def choose_action(self,state):
        

        a=np.argmax(self.q_table[state]) if np.random.rand() > self.epsilon else np.random.randint(self.nb_action)  
        self.elegibility[state][a]=1
        self.map.add((state,a))
        return a

    def update_q_table(self, state, action, reward,next_state=None):
        current_q = self.q_table[state][action]
        if next_state != None :
            next_action=np.argmax(self.q_table[next_state])
            delta=reward - current_q + self.gamma* self.q_table[next_state][next_action]
        else :
            delta =    reward - current_q
        for x in list(self.map) :
            self.q_table[x[0]][x[1]]+=self.alpha*delta*self.elegibility[x[0]][x[1]]
            self.elegibility[x[0]][x[1]]*=self.gamma*self.lam_bda

            if  self.elegibility[x[0]][x[1]] <self.threshold :
                self.map.remove((x[0], x[1]))
                self.elegibility[x[0]][x[1]] = 0.0

    
    def save_q_table(self, path="q_table_beta.npy"):
        np.save(path, self.q_table)
    
    
    def load_q_table(self, path="q_table_beta.npy"):
        try:
            self.q_table = np.load(path,allow_pickle=True)
            print("loaded successfully")
        except FileNotFoundError:
            raise ValueError(f"No Q-table found at {path}. Please train the agent first.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Q-table: {e}")

    
    
    def reset(self):
        self.q_table = np.zeros((self.nb_state, self.nb_action))



