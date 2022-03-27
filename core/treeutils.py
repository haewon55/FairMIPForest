import numpy as np 
import sys
import os 
import gurobipy as gp
from gurobipy import GRB
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from scipy import stats

from missing_module import * 

def create_tree_struct(D): 
    LL = []
    LR = []
    for l in range(2**D-1):
        LL.append([])
        LR.append([])
    for l in range(D):
        for i in range(2**l):
            v = i + (2**l-1)
            suf_len = D - l -1
            # LL # 
            prefix = (i<<1)|0
            for j in range(2**suf_len):
                leaf = (prefix<<suf_len)|j
                LL[v].append(leaf)
            # LR # 
            prefix = (i<<1)|1
            for j in range(2**suf_len):
                leaf = (prefix<<suf_len)|j
                LR[v].append(leaf)
                
    return LL, LR


def bin_to_int(bin_arr):
    '''
        Converting an array of 0,1's to an integer
        bin_arr[0] is the MSB
    ''' 
    arr_len = len(bin_arr)
    result = 0
    for i in range(arr_len):
        result += bin_arr[i]*2**(arr_len-i-1)
    return result 


def int_to_bin(n, arr_len): 
    '''
        Coverting an integer n into an array of 0,1's of length arr_len
    '''
    bin_str = '{0:b}'.format(n)

    start_idx = arr_len - len(bin_str)
    if start_idx < 0: 
        raise ValueError 
        
    bin_arr = np.zeros(arr_len)
    for i in range(len(bin_str)):
        bin_arr[i+start_idx] = int(bin_str[i])
    
    return bin_arr
    
    


def w_to_z(w, D):
    n = w.shape[0]
    z = np.zeros(n)
    
    for i in range(n):
        d = np.zeros(D)        
        for idx in range(D): 
            if idx == 0:
                d[idx] = w[i,0]
                continue
            new_idx = int((2**idx-1) + bin_to_int(1-d[:idx]))
            d[idx] = w[i,new_idx]
            
        z[i] = bin_to_int(1-d)
    
    return z

M = 10000
mm = -M
epsilon = 0.01

class DecisionTree(object):

    def __init__(self, D=3): 
        self.D = D              # tree depth
        self.num_node = 2**D-1  # number of branching nodes
        self.num_leaf = 2**D    # number of leaf nodes

    
    def build_model(self, X, m, y, fair=None, lambd = 1, S = None):

        if fair is not None: 
            if lambd == 0 or S is None: 
                raise ValueError
            if fair not in ['fnr', 'fpr', 'acc']:
                # TODO: implement 'eqodds'
                raise ValueError

        num_node = self.num_node
        num_leaf = self.num_leaf
        n, d = X.shape[0], X.shape[1]
        self.d = d

        model = gp.Model("compas_tree")


        ###################
        #### VARIABLES ####
        ################### 

        p = model.addVars(num_node, d, vtype=GRB.BINARY, name='p') 
        z = model.addVars(n, num_leaf, vtype=GRB.BINARY, name='z')
        u = model.addVars(num_leaf, vtype=GRB.BINARY, name='u')
        w = model.addVars(n, num_node, vtype=GRB.BINARY, name='w')
        w_nm = model.addVars(n, num_node, vtype=GRB.BINARY, name='w_nm')
        w_1 = model.addVars(n, num_node, vtype=GRB.BINARY, name='w_1')
        w_2 = model.addVars(n, num_node, vtype=GRB.BINARY, name='w_2')

        q = model.addVars(num_node, vtype=GRB.CONTINUOUS, name='q') 
        c = model.addVars(num_node, vtype=GRB.BINARY, name='c') 

        loss = model.addVars(num_leaf, vtype=GRB.CONTINUOUS, name='L')

        f0 = model.addVars(num_leaf, vtype=GRB.CONTINUOUS, name='f0')
        f1 = model.addVars(num_leaf, vtype=GRB.CONTINUOUS, name='f1')
        fair_reg = model.addVar(vtype=GRB.CONTINUOUS, name='fair_reg') 

        if fair is None:
            for l in range(num_leaf): 
                f0[l].lb, f0[l].ub = 0, 0
                f1[l].lb, f1[l].ub = 0, 0
            fair_reg.lb, fair_reg.ub = 0, 0

        
        #####################
        #### CONSTRAINTS ####
        #####################

        print('Adding Constraints')

        LL, LR = create_tree_struct(self.D)

        ## One-hot encoding constraints ##
        model.addConstrs(gp.quicksum(p[v, i] for i in range(d)) == 1 for v in range(num_node))
        model.addConstrs(gp.quicksum(z[i,l] for l in range(num_leaf)) == 1 for i in range(n))

        # Branching constraints ## 
        model.addConstrs((q[v] - gp.quicksum(p[v,j]*(1-m[i,j])*X[i,j] for j in range(d))) 
                        <= M* w_nm[i,v] - epsilon*(1-w_nm[i,v]) for i in range(n) for v in range(num_node))

        model.addConstrs((q[v] - gp.quicksum(p[v,j]*(1-m[i,j])*X[i,j] for j in range(d))) 
                        >= mm*(1-w_nm[i,v]) for i in range(n) for v in range(num_node))

        model.addConstrs((w_1[i,v] + 1) >= (1-gp.quicksum(p[v,j]*m[i,j] for j in range(d)) + w_nm[i,v]) 
                        for i in range(n) for v in range(num_node))

        model.addConstrs(w_1[i,v] <= (1-gp.quicksum(p[v,j]*m[i,j] for j in range(d))) 
                        for i in range(n) for v in range(num_node))

        model.addConstrs(w_1[i,v] <= w_nm[i,v] for i in range(n) for v in range(num_node))

        model.addConstrs((w_2[i,v] + 1) >= (gp.quicksum(p[v,j]*m[i,j] for j in range(d)) + c[v]) 
                        for i in range(n) for v in range(num_node))

        model.addConstrs(w_2[i,v] <= gp.quicksum(p[v,j]*m[i,j] for j in range(d)) 
                        for i in range(n) for v in range(num_node))
        model.addConstrs(w_2[i,v] <= c[v] for i in range(n) for v in range(num_node))

        model.addConstrs(w[i,v] >= w_1[i,v] for i in range(n) for v in range(num_node))
        model.addConstrs(w[i,v] >= w_2[i,v] for i in range(n) for v in range(num_node))
        model.addConstrs(w[i,v] <= (w_1[i,v]+w_2[i,v]) for i in range(n) for v in range(num_node))

        ## Leaf topology constraints ## 
        model.addConstrs(z[i,l] <= w[i,v] for i in range(n) for v in range(num_node) for l in LL[v])
        model.addConstrs(z[i,l] <= (1-w[i,v]) for i in range(n) for v in range(num_node) for l in LR[v])

        ## Leaf prediction constraints ## 

        model.addConstrs(gp.quicksum((2*y[i]-1)*z[i,l] for i in range(n)) <= M*u[l]-epsilon*(1-u[l]) for l in range(num_leaf))
        model.addConstrs(gp.quicksum((2*y[i]-1)*z[i,l] for i in range(n)) >= mm*(1-u[l]) for l in range(num_leaf))

        model.addConstrs(loss[l] <= gp.quicksum((1-y[i])*z[i,l] for i in range(n)) for l in range(num_leaf))
        model.addConstrs(loss[l] <= gp.quicksum(y[i]*z[i,l] for i in range(n)) for l in range(num_leaf))
        model.addConstrs(loss[l] >= gp.quicksum((1-y[i])*z[i,l] for i in range(n)) - (1-u[l])*M for l in range(num_leaf))
        model.addConstrs(loss[l] >= gp.quicksum(y[i]*z[i,l] for i in range(n)) - u[l]*M for l in range(num_leaf))

        if fair is not None: 
            ## Fairness regularizer constraints ## 
            
            if fair == 'acc': 
                n0, n1 = np.count_nonzero(S==0), np.count_nonzero(S==1)
    
                model.addConstrs(f0[l] >= gp.quicksum((1-y[i])*z[i,l]*(1-S[i]) for i in range(n)) + mm *(1-u[l]) for l in range(num_leaf))
                model.addConstrs(f0[l] <= gp.quicksum((1-y[i])*z[i,l]*(1-S[i]) for i in range(n)) + M*(1-u[l])+epsilon for l in range(num_leaf))
                model.addConstrs(f0[l] >= gp.quicksum(y[i]*z[i,l]*(1-S[i]) for i in range(n)) + mm*u[l] for l in range(num_leaf))
                model.addConstrs(f0[l] <= gp.quicksum(y[i]*z[i,l]*(1-S[i]) for i in range(n)) +M*u[l] + epsilon for l in range(num_leaf))
                model.addConstrs(f1[l] >= gp.quicksum((1-y[i])*z[i,l]*S[i] for i in range(n)) + mm *(1-u[l]) for l in range(num_leaf))
                model.addConstrs(f1[l] <= gp.quicksum((1-y[i])*z[i,l]*S[i] for i in range(n)) + M*(1-u[l])+epsilon for l in range(num_leaf))
                model.addConstrs(f1[l] >= gp.quicksum(y[i]*z[i,l]*S[i] for i in range(n)) + mm*u[l] for l in range(num_leaf))
                model.addConstrs(f1[l] <= gp.quicksum(y[i]*z[i,l]*S[i] for i in range(n)) +M*u[l] + epsilon for l in range(num_leaf))

            elif fair == 'fpr': 
                n0, n1 = np.count_nonzero((S==0) & (y==0)), np.count_nonzero((S==1) & (y==0))

                model.addConstrs(f0[l] >= gp.quicksum((1-y[i])*z[i,l]*(1-S[i]) for i in range(n)) + mm *(1-u[l]) for l in range(num_leaf))
                model.addConstrs(f0[l] <= gp.quicksum((1-y[i])*z[i,l]*(1-S[i]) for i in range(n)) + M*(1-u[l])+epsilon for l in range(num_leaf))
                model.addConstrs(f0[l] >= mm*u[l] for l in range(num_leaf))
                model.addConstrs(f0[l] <= M*u[l] + epsilon for l in range(num_leaf))
                model.addConstrs(f1[l] >= gp.quicksum((1-y[i])*z[i,l]*S[i] for i in range(n)) + mm *(1-u[l]) for l in range(num_leaf))
                model.addConstrs(f1[l] <= gp.quicksum((1-y[i])*z[i,l]*S[i] for i in range(n)) + M*(1-u[l])+epsilon for l in range(num_leaf))
                model.addConstrs(f1[l] >= mm*u[l] for l in range(num_leaf))
                model.addConstrs(f1[l] <= M*u[l] + epsilon for l in range(num_leaf))

            elif fair == 'fnr': 
                n0, n1 = np.count_nonzero((S==0) & (y==1)), np.count_nonzero((S==1) & (y==1))

                model.addConstrs(f0[l] >= mm *(1-u[l]) for l in range(num_leaf))
                model.addConstrs(f0[l] <= M*(1-u[l])+epsilon for l in range(num_leaf))
                model.addConstrs(f0[l] >= gp.quicksum(y[i]*z[i,l]*(1-S[i]) for i in range(n)) + mm*u[l] for l in range(num_leaf))
                model.addConstrs(f0[l] <= gp.quicksum(y[i]*z[i,l]*(1-S[i]) for i in range(n)) +M*u[l] + epsilon for l in range(num_leaf))
                model.addConstrs(f1[l] >= mm *(1-u[l]) for l in range(num_leaf))
                model.addConstrs(f1[l] <= M*(1-u[l])+epsilon for l in range(num_leaf))
                model.addConstrs(f1[l] >= gp.quicksum(y[i]*z[i,l]*S[i] for i in range(n)) + mm*u[l] for l in range(num_leaf))
                model.addConstrs(f1[l] <= gp.quicksum(y[i]*z[i,l]*S[i] for i in range(n)) +M*u[l] + epsilon for l in range(num_leaf))

            model.addConstr(fair_reg >=0)
            model.addConstr(fair_reg >= gp.quicksum(f0[l] for l in range(num_leaf))/n0 - gp.quicksum(f1[l] for l in range(num_leaf))/n1  )
            model.addConstr(fair_reg >= -( gp.quicksum(f0[l] for l in range(num_leaf))/n0 - gp.quicksum(f1[l] for l in range(num_leaf))/n1  ))


        ###################
        #### OBJECTIVE ####
        ###################

        acc_loss = gp.quicksum(loss[l] for l in range (num_leaf))/n
        if fair == None: 
            obj = acc_loss 
        else: 
            obj = acc_loss + lambd*fair_reg

        model.setObjective(obj, GRB.MINIMIZE)

        self.model = model 
        self.p, self.q, self.c, self.u = p, q, c, u
        self.w, self.w_1, self.w_2, self.w_nm, = w, w_1, w_2, w_nm
        self.z = z
        self.acc_loss, self.fair_reg = acc_loss, fair_reg

        return model 
    
    def set_start_sol(self, sol, X, m): 
        n = X.shape[0]

        for v in range(self.num_node):
            for j in range(self.d): 
                self.p[v,j].start = sol['p'][j,v]
            self.c[v].start = sol['c'][v]
            self.q[v].varHintVal = sol['q'][v]

        missing =  m @ sol['p']   # n x |V|
        a = (missing==0)
        w_nm_tmp = ((1-m)*X)@ sol['p'] <= np.kron(np.ones((n,1)), sol['q'].reshape((1,len(sol['q']))))
        w_1_tmp = a & w_nm_tmp

        a = (missing>0)
        b = np.kron(np.ones((n,1)), sol['c'].reshape((1,len(sol['c'])))).astype(bool)
        w_2_tmp =  a & b
        
        w_tmp = w_1_tmp | w_2_tmp
        z_tmp = w_to_z(w_tmp, self.D)

        
        for i in range(n):
            for v in range(self.num_node):
                self.w[i,v].start = w_tmp[i,v]
                self.w_1[i,v].varHintVal = w_1_tmp[i,v]
                self.w_2[i,v].start = w_2_tmp[i,v]
                self.w_nm[i,v].varHintVal = w_nm_tmp[i,v]

        for i in range(n):
            for l in range(self.num_leaf):
                if l == z_tmp[i]: 
                    self.z[i,l].start = 1
                else: 
                    self.z[i,l].start = 0


    def run_mip(self, params={}, filename='default.lp'):
        print('Starting Optimization')
        for key in params:
            self.model.setParam(key, params[key])

        self.model.write(filename)
        self.model.optimize()

        sol = {}
        sol['p'] = np.array([[int(self.p[v,j].x) for v in range(self.num_node)] for j in range(self.d)])
        sol['q'] = np.array([self.q[v].x for v in range(self.num_node)])
        sol['c'] = np.array([int(self.c[v].x) for v in range(self.num_node)])
        sol['u'] = np.array([int(self.u[i].x) for i in range(self.num_leaf)])
        sol['acc_loss'] = self.acc_loss.getValue()
        sol['fair_reg'] = self.fair_reg.x

        self.sol = sol 

        return sol 

    
    def predict(self, X, m):
        n = X.shape[0]

        missing =  m @ self.sol['p']   # n x |V|
        a = (missing==0)
        b = ((1-m)*X)@ self.sol['p'] <= np.kron(np.ones((n,1)), self.sol['q'].reshape((1,len(self.sol['q']))))
        w_1 = a & b

        a = (missing>0)
        b = np.kron(np.ones((n,1)), self.sol['c'].reshape((1,len(self.sol['c'])))).astype(bool)
        w_2 =  a & b
        w = (w_1 | w_2).astype(int)

        z = w_to_z(w, self.D)
        y_hat = [self.sol['u'][int(z[i])] for i in range(n)]

        return y_hat


    def predict_score(self, X, m, y, metric='acc'): 
        y_hat = self.predict(X,m) 
        if metric == 'acc':
            return 1-np.count_nonzero(y_hat-y)/len(y)
        
        c= confusion_matrix(y, y_hat)
        if metric == 'fnr': 
            return c[1][0]/(c[1][0]+c[1][1])
        if metric == 'fpr': 
            return c[0][1]/(c[0][0]+c[0][1])

            
    def pickle_tree(self, filename): 
        with open(filename, 'wb') as handle:
            pickle.dump(self,handle)



class EnsembleTree(object):
    def __init__(self, trees): 
        self.trees = trees

    def predict(self, X, m):
        y_hats = np.array([t.predict(X, m) for t in self.trees])
        return np.array([stats.mode(y_hats[:,i])[0][0] for i in range(y_hats.shape[1])])






def run_ensemble_tree(D, num_tree, batch_size, lambd, t_limit, fair, input_file, seed):
    with open(input_file, 'rb') as handle: 
        data = pickle.load(handle)

    X_orig = data['X']
    y_orig = data['y']
    m_orig = data['m']
    S_orig = data['S']

    X_train, X_test, y_train, y_test, m_train, m_test, S_train, S_test = train_test_split(X_orig, y_orig, m_orig, S_orig, test_size=0.3, random_state=seed)

    mip_tree = DecisionTree(D)

    sol = None
    trees = [] 
    acc_list = []

    for i in range(num_tree):
        
        X, y, m, S = get_mini_batch(X_train, y_train, m_train, S_train, batch_size=batch_size)
        X = np.nan_to_num(X, copy=False, nan=-999)

        mip_tree = DecisionTree(D)
        mip_tree.build_model(X, m, y, fair=fair, lambd = lambd, S = S)

        if sol is not None:
            mip_tree.set_start_sol(sol, X, m)
        
        params = {'TimeLimit': t_limit, 'MIPFocus': 1}
        sol = mip_tree.run_mip(params)
        acc = mip_tree.predict_score(X_test, m_test, y_test)
        trees.append(mip_tree)
        acc_list.append(acc)
        
        print('Iteration', i, ':')
        print('Accuracy:', acc )
        print('============================================================================\n')

    ens_tree = EnsembleTree(trees)
    metrics = eval_binary_metrics(ens_tree, X_test, y_test, S_test, missing=True, m_eval=m_test)

    print(metrics)


    output_file = 'forests/d3trees_seed{}/{}_L{:.2f}_N{}_b{}.pkl'.format(seed, fair, lambd, num_tree, batch_size)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        sol_trees = [t.sol for t in trees]
        pickle.dump(sol_trees, f)


def sol_to_forest(sol_trees, D):
    trees= [] 
    for i in range(len(sol_trees)):
        tree =DecisionTree(D) 
        tree.sol = sol_trees[i]
        trees.append(tree)

    return EnsembleTree(trees)