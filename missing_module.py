import pandas as pd
import numpy as np
import pickle 

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit




def binom_se(n, p):
    return np.sqrt(p*(1-p)/n)


def binary_score(y_true, y_hat, metric='acc'): 
    c= confusion_matrix(y_true, y_hat)

    if metric == 'fnr':
        return c[1][0]/(c[1][0]+c[1][1])
    elif metric == 'fpr': 
        return c[0][1]/(c[0][0]+c[0][1])
    elif metric == 'acc': 
        return (c[0][0]+c[1][1])/len(y_true)

    
def eval_binary_metrics(pred_model, X_eval, y_eval, sens_attr, missing=False, m_eval=None): 
    # For binary labels #
    metrics = {}
    if len(X_eval) != len(sens_attr):
        raise ValueError
    
    for g_sel in [0,1]: 
        X_eval_g = X_eval[sens_attr == g_sel]
        y_eval_g= y_eval[sens_attr == g_sel]
        if missing: 
            m_eval_g= m_eval[sens_attr == g_sel]
            y_pred_g = pred_model.predict(X_eval_g, m_eval_g)
        else:
            y_pred_g = pred_model.predict(X_eval_g)
    
        c= confusion_matrix(y_eval_g, y_pred_g)
        metrics[g_sel] = {}
        metrics[g_sel]['Size'] = c[0][0]+c[0][1]+c[1][0]+c[1][1]
        metrics[g_sel]['True Parity'] = (c[1][0]+c[1][1])/(c[0][0]+c[0][1]+c[1][0]+c[1][1])
        metrics[g_sel]['Parity'] = (c[0][1]+c[1][1])/(c[0][0]+c[0][1]+c[1][0]+c[1][1])
        metrics[g_sel]['FPR'] = c[0][1]/(c[0][0]+c[0][1])
        metrics[g_sel]['FNR'] = c[1][0]/(c[1][0]+c[1][1])
        metrics[g_sel]['PPV'] = c[1][1]/(c[0][1]+c[1][1])
        metrics[g_sel]['NPV'] = c[0][0]/(c[0][0]+c[1][0])
        metrics[g_sel]['Accuracy'] = accuracy_score(y_eval_g, y_pred_g)
          
    return metrics


def eval_binary_metrics_se(pred_model, X_eval, y_eval, sens_attr): 
    # For binary group labels #
    # Return metrics dictionary # 
    metrics = {}
    for g_sel in [0,1]: 
        X_eval_g = X_eval[sens_attr == g_sel]
        y_eval_g= y_eval.loc[X_eval_g.index]
        y_pred_g = pred_model.predict(X_eval_g)
    
        c= confusion_matrix(y_eval_g, y_pred_g)
        metrics[g_sel] = {}
        metrics[g_sel]['Size'] = c[0][0]+c[0][1]+c[1][0]+c[1][1]

        metrics[g_sel]['True Parity'] ={}
        metrics[g_sel]['True Parity']['val'] = (c[1][0]+c[1][1])/metrics[g_sel]['Size']
        metrics[g_sel]['True Parity']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['True Parity']['val'])

        metrics[g_sel]['Parity'] ={}
        metrics[g_sel]['Parity']['val'] = (c[0][1]+c[1][1])/metrics[g_sel]['Size']
        metrics[g_sel]['Parity']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['Parity']['val'])

        metrics[g_sel]['FPR'] ={}
        metrics[g_sel]['FPR']['val'] = c[0][1]/(c[0][0]+c[0][1])
        metrics[g_sel]['FPR']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['FPR']['val'])

        metrics[g_sel]['FNR'] ={}
        metrics[g_sel]['FNR']['val'] = c[1][0]/(c[1][0]+c[1][1])
        metrics[g_sel]['FNR']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['FNR']['val'])

        metrics[g_sel]['PPV'] ={}
        metrics[g_sel]['PPV']['val'] = c[1][1]/(c[0][1]+c[1][1])
        metrics[g_sel]['PPV']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['PPV']['val'])

        metrics[g_sel]['NPV'] ={}
        metrics[g_sel]['NPV']['val'] = c[0][0]/(c[0][0]+c[1][0])
        metrics[g_sel]['NPV']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['NPV']['val'])

        metrics[g_sel]['Accuracy'] ={}
        metrics[g_sel]['Accuracy']['val'] = (c[0][0]+c[1][1])/metrics[g_sel]['Size']
        metrics[g_sel]['Accuracy']['se'] = binom_se(metrics[g_sel]['Size'], metrics[g_sel]['Accuracy']['val'])

    
    return metrics




def metric_se_to_diff(metrics, se=True):
    # For binary group labels #

    metric_diff ={}
    metric_labels = ['True Parity', 'Parity', 'FPR', 'FNR', 'PPV', 'NPV', 'Accuracy']
    for m in metric_labels:
        if se:
            metric_diff[m] = {}
            metric_diff[m]['val'] = metrics[0][m]['val'] - metrics[1][m]['val']
            metric_diff[m]['se'] = np.sqrt(metrics[0][m]['se']**2 + metrics[1][m]['se']**2)
        else: 
            metric_diff[m] = metrics[0][m] - metrics[1][m]
        
    return metric_diff


def metric_dict_to_table(metric_dict):
    metric_table ={}
    metric_labels = ['True Parity', 'Parity', 'FPR', 'FNR', 'PPV', 'NPV', 'Accuracy']
    for m in metric_labels: 
        metric_table[m] = '{:.3f} \u00B1 {:.3f}'.format(metric_dict[m]['val'], metric_dict[m]['se'])

    return metric_table


def eval_binary_diff(pred_model, X_eval, y_eval, g_label): 
    metrics = eval_binary_metrics(pred_model, X_eval, y_eval, g_label)
    return metric_se_to_diff(metrics, se=False)



def eval_binary_diff_table(pred_model, X_eval, y_eval, g_label): 
    metrics = eval_binary_metrics_se(pred_model, X_eval, y_eval, g_label)
    return metric_dict_to_table(metric_se_to_diff(metrics))
    

def generate_missing(df, c_label, ms_label, p_ms0, p_ms1, seed=0, true_label = False): 
    '''
    Artifically delete the entry with different probabilities for c_label=0 and c_label =1.
    Features in ms_label will be deleted for the selected rows. 
    Rows are selected at random following the probability p_ms0 and p_ms1. 
    '''
    np.random.seed(seed)
    ms_idx0 = np.random.choice(list(df[df[c_label] ==0].index), int(len(df[df[c_label]==0].index)*p_ms0), replace=False)
    np.random.seed(seed)
    ms_idx1 = np.random.choice(list(df[df[c_label] ==1].index), int(len(df[df[c_label]==1].index)*p_ms1), replace=False)

    df_new = df.copy()
    df_new.loc[ms_idx0, ms_label] = np.NaN
    df_new.loc[ms_idx1, ms_label] = np.NaN
    
    if true_label:
        return df.loc[ms_idx0.tolist()+ms_idx1.tolist(), ms_label], df_new
    else:
        return df_new




def train_model(X_train, y_train, model='logistic'): 
    if model == 'randomforest':
        pred_model =  RandomForestClassifier(max_depth=16, min_samples_leaf=3, random_state=42)
    elif model == 'logistic':
        pred_model = LogisticRegression(multi_class='multinomial')
    elif model == 'svm': 
        pred_model = svm.SVC(kernel='rbf') 
    elif model == 'decision': 
        pred_model = tree.DecisionTreeClassifier(random_state=42, max_depth=3)
        
    return pred_model.fit(X_train, y_train)
    

    
def eval_model(pred_model, X_test, y_test, sens_attr, diff=False): 
    if diff:
        return eval_binary_diff_table(pred_model, X_test, y_test, sens_attr)
    
    return eval_binary_metrics(pred_model, X_test, y_test, sens_attr)



def train_eval_metric(X_train, y_train, X_test, y_test, model='logistic', sens_attr='gender', diff=False): 
    
    pred_model = train_model(X_train, y_train, model)
    return eval_model(pred_model, X_test, y_test, sens_attr, diff)
    



def process_missing_values(df, method='drop', per_group=False, sens_attr=None, ms_attr=None):    

    is_binary = False
    if ms_attr is not None:
        if df[ms_attr].nunique() ==2: 
            is_binary = True
            
    if method == 'drop':
        df_imp = df.dropna()  # Dropping all the rows with "any" missing value
    elif method == 'mean':
        if per_group:
            df0 = df[sens_attr == 0]
            df1 = df[sens_attr == 1]
            df0 = df0.fillna(df0.mean())
            df1 = df1.fillna(df1.mean())
            df_imp = pd.concat([df0, df1], ignore_index=True)
        else:
            df_imp = df.fillna(df.mean())
    elif method == 'knn':      
        imputer = KNNImputer(n_neighbors=5)
        if per_group: 
            df0 = df[sens_attr == 0]
            df1 = df[sens_attr == 1]
            df0 = pd.DataFrame(imputer.fit_transform(df0), columns=df0.columns, index=df0.index)
            df1 = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns, index=df1.index)
            df_imp = pd.concat([df0, df1], ignore_index=True)
        else: 
            df_imp = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        if is_binary: 
            df_imp.loc[ms_attr] = df_imp[ms_attr].round()
            
            
    return df_imp



def balance_data(df, attr, major=1): 
    sample_size = len(df[df[attr] == major]) - len(df[df[attr]==(1-major)])
    if sample_size < 0:
        raise ValueError
    
    np.random.seed(0)
    drop_idx = np.random.choice(df[df[attr]==major].index, sample_size, replace=False)
    return df.drop(drop_idx)


def df_ms_to_pickle(df_ms, sens_attr, filename='df_ms.pkl'): 
    S = df_ms[sens_attr]
    X = df_ms.iloc[:,:-1].copy()
    X = X.drop(columns=[sens_attr])
    y = df_ms.iloc[:,-1].copy()
    m = X.isna().astype(int)
    y[y<0] = 0
    X, y, m, S = np.array(X), np.array(y), np.array(m), np.array(S)
    data = {}
    data['X'] = X
    data['y'] = y
    data['m'] = m
    data['S'] = S
    
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle)
    

def get_mini_batch(*args, batch_size=100):
    n = args[0].shape[0]
    batch_idx = np.random.choice(n, batch_size, replace=False)

    return tuple(a[batch_idx] for a in args)