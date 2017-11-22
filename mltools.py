import pandas as pd
import numpy as np
import pickle,itertools,sqlite3

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import StandardScaler


class CustomScaler(StandardScaler):

    def fit(self,X):
        K = np.int64(X.shape[1]/2)
        StandardScaler.fit(self,X)
        m = (self.mean_[:K] + self.mean_[K:])/2
        self.mean_[:K] = m
        self.mean_[K:] = m
        
        v = (self.scale_[:K]**2 + self.scale_[K:]**2)/2
        self.scale_[:K] = np.sqrt(v)
        self.scale_[K:] = np.sqrt(v)
        
        return self

class Transformer:

    def __init__(self,polynomialFeatures=True):
        self.polynomialFeatures = polynomialFeatures

    def fit_transform(self,XTrain):
        myRange = np.arange(XTrain.shape[0])
        offset = XTrain.shape[0]
        addOffset = np.random.rand(XTrain.shape[0])>0.5
        
        idx = myRange + offset*addOffset
        
        self.scaler = CustomScaler().fit(XTrain)
        
        XTrain = pd.DataFrame(self.scaler.transform(XTrain),
                                  columns=XTrain.columns)

        
        if self.polynomialFeatures:
            XTrain = polynomial_features(XTrain)
        
        return XTrain

    def transform(self,XTest):

        XTest =  pd.DataFrame(self.scaler.transform(XTest),
                                  columns=XTest.columns)

        if self.polynomialFeatures:
            XTest = polynomial_features(XTest)

        return XTest
    

def build_matchup(fighter1,fighter2):
    ''' 
    Builds a single feature vector for a fight between two fighters.
    Note that this only considers fighter stats at the present moment
    in time (except age... maybe)

    Parameters
    ----------
    fighter1 : dict
    	       fighter dictionary containing stats on the first fighter
    fighter2 : dict
    	       fighter dictionary containing stats on the second fighter

    Returns
    -------
    X : pd.DataFrame
    	A single-row data frame corresponding to our feature vector   
    '''

    feature_list = ['height','reach','sapm','slpm','stance','stracc',\
                    'strdef','subavg','tdacc','tdavg','tddef',\
                    'weight','dob','wins','losses','cumtime']

    tagged_features = ['f1_'+f for f in feature_list]+['f2_'+f for f in feature_list]
    X = pd.DataFrame(columns=tagged_features,dtype=float)

    for feature in feature_list:
        f1='f1_'+feature
        f2='f2_'+feature
        
        cf1=fighter1[feature]
        cf2=fighter2[feature]
            
        if feature == 'dob':
            if cf1=='--':
                cf1=np.nan
            else:
                cf1=2017-float(cf1[-4:])

            if cf2=='--':
                cf2=np.nan
            else:
                cf2=2017-float(cf2[-4:])

        if feature == 'stance':
            if cf1 == 'Orthodox':
                cf1=0.0
            else:
                cf1=1.0
                
            if cf2 == 'Orthodox':
                cf2=0.0
            else:
                cf2=1.0

        X.loc[0,f1]=float(cf1)
        X.loc[0,f2]=float(cf2)



    return X

def get_fighters(dbfile='fighterdb.sqlite'):

    conn = sqlite3.connect(dbfile)

    cur = conn.cursor()

    dataList = sql_to_list('Fighters',cur)

    dataDict = {}

    for entry in dataList:
        
        name = entry.pop('name')
        _ = entry.pop('id')

        dataDict[name] = entry

    conn.close()
    return dataDict

def strip_name(name):
    newName = name.lower().replace(' ','').replace(',','').replace('.','').replace('-','')
    return newName

class FightClassifier(RandomForestClassifier):

    def __init__(self,transformer=None):
        
        super().__init__(max_depth=6,n_estimators=50)
        self.displayDict = {}
        self.transformer = transformer

        oldFighters = get_fighters()
        self.fighters = {}

        for f in oldFighters:
            if f.lower() == f:
                continue
            
            f2 = strip_name(f)
            self.fighters[f2] = oldFighters[f]
            self.fighters[f2]['name'] = f

        self.data_repair()


    def predict_fight(self,fighter1,fighter2):

        f1 = strip_name(fighter1)
        f2 = strip_name(fighter2)

        if f1 not in self.fighters:
            return '%s not found'%fighter1

        if f2 not in self.fighters:
            return '%s not found'%fighter2

        x = build_matchup(self.fighters[f1],
                                    self.fighters[f2])

        if self.transformer is not None:
            x = self.transformer.transform(x)

        y = self.predict_proba(x)[:,1]

        return y

    def data_repair(self):

        reach = np.array([f['reach'] for _,f in self.fighters.items()])
        height = np.matrix([f['height'] for _,f in self.fighters.items()])
        idx = reach != 0.0

        self.reachRegressor = LinearRegression().fit(height[0,idx].T,reach[idx])

        for f in self.fighters:
            if self.fighters[f]['reach'] == 0.0:
                self.fighters[f]['reach'] = \
                  self.reachRegressor.predict(self.fighters[f]['height'])


def polynomial_features(X):
    
    cols = [k+i for k in ['f1_','f2_'] for i in ['stance','height','tdavg','stracc']]

    colCombinations = list(itertools.combinations(cols,2))
    
    W = [X.loc[:,c[0]]*X.loc[:,c[1]] for c in colCombinations]
    colNames = [c[0]+'*'+c[1] for c in colCombinations]

    Wdf = pd.concat(W,axis=1)
    
    Wdf.columns = colNames

    Z = pd.concat([X,Wdf],axis=1)
    return Z

    

def train_classifier(loadData=True,polynomialFeatures=True):

    classifier = FightClassifier()
    
    if loadData:
        X = pd.read_csv('matchup_train.csv',header=0,index_col=0)
        y = pd.read_csv('outcome_train.csv',header=0,index_col=0).iloc[:,0]

    else:
        import build_dataset
        X,y = build_datasets.build_features(classifier.fighters)
        X.to_csv('matchup_train.csv',header=True)
        pd.Series(y).to_csv('outcome_train.csv',header=True)

        
    classifier.transformer = Transformer(polynomialFeatures)
    
    X = classifier.transformer.fit_transform(X)

    classifier.fit(X,y)

    return classifier


def sql_to_list(tableName,cur):

    pragmaExpr = 'PRAGMA table_info( %s )'%tableName
    
    cur.execute(pragmaExpr)

    columnData = cur.fetchall()

    columnNames = [t[1] for t in columnData]

    selectExpr = 'SELECT * FROM %s'%tableName
    cur.execute(selectExpr)

    tableData = cur.fetchall()

    dataList = []
    
    for entry in tableData:
        currentFight = {name:entry[i] for i,name in enumerate(columnNames)}
                
        dataList.append(currentFight)

    return dataList


def build_features(fighters):

    fights = []
    for key in fighters:
        
        currentFights = get_fights(fighters[key]['name'])
        
        fights.extend(currentFights)

    X = []

    y = []

    for fight in fights:
        f1Name = fight['fighter1']
        f2Name = fight['fighter2']
        f1,f2 = strip_name(f1Name),strip_name(f2Name)

        if (strip_name(f1Name) not in fighters) or (strip_name(f2Name) not in fighters.keys()):
            continue
        
        currentFeatureVector = build_matchup(fighters[f1],\
                                             fighters[f2])

        X.append(currentFeatureVector)

        y.append(np.double(fight['winner'] == f2Name))
        

    X = pd.concat(X)
    
    y = np.array(y)

    
    # if we're missing date of birth for one fighter, set their date of births to be the same
    # this is not a good way of solving this, but the idea is that if you don't know better,
    # just assume that two fighters are the same age
    missingF1DobIdx = np.isnan(X.f1_dob)
    missingF2DobIdx = np.isnan(X.f2_dob)
    missingBothDobIdx = missingF1DobIdx & missingF2DobIdx
    X.loc[missingF1DobIdx,'f1_dob'] = X.loc[missingF1DobIdx,'f2_dob'].values
    X.loc[missingF2DobIdx,'f2_dob'] = X.loc[missingF2DobIdx,'f1_dob'].values
    X.loc[missingBothDobIdx,'f1_dob'] = 1988
    X.loc[missingBothDobIdx,'f2_dob'] = 1988
        
    return X,y

def get_fights(fighter,dbfile='fighterdb.sqlite'):
    '''
    Takes a fighters dict, processes the fights and returns a list of
    fights

    Parameters
    ----------
    fighter : str
    	      Name of the fighter
    	       
	      

    Returns
    -------
    fights : list
    	     A list of all the fights from fighters   
    '''

    # this lets us optionally pass a cursor instead of the database file name
    if type(dbfile)==str:
        conn = sqlite3.connect(dbfile)
        cur = conn.cursor()
    else:
        cur = dbfile
        
    data = sql_to_list('Fights',cur)

    # So we need to check whether we have repetitions in the Fights and Fighters,
    # and also work out the best way to structure this data.
    
    fights = [fight for fight in data if fighter in [fight['fighter1'],fight['fighter2']]]

    if type(dbfile)==str:
        conn.close()

    return fights


if __name__ == "__main__":

    classifier = train_classifier()

    with open('classifier.save','wb') as f:
        pickle.dump(classifier,f)
        
