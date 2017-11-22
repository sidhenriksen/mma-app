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
