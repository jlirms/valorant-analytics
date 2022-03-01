import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib import colors
from skimage import io
from skimage.transform import resize
from sklearn import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score, train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#%% Generic data functions


class MatchesData():
    """This class:
    - Loads in all the matches, MatchIDs of collected data
    - Splits to train, test and validation by matchIds
    For the paper we train on the 'Main Events', validate on 'Challenge Playoffs' and take the final test on 'Last Chance'
    """

    def __init__(self):
        self.df_matches = pd.read_csv('DATA_XY/VCT3NAmatches.csv')
        self.df_ids = pd.read_csv('DATA_XY/playerId.csv',index_col = 'Player Id')
        self.collected_files = [
            'VCT North America 2021 - Stage 3 Challengers 2 - Open Qualifier',
            'VCT North America 2021 - Stage 3 Challengers 2 - Main Event',
            'VCT North America 2021 - Stage 3 Challengers Playoffs',
            'VCT North America 2021 - Stage 3 Challengers 1 - Main Event',
            'VCT North America 2021 - Stage 3 Challengers 1 - Open Qualifier',
            'VCT North America 2021 - Last Chance Qualifier'
        ]

        train_events = [f for f in self.collected_files if 'Main Event' in f ] #or 'Nerd' in f
        valid_events = [f for f in self.collected_files if 'Challengers Playoffs' in f]
        test_events = [f for f in self.collected_files if 'Last Chance' in f]
        valid_test = valid_events + test_events
        train_valid = train_events + valid_events

        self.train_matches = self.df_matches[self.df_matches['Event Name'].isin(train_events)]['Match Id'].values
        self.valid_matches = self.df_matches[self.df_matches['Event Name'].isin(valid_events)]['Match Id'].values
        self.test_matches = self.df_matches[self.df_matches['Event Name'].isin(test_events)]['Match Id'].values
        self.train_valid_matches = self.df_matches[self.df_matches['Event Name'].isin(train_valid)]['Match Id'].values 
        self.valid_test_matches = self.df_matches[self.df_matches['Event Name'].isin(valid_test)]['Match Id'].values 


class DuelsData():
    """
    This class:
        - Loads in the data given and shifts it to fit the geometry of the map
        - Encodes the gun names as integers, with Vandal and Phantom being 2901 and 2900 respectively
        - Separate to duels taken from the attackers perspective (eg killed or got killed by a defender) and the defenders perspective
        - Get training data, depending on which perspective you want
    """

    shift_map = {'Haven':{'xshift': -50, 'yshift':-10, 'xmax': 915, 'ymin': -1015}, 
          'Icebox':{'xshift': -36.5, 'yshift':27, 'xmax': 915, 'ymin': -1015},
          'Ascent':{'xshift': -53.5, 'yshift':-12, 'xmax': 955, 'ymin': -1025},
          'Bind':{'xshift': -112, 'yshift':15, 'xmax': 865, 'ymin': -1038},
          'Split':{'xshift': -1.5, 'yshift':-23, 'xmax': 1010, 'ymin': -956},
          'Breeze':{'xshift': 2, 'yshift':-12, 'xmax': 992, 'ymin': -975}
          }

    def __init__(self, filename, mapname = None, verbose = False):
        """Inits with filename, parses file type and mapname
        TODO Parse filetype, mapname
        """
        self.filename = filename
        self.mapname = mapname
        self.verbose = verbose
        self.filetype = '.csv'
        self.dfall = self.read_shift_dfall()
        self.dfATKwon, self.dfATKlost = None,None
        self.dfDEFwon, self.dfDEFlost = None, None
        self.dfkills = None
        self.dfrefs = None

    @staticmethod
    def downcast(df, verbose=False):
        if verbose:
            start_mem = df.memory_usage().sum() / (1024 ** 2)
            
        int_columns = df.select_dtypes(include=["int"]).columns
        float_columns = df.select_dtypes(include=["float"]).columns
    
        for col in int_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        if verbose:
            end_mem = df.memory_usage().sum() / (1024 ** 2)
            print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem))
            
        return df 
    
    def read_shift_dfall(self):
        """
        Read data and shift according to maps

        TODO: change index_col on csv file
        """
        if self.filetype == '.pickle':
            dfall = pd.read_pickle(self.filename)
        
        elif self.filetype == '.csv':
            dfall = pd.read_csv(self.filename, index_col=0) ##update this

        if self.mapname is not None:
            dfall = dfall[dfall.MapName == self.mapname]
        
        dfall = dfall[~dfall[['kx', 'ky']].isnull().any(axis = 1)]

        for singlemap in self.shift_map.keys():
            dfall.loc[dfall['MapName'] == singlemap, 'kx'] = dfall.loc[dfall['MapName'] == singlemap, 'kx'] +  self.shift_map[singlemap]['xshift'] 
            dfall.loc[dfall['MapName'] == singlemap, 'rx'] = dfall.loc[dfall['MapName'] == singlemap, 'rx'] +  self.shift_map[singlemap]['xshift'] 
            dfall.loc[dfall['MapName'] == singlemap, 'ky'] = dfall.loc[dfall['MapName'] == singlemap, 'ky'] +  self.shift_map[singlemap]['yshift'] 
            dfall.loc[dfall['MapName'] == singlemap, 'ry'] = dfall.loc[dfall['MapName'] == singlemap, 'ry'] +  self.shift_map[singlemap]['yshift'] 

        ## Rename columns
        dfall = dfall.rename(columns = {
                        'ATKloadoutValue':'atkLoadout',
                        'DEFloadoutValue':'defLoadout',
                        'ATKNumExpensiveGuns': 'atkNumGuns',
                        'DEFNumExpensiveGuns':'defNumGuns',
                        'ATKAlive': 'atkAlive',
                        'DEFAlive': 'defAlive',
                        'roundTime': 'roundTime',
                        'Planted': 'isPlanted',
                        'Site': 'sitePlanted',
                        'SpikeBeepsPerSecond': 'spikeBPS',
                        'ATKWonLast': 'atkWonLast',
                        'DEFWonLast': 'defWonLast',
                        'roundNumber': 'roundNumber',
                        'roundHalf': 'roundHalf',
                        'MatchID':'matchId',
                        'SeriesID':'seriesId',
                        'Date': 'datesSTR'})

        guns_cost_S3 = {
                'Odin':3200,
                'Ares':1550,
                'Vandal':2901,
                'Bulldog':2050,
                'Phantom':2900,
                'Judge':1850,
                'Bucky':900,
                'Frenzy':450,
                'Classic':0,
                'Ghost':500,
                'Sheriff':800,
                'Shorty':200,
                'Operator':4700,
                'Guardian':2250,
                'Marshal':950,
                'Spectre':1600,
                'Stinger':950,
                ##abilities
                'bladestorm': -4700,
                'showstopper': -4000,
                'hunters-fury': -3000,
                'orbital-strike': -2000,
                'shock-dart': -400,
                'boombot': -400,
                'nanoswarm': -400,
                'turret': -400,
                'paint-shells': -400,
                'hot-hands': -400,
                'fragment': -400,
                'snake-bite': -400,
                'blast-pack': -400,
                'molly': -400,
                'trailblazer': -400,
                'aftershock': -400,
                'blaze': -400,
                }

        for gunCol in ['weaponK', 'weaponKS', 'weaponR']:
            ##fill na for all other weapons
            dfall[gunCol] = dfall[gunCol].map(guns_cost_S3)

        dfall['MapInt'] = dfall['MapName'].map({
            'Haven' : 0,
            'Icebox' : 1,
            'Ascent' : 2,
            'Bind' : 3,
            'Split' : 4,
            'Breeze' : 5,
        })

        dfall = dfall[dfall['weaponK'] >= 0]

        return dfall.pipe(self.downcast)
    

    def seperate_atkdef(self, isAtkSide = 1):
        """
        This function: 
        Takes all data, dfall and separates it into 2

        when isAtkSide = 1
        first df is duels attackers won, player is the attacker
        second df is duels attackers lost, player is still attacker, enemy is defender


        when isAtkSide = 0
        first df is duels defender won, player is defender, with killerATK = 0
        second df is duels defenders lost, enemy is attacker

        """
        df_atkwon = self.dfall[self.dfall["killerATK"] == isAtkSide]
        df_atkwon = df_atkwon.rename(columns = {
            'kx':'px',
            'ky':'py',
            'rx':'ex',
            'ry':'ey',
            'weaponK':'pgun',
            'weaponR':'egun',
            'weaponKS':'sgun',
            'armorK':'parmor',
            'armorR':'earmor',
            'agentK':'pagent',
            'agentR':'eagent',
            'referencePlayerId':'enemyId'
        })

        df_atklost = self.dfall[self.dfall["killerATK"] == (1-isAtkSide)]
        df_atklost = df_atklost.rename(columns = {
            'kx':'ex',
            'ky':'ey',
            'rx':'px',
            'ry':'py',
            'weaponK':'egun',
            'weaponR':'pgun',
            'weaponKS':'sgun',
            'armorK':'earmor',
            'armorR':'parmor',
            'agentK':'eagent',
            'agentR':'pagent',
            'playerId':'enemyId',
            'referencePlayerId':'playerId'
        })

        df_atklost = df_atklost[df_atkwon.columns]
        return df_atkwon, df_atklost

    def dfatk(self):
        """Using seperate_atkdef, combines the winning and losing fights
        returns single dataframe with Won column = if attacking player won"""
        self.dfATKwon, self.dfATKlost = self.seperate_atkdef(isAtkSide=1)
        atk_df = pd.concat([self.dfATKwon,self.dfATKlost]).sort_index()
        atk_df = atk_df.rename(columns = {'killerATK': 'Won'})
        atk_df['isATK'] = 1
        return atk_df

    def dfdef(self):
        """Using seperate_atkdef, combines the winning and losing fights
        returns single dataframe with Won column = if defending player won"""
        self.dfDEFwon, self.dfDEFlost = self.seperate_atkdef(isAtkSide=0)

        ## Flip the column that will become "Won"
        self.dfDEFwon['killerATK'] = 1 - self.dfDEFwon['killerATK']
        self.dfDEFlost['killerATK'] = 1 - self.dfDEFlost['killerATK']
        def_df = pd.concat([self.dfDEFwon,self.dfDEFlost]).sort_index() 
        def_df = def_df.rename(columns = {'killerATK': 'Won'})
        def_df['isATK'] = 0
        return def_df

    def train_val(self, train_ids, valid_ids, isAtkSide = 0):
        """using dfatk or dfdef, specified by isAtkSide
        split data using one of the ID columns and specified percentage
        returns training X,y and validation X, y"""

        if isAtkSide == 0:
            dftv = self.dfdef()
        elif isAtkSide == 1:
            dftv = self.dfatk()

        dftrain = dftv[dftv["matchId"].isin(train_ids)]
        dfval = dftv[dftv["matchId"].isin(valid_ids)]
        
        if self.verbose:
            print("Training data size {:.1f}% of rows".format( 100*dftrain.shape[0] / (dftrain.shape[0] + dfval.shape[0])))

        return dftrain, dfval

    def generate_data_grid(self, N = 5, mapName = 'Ascent'):
        """This function:
        Generates a 2-D grid using meshgrid
        Will provide the XY coordinates for us to calculate the Gaussian"""
        yend =  -self.shift_map[mapName]['ymin']
        xend = self.shift_map[mapName]['xmax']
        X = np.linspace(0, xend, int(xend/N))
        Y = np.linspace(0, yend, int(yend/N))
        X, Y = np.meshgrid(X, Y)

        pos = np.zeros(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        
        griddf = pd.DataFrame(np.around(np.c_[X.ravel(), Y.ravel()], 2), 
                                columns = ['px','py'])
        return griddf

#%% Useful functions
def plotCalibrationCurve(y_true, y_pred, title_str = 'VCT NA', n_bins = 10):
    """Takes in model with .predict_proba() method
    Calculates calibration curve characteristics"""

    print('Brier Score Loss: ', brier_score_loss(y_true, y_pred))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins = n_bins)

    fig = plt.figure(figsize=(7, 5))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "w:", label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label = 'Model Predictions')

    ax2.hist(y_pred, range = (0,1), bins = n_bins, histtype="step")

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots ' + title_str)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

    return fig

def printPermutationImpt(clf, X_test, y_test):
    """Takes in classifier, X and y value
    prints out permutation importance"""
    r = permutation_importance(clf, X_test, y_test,
                        n_repeats=10,
                        random_state=0)


    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")
    return None

def plotShapSummary(clf, X_train):
    explainer = shap.TreeExplainer(clf); # data=test_x, model_output="probability");
    shap_values = explainer.shap_values(X_train)#[1]; # index 1 pulls the P(y=1) SHAP values for RandomForest.
    return shap.summary_plot(shap_values, X_train, show=True,
                    max_display=10,
                    plot_type="violin")

class PredATKDEF():
    def __init__(self, DataSource, train_matches, test_matches , features, model, df_ids):
        self.Data = DataSource
        self.train_matches = train_matches
        self.test_matches = test_matches
        self.features = features
        self.modelATK = clone(model)
        self.modelDEF = clone(model)
        self.df_ids = df_ids
    
    def fitpred_ATKDEF(self):
        df_train, df_valid = self.Data.train_val(self.train_matches, self.test_matches, isAtkSide=0)
        df_train = df_train.dropna()
        self.XDEF = df_train[self.features]
        y = df_train['Won']

        df_valid = df_valid.dropna()
        X_valid = df_valid[self.features]
        y_validDEF = df_valid['Won']

        self.modelDEF.fit(self.XDEF,y)
        y_predDEF= self.modelDEF.predict_proba(X_valid)[:,1]

        dfexpDEF = df_valid.merge(self.df_ids, left_on = 'playerId', right_index = True, how = 'left')
        dfexpDEF['PROBA'] = self.modelDEF.predict_proba(dfexpDEF[self.features])[:,1]
        dfexpDEF['DIFF'] = dfexpDEF['Won'] - dfexpDEF['PROBA']

        #############################

        df_train, df_valid = self.Data.train_val(self.train_matches, self.test_matches, isAtkSide = 1)
        df_train = df_train.dropna()
        self.XATK = df_train[self.features]
        y = df_train['Won']

        df_valid = df_valid.dropna()
        X_valid = df_valid[self.features]
        y_validATK = df_valid['Won']

        self.modelATK.fit(self.XATK, y)
        y_predATK = self.modelATK.predict_proba(X_valid)[:,1]

        dfexpATK = df_valid.merge(self.df_ids, left_on = 'playerId', right_index = True, how = 'left')
        dfexpATK['PROBA'] = self.modelATK.predict_proba(dfexpATK[self.features])[:,1]
        dfexpATK['DIFF'] = dfexpATK['Won'] - dfexpATK['PROBA']

        y_true = pd.concat([y_validATK, y_validDEF]).values
        y_pred = np.hstack([y_predATK, y_predDEF])
        dfexp = pd.concat([dfexpATK, dfexpDEF])

        return y_true, y_pred, dfexp


def plotHeatMap(dfpred, N = 120, midpoint = 0.5, c_range = 0.5):
    MAP = 'Ascent'
    map_image = io.imread('IMAGES/MAP_PNGS/{}_contour.png'.format(MAP))
    SIZE_FIG =(14, 13.2)
    shift_map = {'Haven':{'xshift': -50, 'yshift':-10, 'xmax': 915, 'ymin': -1015}, 
            'Icebox':{'xshift': -36.5, 'yshift':27, 'xmax': 915, 'ymin': -1015},
            'Ascent':{'xshift': -53.5, 'yshift':-12, 'xmax': 955, 'ymin': -1025},
            'Bind':{'xshift': -112, 'yshift':15, 'xmax': 865, 'ymin': -1038},
            'Split':{'xshift': -1.5, 'yshift':-23, 'xmax': 1010, 'ymin': -956},
            'Breeze':{'xshift': 2, 'yshift':-12, 'xmax': 992, 'ymin': -975}
            }

    xmin, xmax = 0, shift_map[MAP]['xmax']
    ymax, ymin = 0, shift_map[MAP]['ymin']
    holes = map_image[:,:,2]
    base = resize(holes, [ymax-ymin, xmax-xmin])
    XMAX = base.shape[1]
    YMAX = base.shape[0]

    xx, yy = np.meshgrid(np.linspace(0, XMAX, int(XMAX/5)),
                        np.linspace(0, YMAX, int(YMAX/5)))
    gridpred = dfpred.values[:,-1]
    gridz = gridpred.reshape(xx.shape)
    fig = plt.figure(figsize=SIZE_FIG)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)

    grid_show = resize(gridz, [ymax-ymin, xmax-xmin])
    ax.imshow(grid_show, cmap = 'PiYG', 
                norm = colors.TwoSlopeNorm(vmin=midpoint-c_range, vcenter=midpoint, 
                                                vmax=midpoint+c_range), 
                alpha = base) ## actual grid

    return None


def getPlayersdf(dfexp, games_threshold = 5):
    playersG = dfexp.groupby(['playerId', 'Ign'])

    dfplayers = playersG.agg(
        team_name = pd.NamedAgg(column = "Team", aggfunc = "last"),
        tot_duels = pd.NamedAgg(column = "DIFF", aggfunc = "count"),
        tot_rounds = pd.NamedAgg(column = 'roundId', aggfunc = "nunique"),
        tot_games = pd.NamedAgg(column = 'matchId', aggfunc = 'nunique'),
        exp_kills = pd.NamedAgg(column="PROBA", aggfunc="sum"), 
        tot_kills = pd.NamedAgg(column="Won", aggfunc="sum"),
        wins_above_expected =pd.NamedAgg(column="DIFF", aggfunc="mean"),
        med_diff =pd.NamedAgg(column="DIFF", aggfunc="median"),
        kills=pd.NamedAgg(column="Won", aggfunc="sum"),
    ) 
    dfplayers['KDA'] = dfplayers['kills']/(dfplayers['tot_duels'] - dfplayers['kills'])
    dfplayers = dfplayers.query('tot_games >= {}'.format(games_threshold)).dropna()
    print("Total Players: {}".format(dfplayers.shape[0]))
    return dfplayers


def getribURL(df):
    df['ribURL'] = 'https://rib.gg/series/' + df['seriesId'].astype(str) + '?match=' + df['matchId'].astype(str) + '&round=' + df['roundNumber'].astype(str) + '&tab=replay' 
    return df
