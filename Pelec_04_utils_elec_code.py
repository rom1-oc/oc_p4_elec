""" Utils """
import os
import math
import warnings
import math
import datetime
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import statsmodels.api as sm
from catboost import Pool, CatBoostRegressor
from xgboost import XGBRegressor
from datetime import datetime
from hyperopt import tpe
from matplotlib import pyplot as plt
from yellowbrick.features import RadViz
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import AlphaSelection
from sklearn.compose import (ColumnTransformer,
                             TransformedTargetRegressor
                            )
from sklearn.preprocessing import (StandardScaler,
                                   OneHotEncoder,
                                   LabelEncoder,
                                   QuantileTransformer
                                  )
from sklearn.datasets import make_regression
from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor
                             ) 
from sklearn.dummy import DummyRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (IterativeImputer,
                            SimpleImputer
                           )
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     cross_validate,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold,
                                     GridSearchCV,
                                     learning_curve
                                    )
from sklearn.metrics import (mean_squared_error, # Regression
                             explained_variance_score, # Regression
                             r2_score, # Regression
                             mean_absolute_error, # Regression
                             max_error, # Regression
                             accuracy_score, # Classification
                             precision_score, # Classification
                             recall_score, # Classification
                             plot_confusion_matrix, # Classification
                             classification_report, # Classification
                            )
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  RidgeCV,
                                  Lasso,
                                  BayesianRidge
                                 )
from sklearn.svm import (LinearSVR,
                         SVR
                        )
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import (Pipeline,
                              make_pipeline
                             )
#from lazypredict.Supervised import LazyClassifier
from string import ascii_letters


    
    
def regression_eval_model_function(X, 
                                   y, 
                                   numerical_list, 
                                   categorical_list, 
                                   target_name, 
                                   regression_type, 
                                   scaler, 
                                   test_size, 
                                   grid_use, 
                                   cv,
                                   **kwargs):
    
    """ General function for model creation and evaluation

    Args:
        X (pd.Dataframe): input
        y (pd.Dataframe): target
        numerical_list (list(str)): list of numerical features in X
        categorical_list (list(str)): list of categorical features in X
        target_name (str): name of target variable
        regression_type (sklearn function): type of regression
        scaler (sklearn function): type of scaler in pipeline
        test_size (int): share of the test set in train_test_split()
        grid_use (bit): Use of grid or not
        cv (int): number of cross-validations

    Returns:
        Prints list of the model parameters and stats 
        Yellowbrick residuals plot
    """  
    plt.rcParams['figure.figsize'] = (9,6)

    #
    stats_cv=[]
    stats_cv_train=[]
    stats_cv_test=[]
    
    #
    regression_type_string = str(regression_type)
    #print(regression_type_string + "\n")
    
    if "RidgeCV" in regression_type_string:
        # RidgeCV parameters
        n_alphas_ridge_cv = 150
        alphas_ridge_cv = np.logspace(-8, 8, n_alphas_ridge_cv)
        regression_type = RidgeCV(alphas_ridge_cv)
        regression_type_string = str(regression_type)
        
    elif regression_type_string == "Lasso()":
        #
        regression_type = Lasso(fit_intercept=False)
        regression_type_string = str(regression_type)
    
    #
    start_at = datetime.now()

    # Target variable
    variable = target_name

    # Split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    
    # Create sub pipeline for numartical features
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_median", SimpleImputer(missing_values=np.nan, strategy='median')),
            ("scaler", scaler),
        ]
    )
    
    # Create sub pipeline for categorical features
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Associate both pipelines
    preprocessor = ColumnTransformer(
        [
            ("numerical", numeric_preprocessor, numerical_list), 
            ("categorical", categorical_preprocessor, categorical_list),
                     
        ]
    )

    # Create general pipeline
    pipe_ = make_pipeline(preprocessor, regression_type)

    # Normaize target
    #target_transformer = QuantileTransformer(output_distribution='normal')
    pipe = TransformedTargetRegressor(regressor=pipe_, func=np.log1p, inverse_func=np.expm1)
    #print(pipe.get_params().keys())
       
    regression_list = ["LinearRegression",
                       "KNeighborsRegressor",
                       "LinearSVR",
                       "SVR",
                       "GradientBoostingRegressor",
                       "XGBRegressor",
                       "RandomForestRegressor",
                       "DummyRegressor",
                       "CatBoostRegressor",
                      ]
    
    if grid_use == 0:
        for regr in regression_list :     
            if regr in regression_type_string : 
                     
                # Fit the model to the data
                pipe.fit(X_train, y_train)
    
                # Compute target
                pred_y = pipe.predict(X_test)
                #print(pipe.get_params().keys())
                
                # Instantiate the linear model and visualizer
                visualizer = ResidualsPlot(pipe)
                #visualizer = PredictionError(pipe)
                visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
                visualizer.score(X_test, y_test)  # Evaluate the model on the test data
                
                
        if "Ridge(" in regression_type_string:

            baseline_mse_error = float(kwargs['bme'])

            # Hyperparameters
            n_alphas_ridge = 100
            alphas_ridge = np.logspace(-2, 14, n_alphas_ridge)

            # Lists to fill
            coefs_ridge = []
            errors_ridge = []

            # Predict
            for a in alphas_ridge:
                pipe.set_params(regressor__ridge__alpha=a)
                pipe.fit(X_train, y_train)
                coefs_ridge.append(pipe.regressor_[1].coef_)
                errors_ridge.append([baseline_mse_error, np.mean((pipe.predict(X_test) - y_test) ** 2)])

            #   
            pred_y = pipe.predict(X_test)
            
            # Instantiate the linear model and visualizer
            visualizer = ResidualsPlot(pipe)
            visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
            visualizer.score(X_test, y_test)  # Evaluate the model on the test data

            # Best hyperparameters
            errors_ridge_min = min(errors_ridge)[1]
            index_ridge_min = np.argmin(errors_ridge, axis=0)[1]
            alpha_ridge_min = alphas_ridge[index_ridge_min]

            # Display hyperparameters
            print("Hyperparameters: ")
            print("Ridge index of minimum Mean square error: " + str(index_ridge_min))
            print("Ridge alpha of minimum Mean square error: " + str(alpha_ridge_min))
            print("Ridge minimum Mean square error: " + str(errors_ridge_min))
            print("Ridge minimum Root mean square error: " + str(math.sqrt(errors_ridge_min))+ "\n")

        elif "Lasso" in regression_type_string:

            baseline_mse_error = float(kwargs['bme'])

            # Hyperparameters
            n_alphas_lasso = 100
            alphas_lasso = np.logspace(-2, 10, n_alphas_lasso)

            # Lists to fill
            coefs_lasso = []
            errors_lasso = []

            # Predict
            for a in alphas_lasso:
                pipe.set_params(regressor__lasso__alpha=a)
                pipe.fit(X_train, y_train)
                coefs_lasso.append(pipe.regressor_[1].coef_)
                errors_lasso.append([baseline_mse_error, np.mean((pipe.predict(X_test) - y_test) ** 2)])

            #   
            pred_y = pipe.predict(X_test)
            
            # Instantiate the linear model and visualizer
            visualizer = ResidualsPlot(pipe)
            visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
            visualizer.score(X_test, y_test)  # Evaluate the model on the test data

            # Best hyperparameters
            errors_lasso_min = min(errors_lasso)[1]
            index_lasso_min = np.argmin(errors_lasso, axis=0)[1]
            alpha_lasso_min = alphas_lasso[index_lasso_min]

            # Display hyperparameters
            print("Hyperparameters: ")
            print("Lasso index of minimum Mean square error: " + str(index_lasso_min))
            print("Lasso alpha of minimum Mean square error: " + str(alpha_lasso_min))
            print("Lasso minimum Mean square error: " + str(errors_lasso_min))
            print("Lasso minimum Root mean square error: " + str(math.sqrt(errors_lasso_min))+ "\n\n")
            
                      
    elif grid_use == 1: 
            
        if "KNeighborsRegressor" in regression_type_string :
            # Hyperparameters to optimize
            params = {'regressor__kneighborsregressor__n_neighbors': [3],
                      'regressor__kneighborsregressor__weights': ['distance'] 
                             
                     }

        elif "SVR" in regression_type_string :   
            # Hyperparameters
            params = {'regressor__svr__C': np.logspace(-3, 3, 7),
                      'regressor__svr__epsilon': [0.1],
                      'regressor__svr__kernel': ['rbf']
                     }
            
        elif "GradientBoostingRegressor" in regression_type_string :   
            # Hyperparameters
            params = {'regressor__gradientboostingregressor__n_estimators': [100],
                     'regressor__gradientboostingregressor__learning_rate': [0.15]            
                     }
            
        elif "RandomForestRegressor" in regression_type_string :             
            # Hyperparameters
            params = {'regressor__randomforestregressor__n_estimators': [100],
                      'regressor__randomforestregressor__ccp_alpha': [0]
                     }
            
        elif "DummyRegressor" in regression_type_string :
            params = {'regressor__dummyregressor__strategy': ['mean']}
            
        elif "CatBoostRegressor" in regression_type_string :             
            # Hyperparameters
            params = {}
            
        elif "XGBRegressor" in regression_type_string :  
            # Hyperparameters
            params = {#'regressor__xgbregressor__base_score':[], 
                      #'regressor__xgbregressor__booster': ['gbtree'], 
                      #'regressor__xgbregressor__colsample_bylevel': [],
                      #'regressor__xgbregressor__colsample_bynode': [], 
                      #'regressor__xgbregressor__colsample_bytree': [], 
                      #'regressor__xgbregressor__gamma': [0],
                      #'regressor__xgbregressor__gpu_id': [], 
                      #'regressor__xgbregressor__importance_type':['gain'], 
                      #'regressor__xgbregressor__interaction_constraints': [],
                      'regressor__xgbregressor__learning_rate': [0.3], 
                      #'regressor__xgbregressor__max_delta_step': [], 
                      'regressor__xgbregressor__max_depth': [4],
                      #'regressor__xgbregressor__min_child_weight': [], 
                      #'regressor__xgbregressor__missing':=['nan'], 
                      #'regressor__xgbregressor__monotone_constraints': [],
                      #'regressor__xgbregressor__n_estimators': [100,120,150], 
                      #'regressor__xgbregressor__n_jobs': [], 
                      #'regressor__xgbregressor__num_parallel_tree': [],
                      #'regressor__xgbregressor__random_state': [1], 
                      #'regressor__xgbregressor__reg_alpha': [0], 
                      #'regressor__xgbregressor__reg_lambda': [1],
                      #'regressor__xgbregressor__scale_pos_weight': [], 
                      #'regressor__xgbregressor__subsample': [], 
                      #'regressor__xgbregressor__tree_method': [],
                      #'regressor__xgbregressor__validate_parameters': [], 
                      #'regressor__xgbregressor__verbosity': []
                     }
            
        ###   
        # Create grid
        grid = GridSearchCV(pipe, params)
        grid_gs = GridSearchCV(pipe, params)
        
        # Train grid
        grid.fit(X_train, y_train)
        #print(grid.get_params().keys())
        
        # Compute target
        pred_y = grid.predict(X_test)
        
        # Instantiate the linear model and visualizer
        visualizer = ResidualsPlot(grid)
        
        #visualizer = PredictionError(pipe)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
                
        # Display best hyperparameters
        print("Hyperparameters:")
        print("Best params:\n", str(grid.best_params_ )+ "\n")
            
        if "KNeighborsRegressor(" in regression_type_string :
                
            # Score to optimize
            score='accuracy'

            #
            for mean, std, params in zip(grid.cv_results_['mean_test_score'], # score moyen
                                         grid.cv_results_['std_test_score'],  # écart-type du score
                                         grid.cv_results_['params']           # valeur de l'hyperparamètre
                                        ):

                print("{} = {:.3f} (+/-{:.03f}) for {}".format(score, mean, std*2, params))    
            print("\n")

     
    #
    scoring_list = ['neg_mean_squared_error',
                    'neg_root_mean_squared_error', 
                    'neg_mean_absolute_error',
                    'max_error',
                    'r2'
                   ]

    #
    if grid_use == 0: 
        model = pipe
        
    elif grid_use == 1: 
        model = grid

    # 
    mse_train = mean_squared_error(y_train, model.predict(X_train), squared=True)
    rmse_train = mean_squared_error(y_train, model.predict(X_train), squared=False)
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    me_train = max_error(y_train, model.predict(X_train))
    r2_train = r2_score(y_train, model.predict(X_train))

    #
    mse_test = mean_squared_error(y_test, pred_y, squared=True)
    rmse_test = mean_squared_error(y_test, pred_y, squared=False)
    mae_test = mean_absolute_error(y_test, pred_y)
    me_test = max_error(y_test, pred_y)
    r2_test = r2_score(y_test, pred_y)

    #
    end_at = datetime.now()
    running_time = end_at - start_at

    #
    scores_cv = cross_validate(model, X, y, cv=cv, scoring=scoring_list, return_train_score=True, return_estimator=True)
    #model_2 = scores_cv['estimator']
    
    #  #
    end_at_cv = datetime.now()
    running_time_cv = end_at_cv - start_at

    # 
    mse_cv_train = np.mean(-scores_cv['train_neg_mean_squared_error'])
    rmse_cv_train = np.mean(-scores_cv['train_neg_root_mean_squared_error'])
    mae_cv_train = np.mean(-scores_cv['train_neg_mean_absolute_error'])
    me_cv_train = np.mean(scores_cv['train_max_error'])
    r2_cv_train = np.mean(scores_cv['train_r2'])

    #
    mse_cv_test = np.mean(-scores_cv['test_neg_mean_squared_error'])
    rmse_cv_test = np.mean(-scores_cv['test_neg_root_mean_squared_error'])
    mae_cv_test = np.mean(-scores_cv['test_neg_mean_absolute_error'])
    me_cv_test = np.mean(scores_cv['test_max_error'])
    r2_cv_test = np.mean(scores_cv['test_r2'])

    #
    stats_cv_train = [rmse_cv_train, mae_cv_train, me_cv_train, r2_cv_train, running_time_cv]
    stats_cv_test = [rmse_cv_test, mae_cv_test, me_cv_test, r2_cv_test, running_time_cv]

    # Display non cross validated metrics
    # train set
    print(regression_type_string + ' Performance - 1 fold - train set: ')
    print('RMSE: {}' .format(rmse_train))
    print('MAE: {}' .format(mae_train))
    print('ME: {}' .format(me_train))
    print('R2: {}' .format(r2_train))
    print("")
    # test set
    print(regression_type_string + ' Performance - 1 fold - test set: ')
    print('RMSE: {}' .format(rmse_test))
    print('MAE: {}' .format(mae_test))
    print('ME: {}' .format(me_test))
    print('R2: {}' .format(r2_test))
    print("")
    print('Running time: {}' .format(running_time))
    print("\n")
    
    # Display cross validated metrics
    # train set
    print(regression_type_string + ' Performance - CV ' + str(cv) + ' fold - train set: ')
    print('RMSE: {}' .format(rmse_cv_train))
    print('MAE: {}' .format(mae_cv_train))
    print('ME: {}' .format(me_cv_train))
    print('R2: {}' .format(r2_cv_train))
    print("")
    # test set
    print(regression_type_string + ' Performance - CV ' + str(cv) + ' fold - test set: ')
    print('RMSE: {}' .format(rmse_cv_test))
    print('MAE: {}' .format(mae_cv_test))
    print('ME: {}' .format(me_cv_test))
    print('R2: {}' .format(r2_cv_test))
    print("")
    print('Running time cv: {}' .format(running_time_cv))
    print("\n")
    
    # Yellowbrick
    visualizer.poof()  
    
    # Lists to return
    stats = [rmse_test, mae_test, me_test, r2_test, running_time]
    stats_cv_train = [rmse_cv_train, mae_cv_train, me_cv_train, r2_cv_train, running_time_cv]
    stats_cv_test = [rmse_cv_test, mae_cv_test, me_cv_test, r2_cv_test, running_time_cv]
    
    #
    df_test = X_test.copy()
    df_test['y_test'] = y_test

    # Return    
    if "LinearRegression" in regression_type_string :
        return stats, stats_cv_train, stats_cv_test, mse_test, model

    elif "Ridge(" in regression_type_string :
        return stats, stats_cv_train, stats_cv_test, alpha_ridge_min, alphas_ridge, errors_ridge, coefs_ridge, model

    elif "Lasso" in regression_type_string :
        return stats, stats_cv_train, stats_cv_test, alpha_lasso_min, alphas_lasso, errors_lasso, coefs_lasso, model

    else :
        return stats, stats_cv_train, stats_cv_test, X_test, y_test, pred_y, model, df_test
    
    
def prepare_input_data(dataframe, target, **kwargs):
    """ General function for model creation and evaluation

    Args:
        dataframe (pd.Dataframe): input
        target (str): name of target variable

    Returns:
        df_general_training, 
        X, 
        y, 
        numerical_list, 
        categorical_list
    """  
    #
    numerical_list = ['PropertyGFATotal',        
                      'PropertyGFABuilding(s)',
                      'LargestPropertyUseTypeGFA',
                      'SecondLargestPropertyUseTypeGFA',
                      'ThirdLargestPropertyUseTypeGFA',
                      'PropertyGFAParking',	
                      'NumberofBuildings',
                      'NumberofFloors',
                      'Age',
                      'ENERGYSTARScore',
                      #'GHGEmissionsIntensity(kgCO2e/ft2)',	
                      #'SourceEUI(kBtu/sf)',	
                      #'SteamUse(kBtu)',	
                      #'NaturalGas(kBtu)',	
                      #'Electricity(kBtu)',
                      #'OtherFuelUse(kBtu)'
                      ]
    
    #
    categorical_list = ['BuildingType', 
                        'Neighborhood',
                        #'ZipCode',
                        'LargestPropertyUseType', 		
                        'PrimaryPropertyType', 
                        'SecondLargestPropertyUseType', 
                        'ThirdLargestPropertyUseType'
                        ]
    
    #
    target_variable = target
    list_variables_model = numerical_list + categorical_list + [target_variable]
    numerical_and_target_list = numerical_list + [target_variable]

    # Filter with list
    df_general_training = dataframe[list_variables_model].copy()
    
    #
    df_general_training = trim_dataframe(df_general_training, numerical_list, categorical_list, target_variable)
    
    #
    if kwargs:
        drop_variable = kwargs['drop_variable']
        df_general_training = df_general_training.drop(columns=[drop_variable])
        numerical_list.remove(drop_variable)

    # Separate input and target values
    X = df_general_training.drop(columns=[target_variable])
    y = df_general_training[target_variable]
    
    #
    return df_general_training, X, y, numerical_list, categorical_list


def trim_dataframe(dataframe, numerical_list, categorical_list, target_variable):
    """ Make dataframe without NaN values

    Args:
        dataframe (pd.Dataframe): input
        numerical_list (str):, 
        categorical_list (str):, 
        target_variable (str): name of target variable,

    Returns:
        X df_general_training (pd.Dataframe): trimmed dataframe, 
  
    """  
    #
    dataframe = dataframe.dropna(subset=['ENERGYSTARScore',target_variable], axis=0)
    dataframe = dataframe.dropna(subset=[target_variable], axis=0)
    dataframe[numerical_list] = dataframe[numerical_list].fillna(0)
    dataframe[categorical_list] = dataframe[categorical_list].fillna("NaN")
    
    return dataframe

    
def schema_function(numerical_variables_list, categorical_variables_list, regression_type):

    # 
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_median", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numeric_preprocessor, numerical_variables_list),
            ("categorical", categorical_preprocessor, categorical_variables_list)
        ]
    )

    # Create pipeline
    pipe = make_pipeline(preprocessor, regression_type)

    return pipe



'''def error_alpha_function(alpha_ridge_min, alphas_ridge, errors_ridge):
    """
    """
    plt.figure(figsize=(9,6))
    plt.vlines(alpha_ridge_min, 0, 50000000000, colors='r', linestyles='dashed', label='', linewidth=1)

    ax = plt.gca()

    ax.plot(alphas_ridge, errors_ridge)
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.axis('Tight')
    plt.show()'''


'''def regularization_path_function(alpha_ridge_min, alphas_ridge, coefs_ridge):
    """
    """
    plt.figure(figsize=(9,6))
    plt.vlines(alpha_ridge_min, 400000, -50000, colors='r', linestyles='dashed', label='', linewidth=1)

    ax = plt.gca()

    ax.plot(alphas_ridge, coefs_ridge)
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()'''

'''def prediction_accuracy_overview_function(X_test, y_test, ridge_cv_pred_y):
    """
    """
    # Prediction accuracy overview for the testing set
    number_of_individuals = range(len(X_test))

    plt.figure(figsize=(15,6))
    plt.scatter(number_of_individuals, y_test, s=5, color="blue", label="original")
    plt.plot(number_of_individuals, ridge_cv_pred_y, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()'''


'''def hyperopt_knn_eval_model_function(X_train,y_train,X_test,y_test):

    # Define search
    model = HyperoptEstimator(regressor=any_regressor('reg'),
                              preprocessing=any_preprocessing('pre'),
                              loss_fn=mean_absolute_error,
                              algo=tpe.suggest,
                              max_evals=50,
                              trial_timeout=30
                             )

    # Perform the search
    model.fit(X_train, y_train)

    # Summarize performance
    mae = model.score(X_test, y_test)
    print("MAE: %.3f" % mae)

    # Summarize the best model
    print(model.best_model())'''



def lazy_predict_on_premise_function(metric, order, **kwargs):
    """ 
    Plots linear regression
    Args:
        metric (string): sklearn metrics
        order (string): ascending or descending

    Returns:
        dataframe (pd.Dataframe): data output
    """                          
    # data
    d = {}
    
    for kwarg in kwargs:
        newline = {kwarg: kwargs[kwarg]}
        d.update(newline)

    index = ['RMSE',
             'MAE',
             'MaxError',
             'R2', 
             'Running time']

    # dataframe
    df = pd.DataFrame(data=d, index=index).transpose()

    order = order.lower()
    
    #
    if order == 'ascending':
        order = True 
    else: 
        order = False 
    
    #
    return df.sort_values(by=metric, axis=0, ascending=order, inplace=False, kind='quicksort')


   
def linear_regression_function(dataframe, variable_x, variable_y, arrange, log) :
    """ 
    Plots linear regression
    Args:
        dataframe (pd.Dataframe): data source
        variable_x (str): x axis variale
        variable_y (str): y axis variale
        arrange (int):

    Returns:
        Plotted linear regression

    """
    start_at = datetime.now()


    # Fill missing values with 0
    x_sample = dataframe[variable_x]
    y_sample = dataframe[variable_y]
    
    if log == 0 :
        # Define variables for regression
        Y = dataframe[variable_y]
        X = dataframe[[variable_x]]
        
    elif log == 1 :
        #
        Y = np.log1p(dataframe[variable_y])
        X = np.log1p(dataframe[[variable_x]])

    # We will modify x_sample so we create a copy of it
    X = X.copy()
    X['intercept'] = 1.

    # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
    result = sm.OLS(Y, X).fit()
    print(result.summary())

    a, b = result.params[variable_x], result.params['intercept']
    print("a: " + str(a), "b: " + str(b))

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(dataframe[variable_x],
             dataframe[variable_y],
             "o",
             alpha = 0.5
             )

    plt.plot(np.arange(arrange),[a*x+b for x in np.arange(arrange)], linewidth=2.5)
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    plt.show()
    
    end_at = datetime.now()
    running_time = end_at - start_at
    print(running_time)
    
    
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    X_test, 
    y_test, 
    pred_y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(20, 20))

    axes[0][0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0][0].set_xlabel("Training examples")
    axes[0][0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0][0].grid()
    axes[0][0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0][0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0][0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0][0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0][0].legend(loc="best")
    #axes[0][0].x_ticks(fontsize=16)
    #axes[0][0].y_ticks(fontsize=16)
    #axes[0][0].title(r"", fontsize=16)

    # Plot n_samples vs fit_times
    axes[0][1].grid()
    axes[0][1].plot(train_sizes, fit_times_mean, "o-")
    axes[0][1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[0][1].set_xlabel("Training examples")
    axes[0][1].set_ylabel("fit_times")
    axes[0][1].set_title("Scalability of the model")
    #axes[0][1].x_ticks(fontsize=16)
    #axes[0][1].y_ticks(fontsize=16)
    #axes[0][1].title(r"", fontsize=16)

    # Plot fit_time vs score
    axes[1][0].grid()
    axes[1][0].plot(fit_times_mean, test_scores_mean, "o-")
    axes[1][0].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[1][0].set_xlabel("fit_times")
    axes[1][0].set_ylabel("Score")
    axes[1][0].set_title("Performance of the model")
    #axes[1][0].x_ticks(fontsize=16)
    #axes[1][0].y_ticks(fontsize=16)
    #axes[1][0].title(r"", fontsize=16)
    
    
    y_test_max = y_test.max()
    pred_y_max = pred_y.max()
    
    #####
    axes[1][1].grid()
    axes[1][1].scatter(pred_y, y_test, color="b")
    axes[1][1].plot([0, pred_y_max], [0, y_test_max], "--k")
       
    axes[1][1].set_xlabel("Predicted Target")
    axes[1][1].set_ylabel("True Target")
    axes[1][1].set_title("Predicted Values")
    #axes[1][1].x_ticks(fontsize=16)
    #axes[1][1].y_ticks(fontsize=16)
    #axes[1][1].title(r"", fontsize=16)


    return plt


def plot_learning_curve_compare(
    estimator_1,
    estimator_2,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1, 10),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator_1 : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    estimator_2 : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 10))

    axes.set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes.set_xlabel("Training examples", fontsize=20)
    axes.set_ylabel("Score", fontsize=20)
  


    train_sizes_1, train_scores_1, test_scores_1, fit_times_1, _ = learning_curve(
        estimator_1,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )

    train_sizes_2, train_scores_2, test_scores_2, fit_times_2, _ = learning_curve(
        estimator_2,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    
    
    test_scores_mean_1 = np.mean(test_scores_1, axis=1)
    test_scores_std_1 = np.std(test_scores_1, axis=1)
    
    test_scores_mean_2 = np.mean(test_scores_2, axis=1)
    test_scores_std_2 = np.std(test_scores_2, axis=1)
  

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes_1,
        test_scores_mean_1 - test_scores_std_1,
        test_scores_mean_1 + test_scores_std_1,
        alpha=0.1,
        color="g",
    )
    axes.fill_between(
        train_sizes_2,
        test_scores_mean_2 - test_scores_std_2,
        test_scores_mean_2 + test_scores_std_2,
        alpha=0.1,
        color="b",
    )
    axes.plot(
        train_sizes_1, test_scores_mean_1, "o-", color="g", label="Avec ENERGYSTARScore"
    )
    axes.plot(
        train_sizes_2, test_scores_mean_2, "o-", color="b", label="Sans ENERGYSTARScore"
    )
    axes.legend(loc="best")


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(r"Learning Curves XGBoostRegressor model - Test Set", fontsize=16)
    return plt


    
'''def performance_final_model_regressor_by_building_type(dataframe, 
                                                       regression, 
                                                       scaler, 
                                                       cv, 
                                                       primary_property_type_list, 
                                                       numerical_list, 
                                                       categorical_list, 
                                                       target_variable,
                                                       min_sample_size
                                                      ):

    
    for primary_property_type in primary_property_type_list :

        print("Sample: " + str(primary_property_type))

        df_general_training_primary_property_type = dataframe[dataframe["PrimaryPropertyType"]==primary_property_type]
        print("Size: " + str(len(df_general_training_primary_property_type)) + "\n") 

        if (len(df_general_training_primary_property_type)>=min_sample_size) :

            # prepare_input_data() is defined in utils_elec_code.py
            df_general_training_primary_property_type, 
            X, 
            y, 
            numerical_list, 
            categorical_list = prepare_input_data(df_general, 'SiteEnergyUse(kBtu)')

            # regression_eval_model_function is defined in utils.py
            results = regression_eval_model_function(X, 
                                                     y, 
                                                     numerical_list, 
                                                     categorical_list, 
                                                     target_variable, 
                                                     regression, 
                                                     scaler, 
                                                     0.3,
                                                     0, 
                                                     cv
                                                    )     
            
        else : 
            print("Sample too small")
    
        print("\n\n")'''
        
        
    
def performance_final_model_by_building_type(dataframe_test, 
                                             regression, 
                                             target_variable, 
                                             primary_property_type_list,
                                             min_sample_size,
                                             metric,
                                             order
                                            ):
    """ 
    Plots linear regression
    Args:
        dataframe_test (pd.Dataframe): data source
        regression (str): sklearn function
        target_variable (pd.Series): target
        primary_property_type_list (list(string))):
        min_sample_size (int): number of minimum buildings for a given type to proceed to prediction
        metric (string):
        order (string): ascending or descending

    Returns:
        Prints list of the model parameters and stats 

    """
    d = {}
    
    for primary_property_type in primary_property_type_list :
        print("Sample: " + str(primary_property_type))

        df_primary_property_type = dataframe_test[dataframe_test["PrimaryPropertyType"]==primary_property_type]
        print("Size: " + str(len(df_primary_property_type)) + "\n") 

        if (len(df_primary_property_type)>=min_sample_size) :
            
            #
            X_test = df_primary_property_type.drop(columns=[target_variable])
            y_test = df_primary_property_type[target_variable]
            
            #
            pred_y = regression.predict(X_test)
            
            #
            mse_test = mean_squared_error(y_test, pred_y, squared=True)
            rmse_test = mean_squared_error(y_test, pred_y, squared=False)
            mae_test = mean_absolute_error(y_test, pred_y)
            me_test = max_error(y_test, pred_y)
            r2_test = r2_score(y_test, pred_y)

            # test set
            print('Performance - 1 fold - test set: ')
            print('RMSE: {}' .format(rmse_test))
            print('MAE: {}' .format(mae_test))
            print('ME: {}' .format(me_test))
            print('R2: {}' .format(r2_test))
            
            # 
            newline = {'RMSE':rmse_test,'Max_Error':mae_test,'ME':me_test,'R2':r2_test}
 
        else : 
            print("Sample too small")   
            # 
            newline = {}
            
        print("\n\n") 
        

