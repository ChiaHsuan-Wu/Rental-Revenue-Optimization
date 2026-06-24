import logging
import pandas as pd
import numpy as np
import json
import zipfile
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform


def baseline():
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')
    
    label = 'revenue'
    numeric_features = ['lat', 'lon', 'num_reviews', 'guests','rooms','bathrooms','beds','min_nights','rating']
    categorical_features = ['host','name','room_type','listing_type','facilities','cancellation']
    
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='median')),
            ]), numeric_features),
            ("categorical", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
        ],
        remainder='drop'
    )
    
    train, valid = train_test_split(train, test_size=1/3, random_state=123)

    dummy = make_pipeline(preprocess, DummyRegressor())
    base = make_pipeline(preprocess, Ridge(alpha=1, random_state=123))
    rf = make_pipeline(preprocess, RandomForestRegressor(random_state=123))
    gb = make_pipeline(preprocess, GradientBoostingRegressor(random_state=123))
    xgb = make_pipeline(preprocess, XGBRegressor(random_state=123))
    models = [("mean",dummy), ("base",base), ("rf",rf), ("gb",gb), ("xgb",xgb)]

    print("=== Basic models evaluation ===")
    for model_name, model in models:
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        for split_name, split in [("train     ", train),
                                  ("valid     ", valid)
                                  ]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} {mae:.3f}")
    
    print("\n=== Starting RandomizedSearchCV ===")
    full_train_X = train.drop([label], axis=1)
    full_train_y = np.log1p(train[label].values)
    
    # define parameters
    param_distributions = {
        'RandomForest': {
            'model': make_pipeline(preprocess, RandomForestRegressor(random_state=123)),
            'params': {
                'randomforestregressor__n_estimators': randint(50, 300),
                'randomforestregressor__max_depth': [5, 10, 15, 20, None],
                'randomforestregressor__min_samples_split': randint(2, 15),
                'randomforestregressor__min_samples_leaf': randint(1, 8),
                'randomforestregressor__max_features': ['sqrt', 'log2', None]
            }
        },
        'GradientBoosting': {
            'model': make_pipeline(preprocess, GradientBoostingRegressor(random_state=123)),
            'params': {
                'gradientboostingregressor__n_estimators': randint(50, 400),
                'gradientboostingregressor__learning_rate': uniform(0.01, 0.19),
                'gradientboostingregressor__max_depth': randint(3, 8),
                'gradientboostingregressor__subsample': uniform(0.7, 0.3),
                'gradientboostingregressor__min_samples_split': randint(2, 15)
            }
        },
        'XGBoost': {
            'model': make_pipeline(preprocess, XGBRegressor(random_state=123, verbosity=0)),
            'params': {
                'xgbregressor__n_estimators': randint(50, 400),
                'xgbregressor__learning_rate': uniform(0.01, 0.19),
                'xgbregressor__max_depth': randint(3, 8),
                'xgbregressor__subsample': uniform(0.7, 0.3),
                'xgbregressor__colsample_bytree': uniform(0.7, 0.3),
                'xgbregressor__reg_alpha': uniform(0, 0.5), 
                'xgbregressor__reg_lambda': uniform(0, 0.5)
            }
        },
    }
    
    # save each model's best result
    best_models = {}
    
    for model_name, config in param_distributions.items():
        print(f"\n--- Random search {model_name} ---")
        
        random_search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=30, 
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1, 
            random_state=123,
            verbose=1,
        )
        
        random_search.fit(full_train_X, full_train_y)
        
        best_models[model_name] = {
            'model': random_search.best_estimator_,
            'params': random_search.best_params_,
            'score': -random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        print(f"{model_name} best parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"{model_name} best CV MAE: {-random_search.best_score_:.4f}")
    
    # evaluate models on valid set
    print("\n=== The performance on validation set after tuning ===")
    validation_results = {}
    
    for model_name, model_info in best_models.items():
        best_model = model_info['model']
        pred_valid = np.expm1(best_model.predict(valid.drop([label], axis=1)))
        mae_valid = mean_absolute_error(valid[label], pred_valid)
        
        validation_results[model_name] = mae_valid
        print(f"{model_name} validation set MAE: {mae_valid:.4f}")
    

    best_model_name = min(validation_results, key=validation_results.get)
    best_model = best_models[best_model_name]['model']
    
    print(f"\n=== the Best model: {best_model_name} ===")
    print(f"Validation set MAE: {validation_results[best_model_name]:.4f}")
    print("the Best hyperparameters:")
    for param, value in best_models[best_model_name]['params'].items():
        print(f"  {param}: {value}")


    # predict real revenue MAE
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer

    full_train = pd.concat([train, valid])
    X_full = full_train.drop(columns=[label])
    y_full = np.log1p(full_train[label])

    def rev_mae(y_true_log, y_pred_log):
        return mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    rev_scorer = make_scorer(rev_mae, greater_is_better=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = cross_val_score(
        best_model,
        X_full, y_full,
        scoring=rev_scorer,
        cv=kf,
        n_jobs=-1
    )
    logging.info(f"5-fold CV real revenue MAE = {-scores.mean():.3f}")
    full_train = pd.concat([train, valid])
    best_model.fit(full_train.drop([label], axis=1), np.log1p(full_train[label].values))
    
    # predict test set
    pred_test_final = np.expm1(best_model.predict(test))
    test[label] = pred_test_final
    predicted_final = test[['revenue']].to_dict(orient='records')
    
    filename = f"best_{best_model_name.lower()}_randomsearch.zip"
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("predicted.json", json.dumps(predicted_final, indent=2))
    
    print(f"\n the final prediction is saved in: {filename}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    baseline()