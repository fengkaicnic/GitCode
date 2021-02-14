import catboost as ctb
from catboost import CatBoostClassifier
import createmodel


model = CatBoostClassifier(iterations=100, depth=5,cat_features=categorical_features_indices,learning_rate=0.5, loss_function='Logloss',
                            logging_level='Verbose')

model.fit(X_train,y_train,eval_set=(X_validation, y_validation),plot=True)



