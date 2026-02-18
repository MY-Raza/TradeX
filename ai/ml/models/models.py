import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

# ===============================
# CLASSIFIERS
# ===============================

# Linear
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron

# Distance
from sklearn.neighbors import KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# SVM
from sklearn.svm import SVC, LinearSVC

# Trees
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# Ensembles
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

# Neural Network
from sklearn.neural_network import MLPClassifier

# Discriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


# ===============================
# REGRESSORS
# ===============================

# Linear
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor
)

# Distance
from sklearn.neighbors import KNeighborsRegressor

# SVM
from sklearn.svm import SVR

# Trees
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

# Ensembles
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

# Neural Network
from sklearn.neural_network import MLPRegressor

# Specialized
from sklearn.gaussian_process import GaussianProcessRegressor


# ===============================
# External Boosting Libraries
# ===============================
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("models_factory")

# ============================================================
# GENERIC TRAINING FUNCTIONS
# ============================================================

def train_classifier(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n================ MODEL PERFORMANCE ================")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
    return model


def train_regressor(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n================ MODEL PERFORMANCE ================")
    print("MSE:", mean_squared_error(y_test, preds))
    print("R2 Score:", r2_score(y_test, preds))
    return model


# ============================================================
# CLASSIFIER FACTORY
# ============================================================

def get_classifier(name: str):
    classifiers = {

        # Linear
        "logistic": LogisticRegression(max_iter=1000),
        "ridge": RidgeClassifier(),
        "perceptron": Perceptron(),

        # Distance
        "knn": KNeighborsClassifier(),

        # Naive Bayes
        "gaussian_nb": GaussianNB(),
        "multinomial_nb": MultinomialNB(),
        "bernoulli_nb": BernoulliNB(),

        # SVM
        "svc": SVC(),
        "linear_svc": LinearSVC(),

        # Trees
        "decision_tree": DecisionTreeClassifier(),
        "extra_tree": ExtraTreeClassifier(),

        # Ensembles
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1),
        "adaboost": AdaBoostClassifier(),
        "gradient_boost": GradientBoostingClassifier(),

        # Neural
        "mlp": MLPClassifier(max_iter=500),

        # Discriminant
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),

        # Boosting
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "catboost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_state=42
        ),
    }
    return classifiers.get(name.lower())


# ============================================================
# REGRESSOR FACTORY
# ============================================================

def get_regressor(name: str):
    regressors = {

        # Linear
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "elastic": ElasticNet(),
        "bayesian_ridge": BayesianRidge(),

        # Distance
        "knn": KNeighborsRegressor(),

        # SVM
        "svr": SVR(),

        # Trees
        "decision_tree": DecisionTreeRegressor(),
        "extra_tree": ExtraTreeRegressor(),

        # Ensembles
        "random_forest": RandomForestRegressor(n_estimators=300, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, n_jobs=-1),
        "adaboost": AdaBoostRegressor(),
        "gradient_boost": GradientBoostingRegressor(),

        # Neural
        "mlp": MLPRegressor(max_iter=500),

        # Specialized
        "gpr": GaussianProcessRegressor(),
        "poisson": PoissonRegressor(),
        "gamma": GammaRegressor(),
        "tweedie": TweedieRegressor(),

        # Boosting
        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "lightgbm": LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "catboost": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_state=42
        ),
    }
    return regressors.get(name.lower())
