#######################
# Yamac TAN - Data Science Bootcamp - Week 10 - Project 1
#######################

# %%

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 50)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

warnings.simplefilter(action='ignore', category=Warning)

# %%

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# %%
# Reading and quick look to the data:

# Here, using "parse_dates" param in pd.read_csv method allows us to get dtype for column 'date' as datetime64[ns].
# Otherwise, the related dtype will return as object. We could do this operation by manually changing the dtype of
# 'date', but one of the main principles is increasing the code readability and writing a clean code.

test_df = pd.read_csv("Odevler/WEEK_10_ TIME SERIES/test.csv", parse_dates=['date'])
train_df = pd.read_csv("Odevler/WEEK_10_ TIME SERIES/train.csv", parse_dates=['date'])

train_df.info()
test_df.info()

train_df.isnull().sum()
test_df.isnull().sum()

# It would be a better approach to combine the two datasets we have in order to be able to perform exploratory data
# analysis and feature engineering applications at once.
df = pd.concat([train_df, test_df])

df.describe().T

check_df(df)

# %%

train_first_date = train_df['date'].min()  # 2013-01-01
train_last_date = train_df['date'].max()  # 2017-12-31
test_first_date = test_df['date'].min()  # 2018-01-01
test_last_date = test_df['date'].max()  # 2018-03-31

df[['store']].nunique(), df[['item']].nunique()  # Verified : We have 50 unique item and 10 unique store in dataset
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

# %%
# Feature Engineering

check_df(df)
df.head()
df.info()

df['month'] = df.date.dt.month
df['year'] = df.date.dt.year
df.loc[(df["month"] < 3) | (df["month"] == 12), "season"] = 1  #Winter
df.loc[(df["month"] >= 3) & (df["month"] < 6), "season"] = 2  #Spring
df.loc[(df["month"] >= 6) & (df["month"] < 9), "season"] = 3  #Summer
df.loc[(df["month"] >= 9) & (df["month"] < 12), "season"] = 4  #Fall
df['season'] = df['season'].astype('int64')
df['day_of_month'] = df.date.dt.day
df['day_of_year'] = df.date.dt.dayofyear
df['week_of_year'] = df.date.dt.weekofyear
df['day_of_week'] = df.date.dt.dayofweek
df["is_wknd"] = df.date.dt.weekday // 4

df.groupby(["season"]).agg({"sales": ["sum", "mean", "median", "std"]})
# By considering the statistical values, we can say that season feature will have an impact on our model.

# %%
# Random Noise Function

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# In the continuation of the work, a random noise should be added to the features in order to avoid any overfitting in
# the features to be produced for the dataset.

# %%
# Log_Shifted Features

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

def lag_features(dataframe, lags):  #For the given lag values,create shifted features with adding a random noise.
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# Declare the lag values in terms of days. Remind that we will create a 3-month demand forecasting model
lag_values = [91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168]

df = lag_features(df, lag_values)

check_df(df)

# %%
# Rolling Mean Features

# We were creating features by bringing the previous values in log shifted features.
# Now we create it by averaging the previous values.

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

roll_mean_values = [365, 547, 730]

df = roll_mean_features(df,roll_mean_values)

# %%
# Exponentially Weighted Mean Features

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lag_values = [91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168]
df = ewm_features(df, alphas, lag_values)

# %%
# One Hot Encoding

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month', 'season'])

# %%
# log(1+sales)

# In tree problems, if the dependent variable is numerical, that is, a regression or time series problem,
# standardization can be used with the assumption of shortening the iteration time.

df['sales'] = np.log1p(df["sales"].values)

# %%
# Custom Cost - SMAPE Function

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# We created a different variation of our function, in order to use it LightGBM model that we will create.

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# %%
# Creating data subsets:

train = df.loc[(df["date"] < "2017-06-01"), :]

val = df.loc[(df["date"] >= "2017-06-01") & (df["date"] < "2017-09-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# %%
# LightGBM Model:

# These parameters are tested before the training.
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,  # If nothign changes in 200 iteration, early stop the training.
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)  # Give the related info in every 100 iteration.

# Did not meet early stopping. Best iteration is:
# [1000]	training's l2: 0.0283088	training's SMAPE: 13.1618	valid_1's l2: 0.0205204	valid_1's SMAPE: 11.1397

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

# %%
# Feature importances for final model

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=30, plot=True)

important_features = plot_lgb_importances(model, num=165)

importance_zero = important_features[important_features["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)

# %%
# Final model

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv("submission_demand_week10.csv", index=False)