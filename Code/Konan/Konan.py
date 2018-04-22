# -------- #
# PACKAGES #
# -------- #

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import xgboost as xgb
from pandas import concat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, OneHotEncoder


class DataFrameSelector(BaseEstimator, TransformerMixin):
	
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	
	def fit(self, X, y = None):
		return self
	
	def transform(self, X, attribute_names = None):
		if attribute_names:
			return X[attribute_names]
		else:
			return X[self.attribute_names]


def series_to_supervised(data, index, n_in = 1, n_out = 1, dropnan = True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data, index = index)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis = 1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace = True)
	return agg


# -------------#
# LOADING DATA #
# -------------#

url = "https://raw.githubusercontent.com/TheKonanKouassi/Projet-Fil-Rouge/master/Data/"

meteo_train = pd.read_csv(url + "meteo_train.csv", sep = ";")
conso_train = pd.read_csv(url + "conso_train.csv", sep = ";")
meteo_previ = pd.read_csv(url + "meteo_prev.csv", sep = ";")

# ------------------ #
# DATA PREPROCESSING #
# ------------------ #

columns = {"Date UTC"       : "date",
           "T¬∞ (C)"        : "temperature", "P (hPa)": "pression",
           "HR (%)"         : "humidite_relative", "P.ros√©e (¬∞C)": "point_rose",
           "Visi (km)"      : "visibilite", "Vt. moy. (km/h)": "vent_moyen", "Vt. raf. (km/h)": "vent_rafale",
           "Vt. dir (¬∞)"   : "vent_direction", "RR 3h (mm)": "pluie_3h", "Neige (cm)": "neige",
           "Nebul. (octats)": "nebulosite"}

# Renommer les noms des colonnes
meteo_train.rename(columns = columns, inplace = True)
meteo_previ.rename(columns = columns, inplace = True)

# Convertir les dates au bon format
conso_train["date"] = pd.to_datetime(conso_train["date"])
meteo_train["date"] = pd.to_datetime(meteo_train["date"], format = "%d/%m/%y %Hh%M")
meteo_previ["date"] = pd.to_datetime(meteo_previ["date"], format = "%d/%m/%y %Hh%M")

# Definir les dates comme index
conso_train.set_index("date", inplace = True)
meteo_train.set_index("date", inplace = True)
meteo_previ.set_index("date", inplace = True)

# Spécifier les bonnes time zone
conso_train.index = conso_train.index.tz_localize(pytz.utc).tz_convert(pytz.timezone('Europe/Paris'))
meteo_train.index = meteo_train.index.tz_localize(pytz.timezone('Europe/Paris'))
meteo_previ.index = meteo_previ.index.tz_localize(pytz.timezone('Europe/Paris'))

# Supprimer les doublons
conso_train = conso_train.reset_index().drop_duplicates(subset = "date").set_index("date")
meteo_train = meteo_train.reset_index().drop_duplicates(subset = "date").set_index("date")
meteo_previ = meteo_previ.reset_index().drop_duplicates(subset = "date").set_index("date")

# Splitter les données par année
conso_train_2015 = conso_train["2015"]
conso_train_2016 = conso_train["2016"]
meteo_train_2015 = meteo_train["2015"]
meteo_train_2016 = meteo_train["2016"]

# Reajuster des dates pour les données de consommation
conso_train_2015.index = conso_train_2015.index + pd.Timedelta(seconds = 1)

# Reajuster des dates pour les données de météo
meteo_train_2015 = meteo_train_2015.append(meteo_train_2016.iloc[0, :])

# Interpoler les données météo à la maille horaire
meteo_train_2015_resampled = meteo_train_2015.resample("H").interpolate().iloc[1:-1]
meteo_train_2016_resampled = meteo_train_2016.resample("H").interpolate()
conso_train_2015_resampled = conso_train_2015.resample("H").interpolate()
conso_train_2016_resampled = conso_train_2016.resample("H").interpolate()

# Reajuster des dates pour les données de météo de 2016
conso_train_2016_resempled = conso_train_2016.iloc[:-3]

# Supprimer la tranche horaire de données manquantes
meteo_train_2016_resampled.drop(meteo_train_2016_resampled["2016-02-29"].index, inplace = True)
meteo_train_2016_resampled.drop(meteo_train_2016_resampled["2016-02-21":"2016-02-28"].index, inplace = True)
conso_train_2016_resempled.drop(conso_train_2016_resempled["2016-02-21":"2016-02-28"].index, inplace = True)

# Checher si les tailles correspondent
print("2015 lenght match ?", len(meteo_train_2015_resampled) == len(conso_train_2015_resampled))
print("2016 lenght match ?", len(meteo_train_2016_resampled) == len(conso_train_2016_resempled))

# Fusionner les données qui avaient été Splitté
meteo_train_resampled = pd.concat([meteo_train_2015_resampled, meteo_train_2016_resampled])
conso_train_resampled = pd.concat([conso_train_2015_resampled, conso_train_2016_resempled])

# Supprimer la donnée de neige qui étaient null
# meteo_train_resampled.drop("neige", axis = 1, inplace = True)

X = meteo_train_resampled.values
y = conso_train_resampled.values

# Fusionner les données pour appliquer la fonction de lag
data_train = np.c_[X, y]

# Aplliquer la finction de lag avec 73, qui correspond à 3 jours
data_train_reframed = series_to_supervised(data_train, meteo_train_resampled.index, 73, 1)

# Récuperer la variable à expliquer
X = data_train_reframed.iloc[:, :-1]
y = data_train_reframed.iloc[:, -1]

# Extraire le jour et l'heure qui serviront dans la prédiction
X["hour"] = X.index.hour
X["day"] = X.index.weekday

# Distinguer les variables catégorielles et les variables numeriques
cat_features = ["hour", "day"]
num_features = X.drop(["hour", "day"], axis = 1).columns.values

# Créer une pipeline pour les variables numeriques
num_pipeline = Pipeline([
	("selector", DataFrameSelector(num_features)),
	("imputer", Imputer(strategy = "median"))
])

# Créer une pipeline pour les variables catégorielles
cat_pipeline = Pipeline([
	("selector", DataFrameSelector(cat_features)),
	("imputer", OneHotEncoder(sparse = False))
])

# Créer une pipeline finale pour fusionner les données
full_pipeline = FeatureUnion([
	("cat_pipeline", cat_pipeline),
	("num_pipeline", num_pipeline)
])

# Transformer les données météo
X_prep = full_pipeline.fit_transform(X)

# Choisir 329 jours d'apprentissage, 16 jours de validation, et 8 jours de test
n_train = 329 * 24
n_valid = 345 * 24

# Splitter en apprentissage, validation et test
X_train = X_prep[:n_train, :]
X_valid = X_prep[n_train:n_valid, :]
X_test_ = X_prep[n_valid:, :]

y_train = y[:n_train]
y_valid = y[n_train:n_valid:]
y_test_ = y[n_valid:]

# ------------- #
# DATA MODELING #
# ------------- #

xgb_reg = xgb.XGBRegressor(
		max_depth = 5,
		n_estimators = 100,
		learning_rate = 0.07,
		colsample_bytree = 1,
		min_child_weight = 5
)

xgb_reg.fit(X_train, y_train)

y_valid_pred = xgb_reg.predict(X_valid)

print(f"MAE sur validation : {mean_absolute_error(y_valid_pred, y_valid)}")

plt.scatter(y_valid_pred, y_valid)
plt.show()

y_test_pred = xgb_reg.predict(X_test_)

print(f"MAE sur test : {mean_absolute_error(y_test_pred, y_test_)}")

plt.scatter(y_test_pred, y_test_)
plt.show()

# ---------------- #
# DATA FORECASTING #
# ---------------- #

# Interpoler les données de prévision
meteo_previ = meteo_previ.append(meteo_train.iloc[-1, :]).sort_index()

meteo_previ_resampled = meteo_previ.resample("H").interpolate().iloc[:-1, :]

# Creer les donnéses de consommation à prédire
conso_previ = pd.DataFrame(np.zeros((len(meteo_previ_resampled), 1)), columns = ["puissance"],
                           index = meteo_previ_resampled.index)

conso_previ["puissance"].iloc[:4] = conso_train["puissance"].iloc[-4:].values

# Concatener toutes les données
conso_total = pd.concat([conso_train_resampled, conso_previ.iloc[1:, :]])
meteo_total = pd.concat([meteo_train_resampled, meteo_previ_resampled.iloc[1:, :]])

# Supprimer les données inutiles
meteo_total = meteo_total.drop(["neige"], axis = 1)

# Mettre sous la forme avec des lag
X_tatal = meteo_total.values
y_total = conso_total.values

data_total = np.c_[X_tatal, y_total]
data_total_reframed = series_to_supervised(data_total, meteo_total.index, 73, 1)

# --------------------------
# Prédiction avec validation

# Splitter entre apprentissage et validation
X_train = data_total_reframed[:"2016-09-01"].iloc[:, :-1]
y_train = data_total_reframed[:"2016-09-01"].iloc[:, -1]

X_valid = data_total_reframed["2016-09-02":"2016-09-12"].iloc[:, :-1]
y_valid = data_total_reframed["2016-09-02":"2016-09-12"].iloc[:, -1]

# Rajouter les varibles heure et jour
X_train["hour"] = X_train.index.hour
X_train["day"] = X_train.index.weekday

X_valid["hour"] = X_valid.index.hour
X_valid["day"] = X_valid.index.weekday

# Extraire les version laggé de la variable consommation (variable 11)
var11 = X_valid.columns[X_valid.columns.str.contains("var11")]

conso_train_prec = X_train[var11]
conso_valid_prec = X_valid[var11]

# Mettre les version laggé de la variable consommation en fin du DataFrame
X_train.drop(var11, axis = 1, inplace = True)
X_valid.drop(var11, axis = 1, inplace = True)

X_train = X_train.join(conso_train_prec)
X_valid = X_valid.join(conso_valid_prec)

cat_features = ["hour", "day"]
num_features = X_valid.drop(["hour", "day"], axis = 1).columns.values

num_pipeline = Pipeline([
	("selector", DataFrameSelector(num_features)),
	("imputer", Imputer(strategy = "median"))
])

cat_pipeline = Pipeline([
	("selector", DataFrameSelector(cat_features)),
	("imputer", OneHotEncoder(sparse = False))
])

full_pipeline = FeatureUnion([
	("cat_pipeline", cat_pipeline),
	("num_pipeline", num_pipeline)
])

X_train_prep = full_pipeline.fit_transform(X_train)

X_valid_prep = full_pipeline.transform(X_valid)

xgb_reg.fit(X_train_prep, y_train)

y_valid_pred = pd.DataFrame(np.zeros((len(y_valid), 1)), columns = ["puissance"], index = y_valid.index)

y_valid_pred.iloc[0, 0] = y_valid.iloc[0]

data_valid_total = np.c_[X_valid_prep, y_valid_pred.values]

for i in range(len(data_valid_total)):
	
	if data_valid_total[i, -1] == 0:
		
		data = data_valid_total[i, :-1].reshape(1, -1)
		
		y_pred = xgb_reg.predict(data)
		
		n_prop = 73 if len(data_valid_total) - i > 73 else len(data_valid_total) - i
		
		j = data_valid_total.shape[1] - 1
		
		for k in range(n_prop):
			
			data_valid_total[k + i, j] = y_pred
			
			j = j - 1

plt.scatter(y_valid, data_valid_total[:, -1])
plt.show()

print(
		f"MAE sur test avec reutilisation des prédictions : {mean_absolute_error(y_valid, data_valid_total[:, -1])}")  # 35.73143939390947

plt.hist(data_valid_total[:, -1])
plt.show()

# ------------------
# Prédiction finale

# Splitter les donnée d'entrainement et les donnée de prédiction
X_train = data_total_reframed[:"2016-09-12"].iloc[:, :-1]
y_train = data_total_reframed[:"2016-09-12"].iloc[:, -1]

X_previ = data_total_reframed["2016-09-13":].iloc[:, :-1]
y_previ = data_total_reframed["2016-09-13":].iloc[:, -1]

X_train["hour"] = X_train.index.hour
X_train["day"] = X_train.index.weekday

X_previ["hour"] = X_previ.index.hour
X_previ["day"] = X_previ.index.weekday

var11 = X_previ.columns[X_previ.columns.str.contains("var11")]

conso_train_prec = X_train[var11]
conso_previ_prec = X_previ[var11]

X_train.drop(var11, axis = 1, inplace = True)
X_previ.drop(var11, axis = 1, inplace = True)

X_train = X_train.join(conso_train_prec)
X_previ = X_previ.join(conso_previ_prec)

cat_features = ["hour", "day"]
num_features = X_previ.drop(["hour", "day"], axis = 1).columns.values

num_pipeline = Pipeline([
	("selector", DataFrameSelector(num_features)),
	("imputer", Imputer(strategy = "median"))
])

cat_pipeline = Pipeline([
	("selector", DataFrameSelector(cat_features)),
	("imputer", OneHotEncoder(sparse = False))
])

full_pipeline = FeatureUnion([
	("cat_pipeline", cat_pipeline),
	("num_pipeline", num_pipeline)
])

X_train_prep = full_pipeline.fit_transform(X_train)

X_previ_prep = full_pipeline.transform(X_previ)

xgb_reg.fit(X_train_prep, y_train)

y_train_pred = xgb_reg.predict(X_train_prep)

print(f"MAE sur l'ensemble des données :  {mean_absolute_error(y_train_pred, y_train)}")  # 21.248667723245488

plt.scatter(y_train_pred, y_train)
plt.show()

plt.hist(y_train_pred)
plt.show()

data_previ_total = np.c_[X_previ_prep, y_previ.values]

for i in range(len(data_previ_total)):
	
	if data_previ_total[i, -1] == 0:
		
		data = data_previ_total[i, :-1].reshape(1, -1)
		
		y_pred = xgb_reg.predict(data)
		
		n_prop = 73 if len(data_previ_total) - i > 73 else len(data_previ_total) - i
		
		j = data_previ_total.shape[1] - 1
		
		for k in range(n_prop):
			
			data_previ_total[k + i, j] = y_pred
			
			j = j - 1

plt.hist(data_previ_total[:, -1])
plt.show()

y_pred_final = pd.DataFrame(data_previ_total[:, -1], columns = ["puissance"],
                            index = meteo_previ_resampled["2016-09-13":].index)
np.savetxt("conso_prev_2.csv", y_pred_final.values, delimiter = ",")
