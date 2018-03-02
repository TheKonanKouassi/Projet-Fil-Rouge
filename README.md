Projet Fil Rouge
================
Encadrant :  
 - Benoit Gaüzère

Etudiants : 
 - Kassir SAROUKOU
 - Kouassi KONAN
 - Rodrigue GUEDOU

Description:

Disposant de données météorologiques à fréquence tri-horaire, on souhaite pouvoir prédire la consommation horaire (en kwh) électrique de l’île d'Ouessant à 7 jours [lien](https://www.sfds.asso.fr/fr/jeunes_statisticiens/manifestations/610-data_challenge_jds_2018/)

PV Rencontre du 08 Décembre 2017
===============================

Kassir : Définissons nos objectifs finaux avant de se lancer dans le projet véritablement ?

Kassir : Être plus spécifique sur notre « target » afin d’être plus efficace.

Ange : Quel est notre dérivable à la fin ?

Kassir : Livrer une application dans laquelle on renseigne le nom d’une entité et celle-ci fait une analyse de réputation sur cette application.

Ange : Une page web à la Google, dans laquelle on a une bar de recherche et dans laquelle on renseigne le nom d’une entreprise et après on reçoit l’analyse de sentiment et l’analyse de la situation de l’entreprise.

Rodrigue : Faire une application de base, avec une seule source et ensuite rajouter des sources, pour plus agréger.

Kassir : Je ne vois pas l’utilité de rajouter les informations comme les indicateurs financiers et les d’autres infos.

Ange : Faire un Word Cloud du nom renseigné.

Kassir : Faire un classement des entreprises dans leur domaine.

Rodrigue : Faire un algorithme directement

Kassir et Ange : il faudrait faire une base donnée d’abord.


PV Rencontre du 13 Décembre 2017
===============================

Gauzère : Qu’est-ce que vous voulez faire à la fin

Ange : Score pour toute entreprise

Rodrigue : Faire du text-mining a partir de twitter

Kassir : Une entreprise

Gauzère : Il faut d’abord définir ce qu’il y a en entré et en sorti.

Gauzère : C’est quoi le score ? Il faut se comparer à quelque chose qu’on connait déjà. Donc trouver une réputation d’une entreprise, à laquelle comparer nos résultats.

Étapes :
Bibliothèque : Google Scholar et chercher le mot clé e-réputation
Construire un data set
Choix des données (restaurant, ou entreprise, ou les données qu’on vise
Choix du modèle
Évaluation

Pour la semaine prochaine, faire un résumé.
Pour le 16 février faire un résumé de la partie 1) et 2).


PV Rencontre du 16 Février 2018
===============================

Après soumission d’un nouveau sujet portant sur un challenge proposé par le groupe des jeunes Statisticiens.ne.s [links](https://www.sfds.asso.fr/fr/jeunes_statisticiens/manifestations/610-data_challenge_jds_2018/), nous avons discuter sur la manière d’aborder le problème.

Le problème a été identité comme un sujet de série temporelle, dans lequel on dispose d’une fenêtre glissante (si on voit led données sur une semaine) sur une année entière. 

L’objectif étant de prédire les consommations horaires à partir de données météorologique tri-horaire.

Une première méthode de résolution du problème à été proposée (par les étudiants). Elle consiste à prédire la première de consommation à partir du data set, ensuite de prédire la second heure à partir de la data set et de la prédiction fait, et ainsi de suite. 

La prochaine rencontre a été fixé au vendredi 23 Février. Il s’agira de faire un résumé de la bibliographie qui traite du sujet.


PV Rencontre du 23 Février 2018
===============================

Après quelques recherches et lectures, l’on a pu distinguer trois classes de méthodologies.

2.1. Approches Stochastiques : ARIMA


Les modèles Autoregressive Integrated Moving Average (ARIMA) sont les modèles les plus utilisés pour la modélisation et la prédiction des séries temporelles. Ces modèles sont une généralisation des modèles Autoregressive Moving Average (ARMA). Dans le cadre de ces modèles, une hypothèse forte est faite : on considère que la série temporelle est linéaire et suit une certaine distribution statistique connue. Plusieurs modèles sont développés notamment les modèles Seasonal Autoregressive integrated moving average (SARIMA).

La popularité des modèles ARIMA est rendu possible grâce à leur flexibilité et leur facilité d’implementation. Cependant du fait de l’hypothèse de linéarité, ces modèles sont limités et deviennent inadéquates dans plusieurs situations.


2.2 Approches Réseau de Neurones : NN, RNN

Ces modèles sont basés sur les données elle même, ne fait aucune hypothèses statistiques sur celles-ci. Il n’y a aucun besoin de spécifier un model en particulier, car les modèles sont adaptatifs. Les réseaux de neurones, ne faisant aucune hypothèses sur la linéarité des données, ils sont donc plus adaptatif à des modèles complexes et plus précis. 
Cependant les réseaux de neurones ont des points faibles. Un très grand réseau, contenant donc énormément de paramètres à estimer, peut premièrement être très difficile à entrainer, et ensuite peut mener à un sur-apprentissage. Par ailleurs, il n’existe toujours pas de méthodes théoriquement prouvées pour trouver les paramètres optimaux pour le réseau que l’on souhaite construire. 
Il existe plusieurs approches, parmi lesquels l’on pourra citer :

Feed forward Network (FNN)
Time Lagged Neural Network (TLNN)
Seasonal Artificial Neural Network (SANN) 
LSTM
BLSTM
Hidden Markov Model

Par ailleurs, une autre méthode alternative qui elle est un peut plus complexe à mettre en place est la méthode du processus Gaussien (Gaussian Processes).


2.3 Approches Machine à Support de Vecteurs : SVM

Comme son nom l’indique, il s’agit de modèles basés sur  le modèle de Machine à Support de Vecteurs. Les SVM étaient utilisés pour les problèmes de classification, mais aujourd’hui il existe plusieurs modèles qui permettent le traitement de regression et donc de série temporelles, en l’occurrence:

LS-SVM
DLS-SVM


3. Modélisation des Séries Temporelles

Il est en général considéré qu'une série est composée d'une tendance (évolution naturelle), d'une saisonnalité (événements récurrents aux mêmes périodes) et d'un bruit blanc (marche aléatoire) [lien]. On peut distinguer deux classes de modèles:

Modèle Paramétrique Addictif :  Série  = Tendance + Saisonnalité + Bruit
Modèle Multiplicatif : Série = Tendance x Saisonnalité x Bruit


3.1 Tendance

La tendance peut être éliminée par différenciation ou par régression. Une permettant d’étudier cette tendance est statsmodels. La fonction detrend retourne la tendance. On l’obtient en réalisant une régression linéaire de Y sur le temp t.

3.2 Saisonnalité

La saisonnalité peut être déterminée grâce au calcul du coefficient d’auto-corrélation exprimé en fonction du décalage entre deux observations. La période correspond, si elle existe, à un maximum d’auto-corrélation.
Les libraries statsmodels, fipy ou SfePy, permettent de determiner cette composante.


4. References

https://arxiv.org/pdf/1302.6613.pdf

https://moodle.insa-rouen.fr/pluginfile.php/85014/mod_resource/content/3/rnn_2018_mastere.pdf

http://www.robots.ox.ac.uk/~sjrob/Pubs/philTransA_2012.pdf

https://en.wikipedia.org/wiki/Time_series#Models

https://en.wikipedia.org/wiki/Recurrent_neural_network

https://en.wikipedia.org/wiki/Artificial_neural_network

https://en.wikipedia.org/wiki/Long_short-term_memory

https://en.wikipedia.org/wiki/Hidden_Markov_model

https://www.ijerm.com/download_data/IJERM0401035.pdf

http://jmlr.org/papers/volume3/gers02a/gers02a.pdf

https://perso.univ-rennes1.fr/valerie.monbet/ST_M1/CoursST2012.pdf

http://freakonometrics.free.fr/M1_TS_2015.pdf

https://www.unige.ch/~wanner/teaching/Numi/Numi2.pdf

https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

http://olivier.guibe.free.fr/depot/Python_Experiences/slides/deux_inter.html#9

http://python-scientific-lecture-notes.developpez.com/tutoriels/note-cours/calcul-scientifique-haut-niveau-apprendre-scipy/#LVIII

https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f

https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial




Après présentation de ces résultats, le problème de la structuration de la base de donnée se posait.
Rodrigue proposait une version pivoter des données tandis que Kouassi et Kassir optaient pour une utilisation comme tel de la base. Et donc pour trancher, il s'agissait de faire des test et d'en garder la structure de données qui minimisait les erreurs.

PV Rencontre du 02 Mars 2018
============================

- Quelques soucis de connexion on fait que tout le monde n'a pas pu participer à la rencontre nottament Kassir.

- Kouassi a eu a presenter ces premiers resultats. En gros il a supprime les donnees manquantes, fait une interpolation lineaire afin d'opteir des dimmensions egales pour les prédicteurs et la variable a predire. Enfin, il a essayé un reseau de neuronnes (LSTM), mais avec une version modifie du probleme, qu'il a pose comme un probleme de classification multicalsse. Les commentaires de M. Gaüzère est qu'il ne faut pas prendre plus de temps sur ce modèle et qu'il fallait passer directement à de la regression.

- Les ayant des soucis de connexion n'ont pas pu présenter leurs résultats.

- Pour la prochaine rencontre fixé au mardi 13 mars 2918, un pour devra être fait sur la structuration definitve adoptee, et presenter les premiers resultats.


PV Rencontre du 13 Mars 2018
============================
...
