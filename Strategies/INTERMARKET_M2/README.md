# INTERMARKET_M2 : Bitcoin et Ethereum face à la masse monétaire américaine (M2)

Stratégie Freqtrade présentée sur la chaîne YouTube Freqtrade FR (www.youtube.com/@FreqtradeFR).
C'est l'un des « 4 Mousquetaires » de l'Expérience #5 (https://youtu.be/SpW7c9PQsaE), testé seul.
Cette version est celle préparée pour la vidéo [Update sur les 4 Mousquetaires / Intermarché : M2 loin devant !](https://youtu.be/OLdaW165ZL8).
Elle compare chaque crypto uniquement à M2 ; elle ne combine pas les signaux NASDAQ, S&P 500 et DXY.

## Idée
Elle vient de **neurotrader** : « Intramarket Indicator Differences »
(https://www.youtube.com/watch?v=n2mY86S01fg, code : https://github.com/neurotrader888/IntramarketDifference).
Sa formule (CMMA) et sa règle de signal à seuil sont reprises ici. Son dépôt n'ayant pas de licence,
aucun code n'en est copié : `cmma` et `threshold_revert_signal` sont réécrites d'après sa vidéo. Merci à lui.

## Règle
- Pour chaque marché : `CMMA = (clôture − moyenne mobile sur n) / (ATR sur m × √n)`, soit l'écart à
  la moyenne mesuré en unités de volatilité. Cela rend M2 et une crypto comparables.
- `diff = CMMA(M2) − CMMA(crypto)`.
  - L'état passe à −1 quand `diff < −seuil` et à +1 quand `diff > +seuil`.
  - Il revient à 0 quand `diff` recroise zéro.
- `flip = −1` : **achat quand l'état vaut −1**, c'est-à-dire quand la crypto est nettement plus
  « au-dessus de sa moyenne » que M2. Vente quand l'état revient à 0.
- BTC/USDT et ETH/USDT sont tradés séparément, en spot, sur Binance, en bougies de 12 h.
- Paramètres retenus (`INTERMARKET_M2.json`) :
  - moyenne sur 6 semaines ;
  - ATR sur 35 semaines ;
  - seuil 0,05 ;
  - stoploss −95 %, non déclenché dans les résultats présentés.

## Les données M2 et leur délai de publication
- Série **WM2NS** de la Fed (publication H.6) : milliards de dollars, non désaisonnalisée, une
  valeur par semaine finissant le lundi.
- Une semaine n'est publiée que **10 jours** après sa fin jusqu'en février 2021, puis **22 à
  50 jours** après (publication mensuelle).
- **En direct** (dry run ou live), le bot lit la série publiée sur FRED
  (`fredgraph.csv?id=WM2NS`), avec une copie de secours dans `user_data/m2/`.
- **En backtest**, la stratégie rejoue chaque version publiée de M2, archivée par ALFRED
  (St. Louis Fed). À chaque bougie, elle n'utilise que ce qui était publié à sa clôture, soit à
  00:00 UTC le lendemain de la publication (`m2_delay_hours` dans `config_backtest.json`).
  - Le fichier utilisé dans la vidéo est fourni : `user_data/m2/WM2NS_vintages.csv.gz`, versions
    du 17/08/2017 au 22/09/2026.
  - `download_m2_vintages.py` le met à jour.
- Sans ce rejeu, un backtest utiliserait des valeurs de M2 pas encore connues. Sur cette
  stratégie, cela **doublait** à peu près le résultat : +614 % au lieu de +328 %.

## Utilisation (Docker)
Voir `commands.txt` :
- dry run : `docker compose up -d` ;
- téléchargement des bougies Binance ;
- mise à jour des versions de M2 ;
- backtest ;
- hyperopt, avec `--analyze-per-epoch`, car les paramètres servent dans `populate_indicators`.

Contrôle autonome de la règle de signal, du CMMA et du délai de publication :
`docker compose run --rm --entrypoint python freqtrade user_data/check_strategy.py`.
Ce contrôle ne démarre pas le bot et ne télécharge pas de données.
Le cache FRED est local ; l'archive ALFRED fournie est un jeu de données de recherche public.

## Résultats montrés dans la vidéo (1000 USDT, frais 0,1 %)
Chiffres historiques de la vidéo, sans nouveau backtest lors de cette publication du code.
- **Backtest du 19/04/2018 au 07/03/2025** :
  - +328 % (80 trades, 48,8 % gagnants) ;
  - le Buy & Hold BTC fait +998 % sur les mêmes dates ;
  - drawdown : 56,8 % d'après Freqtrade, qui ne compte que les trades clôturés. En valorisant les
    positions ouvertes heure par heure, il atteint 79,8 %.
  - Les paramètres ont été optimisés sur une partie de cette période.
- **Dry run du 07/03/2025 au 23/09/2026** (hors échantillon) :
  - +101 % en comptant les 2 positions ouvertes ;
  - 11 trades clôturés ;
  - le Buy & Hold BTC fait −5 % sur la même période.
- 11 trades, c'est un petit échantillon. Rien de tout cela n'est un conseil financier.

## Différences avec le bot du dry run
- M2 venait de TradingView (tvDatafeed). Les valeurs sont identiques à FRED (×1e9), vérifié.
- Le décalage de 12 h ajouté le 15/08/2026 est retiré. Il faisait ignorer la dernière semaine
  publiée.
- Le code est nettoyé :
  - spot uniquement ;
  - un seul chargeur M2 ;
  - erreur explicite si la timeframe ne divise pas une semaine.

  Sur les mêmes données, les signaux sont identiques à ceux de la version du bot sauvegardée le
  03/12/2025, en service jusqu'au 15/08/2026.
