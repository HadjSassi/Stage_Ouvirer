import sys
import string
import re
from math import *
from .get_data import get_historical_from_path
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os
import subprocess
import importlib
from datetime import time
from datetime import datetime
from datetime import timedelta
import ccxt
from binance.client import Client
import plotly
import plotly.graph_objects as go

lowerCase = string.ascii_lowercase
upperCase = string.ascii_uppercase


def convert_time(dataframe):
    temps = []
    for elm in dataframe['timestamp']:
        temps.append(datetime.fromtimestamp(elm / 1000))
    dataframe['timestamp'] = pd.DatetimeIndex(pd.to_datetime(temps)).tz_localize('UTC').tz_convert('UTC')
    return dataframe


def detection_mauvais_shape(dictionaire_crypto):
    liste_shape = []
    liste_crypto = []
    boulean = []
    for elm in dictionaire_crypto:
        liste_shape.append(dictionaire_crypto[elm].shape[0])
        liste_crypto.append(elm)
    for elm in liste_shape:
        if elm < np.max(liste_shape):
            boulean.append(True)
        else:
            boulean.append(False)
    boulean, liste_crypto = np.array(boulean), np.array(liste_crypto)
    return liste_crypto[boulean]


def correction_shape(dictionaire_crypto, array):
    max_shape = []
    shape_a_manque = []
    liste_final = []
    nom_shape_a_manque = []

    # onc cherche le shape maximun dans tous le array
    for elm in dictionaire_crypto:
        max_shape.append(dictionaire_crypto[elm].shape[0])
    max_shape = np.max(max_shape)

    # on calcul le shape manquant dans le array
    for elm1 in array:
        shape_a_manque.append(max_shape - dictionaire_crypto[elm1].shape[0])
        nom_shape_a_manque.append(elm1)
    for shape, nom in zip(shape_a_manque, nom_shape_a_manque):
        liste_final = [np.ones(shape), np.zeros(shape)]
        df_liste_final = pd.DataFrame(np.transpose(liste_final), columns=[nom[:3] + '_open', nom[:3] + '_close'])
        dictionaire_crypto[nom] = pd.concat((df_liste_final, dictionaire_crypto[nom]), axis=0)
    return dictionaire_crypto


def generation_date(dataframe, delta_pas):
    test_list = []
    pas = int(delta_pas)
    date_ini = dataframe.index[::-1][0]
    inverse_time = dataframe.index[::-1]
    for i in range(len(inverse_time)):
        test_list.append(date_ini - pas * i)
    test_list = test_list[::-1]
    return test_list


def variationN(cryptos, ni):
    ni = ni.upper()
    if (ni == 'N'):
        for crypto in cryptos:
            cryptos[crypto]["Variation_" + crypto[:-5]] = cryptos[crypto][crypto[:-5] + "_close"] / cryptos[crypto][
                crypto[:-5] + "_open"]
            cryptos[crypto]["Variation_N_" + crypto[:-5]] = cryptos[crypto][crypto[:-5] + "_close"] / cryptos[crypto][
                crypto[:-5] + "_open"]
    elif (ni == 'N-1'):
        for crypto in cryptos:
            cryptos[crypto]["Variation_N_" + crypto[:-5]] = cryptos[crypto][crypto[:-5] + "_close"] / cryptos[crypto][
                crypto[:-5] + "_open"]
            cryptos[crypto]["Variation_" + crypto[:-5]] = 0.0
            cryptos[crypto]["Variation_" + crypto[:-5]][0] = float(cryptos[crypto][crypto[:-5] + "_close"][0]) / float(
                cryptos[crypto][crypto[:-5] + "_open"][0])
            for j in range(1, len(cryptos[crypto])):
                cryptos[crypto]["Variation_" + crypto[:-5]][j] = cryptos[crypto][crypto[:-5] + "_close"][j] / \
                                                                 cryptos[crypto][crypto[:-5] + "_open"][j - 1]
    elif (ni == 'N-2'):
        for crypto in cryptos:
            cryptos[crypto]["Variation_N_" + crypto[:-5]] = cryptos[crypto][crypto[:-5] + "_close"] / cryptos[crypto][
                crypto[:-5] + "_open"]
            cryptos[crypto]["Variation_" + crypto[:-5]] = 0.0
            cryptos[crypto]["Variation_" + crypto[:-5]][0] = float(cryptos[crypto][crypto[:-5] + "_close"][0]) / float(
                cryptos[crypto][crypto[:-5] + "_open"][0])
            cryptos[crypto]["Variation_" + crypto[:-5]][1] = float(cryptos[crypto][crypto[:-5] + "_close"][1]) / float(
                cryptos[crypto][crypto[:-5] + "_open"][0])
            for j in range(2, len(cryptos[crypto])):
                cryptos[crypto]["Variation_" + crypto[:-5]][j] = cryptos[crypto][crypto[:-5] + "_close"][j] / \
                                                                 cryptos[crypto][crypto[:-5] + "_open"][j - 2]
    return (cryptos)


def coeffMulti(cryptos):
    for crypto in cryptos:
        for i in range(len(cryptos[crypto].index)):
            if (i == 0):
                cryptos[crypto]["Coeff_mult_" + crypto[:-5]] = cryptos[crypto][crypto[:-5] + "_close"][0] / \
                                                               cryptos[crypto][crypto[:-5] + "_open"][0]
            else:
                cryptos[crypto]["Coeff_mult_" + crypto[:-5]][i] = cryptos[crypto]["Variation_N_" + crypto[:-5]][i] * \
                                                                  cryptos[crypto]["Coeff_mult_" + crypto[:-5]][i - 1]
                # print(cryptos[crypto]["Variation_N_" + crypto[:-5]][i]," * ",  cryptos[crypto]["Coeff_mult_" + crypto[:-5]][i - 1] ," = ",cryptos[crypto]["Coeff_mult_" + crypto[:-5]][i] )

    return cryptos


def mergeCryptoTogether(cryptos):
    for i in cryptos:
        cryptos["BOT_MAX"] = cryptos[i].copy()
        cryptos["BOT_MAX"].rename(columns={"Variation_" + i[:-5]: "Variation_BOTMAX"}, inplace=True)
        cryptos["BOT_MAX"].rename(columns={i[:-5] + "_close": "Variation2BOTMAX"}, inplace=True)
        cryptos["BOT_MAX"].rename(columns={"Coeff_mult_" + i[:-5]: "Coeff_mult_BOTMAX"}, inplace=True)
        cryptos["BOT_MAX"].rename(columns={"Variation_N_" + i[:-5]: "Variation_BOTMAX_N"}, inplace=True)
        break
    cryptos = pd.concat(cryptos, axis=1)
    return cryptos


def botMax(cryptos):
    maxis = []
    for i in range(len(cryptos)):
        v = []
        k = 0
        for j in range(len(cryptos.iloc[i])):
            if (k == 3):
                v.append(cryptos.iloc[i].iloc[j])
            elif (k == 4):
                k = -1
            k += 1
        maxx = max(v)
        maxis.append(v.index(maxx))
        cryptos["BOT_MAX"]["Variation_BOTMAX"][i] = maxx
    return cryptos, maxis


def botMaxVariation2(cryptos, maxis):
    botNames = []
    for crypto in cryptos:
        if (crypto[0] not in botNames):
            botNames.append(crypto[0])
    for i in range(len(cryptos)):
        botName = botNames[maxis[i]]
        cryptos["BOT_MAX"]["Variation_BOTMAX_N"][i] = cryptos[botName]["Variation_N_" + botName[:-5]][i]
    for i in range(0, len(cryptos) - 1):
        botName = botNames[maxis[i]]
        cryptos["BOT_MAX"]["Variation2BOTMAX"][i + 1] = cryptos[botName]["Variation_N_" + botName[:-5]][i + 1]
    cryptos["BOT_MAX"]["Variation2BOTMAX"][0] = 0

    return cryptos


def coeffMultiBotMax(cryptos):
    cryptos["BOT_MAX"]["Coeff_mult_BOTMAX"][0] = 1.000
    for i in range(1, len(cryptos)):
        cryptos["BOT_MAX"]["Coeff_mult_BOTMAX"][i] = cryptos["BOT_MAX"]["Coeff_mult_BOTMAX"][i - 1] * \
                                                     cryptos["BOT_MAX"]["Variation2BOTMAX"][i]
    return cryptos


def coefmultiFinal(cryptos):
    tabe = {}
    for crypto in cryptos:
        if (crypto[1].find("Coeff_mult_") == 0):
            tabe[crypto[1]] = cryptos[crypto]

    tabe = pd.DataFrame(tabe)
    return tabe


def VariationFinal(cryptos):
    tabe = {}
    for crypto in cryptos:
        if (crypto[1].find("Variation") == 0):
            tabe[crypto[1]] = cryptos[crypto]

    tabe = pd.DataFrame(tabe)
    return tabe


def plot_courbes2(df_tableau_multi):
    fig = go.Figure()
    for elm in df_tableau_multi.columns:
        fig.add_trace(go.Scatter(x=df_tableau_multi[elm].index,
                                 y=df_tableau_multi[elm],
                                 mode='lines',
                                 name=elm,
                                 ))
    return plotly.plot(fig)


def execute_terminal_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Print the standard output of the command
        print(result.stdout)

        # If you want to capture the standard error as well, uncomment the line below
        # print(result.stderr)

        # Check the return code
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Command execution failed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def install_package_if_needed(package):
    try:
        # Check if the package is installed
        subprocess.run(['node', '-e', 'require.resolve("{}")'.format(package)], check=True, capture_output=True)
        print(f"{package} is already installed.")
    except subprocess.CalledProcessError as e:
        # Package not found, install it
        print(f"{package} is not installed. Installing...")
        try:
            command = f'npm install {package}'
            execute_terminal_command(command)
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")


def remove_non_csv_files(directory):
    for filename in os.listdir(directory):
        if not filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")


def get_analyisis_from_window(df_list,window=24):
    metric_list = {}
    for coin in df_list:
        try:
            df = df_list[coin]
            df.drop(columns=df.columns.difference(['open', 'high', 'low', 'close', 'volume']), inplace=True)
            df["rvolume"] = df["volume"] * df["close"]
            df["olhc"] = (df["open"] + df["low"] + df["high"] + df["close"]) / 4
            df["atr"] = (ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close']) / df[
                "close"]) * 100
            # df["norm"].dropna(inplace=True)
            volatility = df["atr"].mean()
            mean_volume = df["rvolume"].mean()
            std = df["olhc"].std() / df["olhc"].mean()
            perf = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
            return_vs_vol = perf / volatility
            last_volume = df["rvolume"].iloc[-window:].mean()
            volume_evolution = last_volume / mean_volume
            last_volatility = df.iloc[-window:]["olhc"].std() / df.iloc[-window:]["olhc"].mean()
            last_perf = (df.iloc[-1]["close"] - df.iloc[-window]["close"]) / df.iloc[-window]["close"]
            volatility_evolution = last_volatility / volatility
            last_return_vs_vol = last_perf / last_volatility
            metric_list[coin] = {
                "mean_volume": round(mean_volume),
                "last_volume": round(last_volume),
                "volume_evolution": round(volume_evolution, 2),
                "mean_volatility": round(volatility, 2),
                "last_volatility": round(last_volatility * 100, 2),
                "volatility_evolution": round(volatility_evolution * 100, 2),
                "return": round(perf * 100, 2),
                "last_return": round(last_perf * 100, 2),
                "return_vs_vol": round(return_vs_vol * 100, 2),
                "last_return_vs_vol": round(last_return_vs_vol, 2)
            }
        except:
            # print("Error on", coin)
            # raise
            pass

    return pd.DataFrame.from_dict(metric_list, orient='index')
