import requests
import pandas as pd
import numpy as np
import numba
from scipy.stats import linregress
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@numba.njit
def shift_nb(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))

def get_ranked_extrema(h, l, max_neighbours=10):
    maxima = np.zeros(shape=(h.shape[0]), dtype=int)
    minima = np.zeros(shape=(h.shape[0]), dtype=int)

    for n in range(1, max_neighbours+1):
        maxima_filter = np.where(maxima == n-1)[0]
        maxima[maxima_filter] += ((h >= shift_nb(h, n)) & (h > shift_nb(h, -n)))[maxima_filter]

        minima_filter = np.where(minima == n-1)[0]
        minima[minima_filter] += ((l <= shift_nb(l, n)) & (l < shift_nb(l, -n)))[minima_filter]

    return maxima, minima

def get_best_channel(
        h, l, c, max_rank_possible=20, min_rank_for_detection=3,
        min_pivot_point_count=3, max_age_latest_pivot_point=20,
        min_r_squared=0.9, max_p_value=0.5, max_crossing_perc=0.05,
        min_pivot_points_inbetween=1):
    # Calculate the rank of each high and low
    all_maxima, all_minima = get_ranked_extrema(h, l, max_rank_possible)

    # Filter out minima whose rank is too low
    all_minima_x = np.where(all_minima >= min_rank_for_detection)[0]
    all_minima_y = l[all_minima_x]
    all_minima_weights = all_minima[all_minima_x]

    # Filter out maxima whose rank is too low
    all_maxima_x = np.where(all_maxima >= min_rank_for_detection)[0]
    all_maxima_y = h[all_maxima_x]
    all_maxima_weights = all_maxima[all_maxima_x]

    best_lower_trend_line = None
    best_lower_trend_stats = None
    best_upper_trend_line = None
    best_upper_trend_stats = None
    best_channel_score = -1

    # "Sort minima by their rank and after each iteration remove the minimum with the lowest rank"
    for minima_count in range(min_pivot_point_count, all_minima_x.shape[0]+1):
        minima_comb = np.argpartition(all_minima_weights, -minima_count)[-minima_count:]

        # Get combination of minima
        minima_x = all_minima_x[minima_comb]
        minima_y = all_minima_y[minima_comb]
        minima_weights = all_minima_weights[minima_comb]

        # Do linear regression for these minima
        min_slope, min_intercept, min_r, min_p, min_err = linregress(minima_x, minima_y)
        # The coefficient of determination needs to be high enough
        if min_r*min_r <= min_r_squared or min_p >= max_p_value:
            continue

        # Calculate the regression line
        min_line_x = np.arange(np.min(minima_x), c.shape[0])
        min_line_y = min_intercept + min_slope * min_line_x
        # Calculate the crossing price percentage
        min_crossing_price_perc = np.mean(c[min_line_x] < min_line_y)
        # This needs to be sufficiently small
        if min_crossing_price_perc >= max_crossing_perc:
            continue

        # "Sort maxima by their rank and after each iteration remove the maximum with the lowest rank"
        for maxima_count in range(min_pivot_point_count, all_maxima_x.shape[0]+1):
            maxima_comb = np.argpartition(all_maxima_weights, -maxima_count)[-maxima_count:]

            # Get combination of maxima
            maxima_x = all_maxima_x[maxima_comb]
            maxima_y = all_maxima_y[maxima_comb]
            maxima_weights = all_maxima_weights[maxima_comb]

            # At least <min_pivot_points_inbetween> minima need to be between the first and the last maxima
            if np.sum((np.min(maxima_x) <= minima_x) & (minima_x <= np.max(maxima_x))) < min_pivot_points_inbetween:
                continue
            # At least <min_pivot_points_inbetween> maxima need to be between the first and the last minima
            if np.sum((np.min(minima_x) <= maxima_x) & (maxima_x <= np.max(minima_x))) < min_pivot_points_inbetween:
                continue

            # Do linear regression for these maxima
            max_slope, max_intercept, max_r, max_p, max_err = linregress(maxima_x, maxima_y)
            # The coefficient of determination needs to be high enough
            if max_r*max_r <= min_r_squared or max_p >= max_p_value:
                continue

            # Calculate the regression line
            max_line_x = np.arange(np.min(maxima_x), c.shape[0])
            max_line_y = max_intercept + max_slope * max_line_x
            # Calculate the crossing price percentage
            max_crossing_price_perc = np.mean(c[max_line_x] > max_line_y)
            # This needs to be sufficiently small
            if max_crossing_price_perc >= max_crossing_perc:
                continue

            # The latest pivot point should not be too old
            if c.shape[0] - max(np.max(minima_x), np.max(maxima_x)) > max_age_latest_pivot_point:
                continue

            # Calculate a score that makes channel A better than channel B (difficult and can be definitely changed)
            channel_score = (np.sum(minima_weights) * np.sum(maxima_weights)) * min_r*min_r * max_r*max_r * (1-min_p)*(1-max_p)

            if best_channel_score < channel_score:
                best_channel_score = channel_score
                best_lower_trend_line = {
                    "line_x": min_line_x,
                    "line_y": min_line_y,
                    "points_x": minima_x,
                    "points_y": minima_y,
                    "points_weights": minima_weights
                }
                best_lower_trend_stats = {
                    "slope": min_slope,
                    "intercept": min_intercept,
                    "r": min_r,
                    "p": min_p,
                    "err": min_err,
                    "crossing_price_perc": min_crossing_price_perc
                }
                best_upper_trend_line = {
                    "line_x": max_line_x,
                    "line_y": max_line_y,
                    "points_x": maxima_x,
                    "points_y": maxima_y,
                    "points_weights": maxima_weights
                }
                best_upper_trend_stats = {
                    "slope": max_slope,
                    "intercept": max_intercept,
                    "r": max_r,
                    "p": max_p,
                    "err": max_err,
                    "crossing_price_perc": max_crossing_price_perc
                }

    return best_lower_trend_line, best_lower_trend_stats, best_upper_trend_line, best_upper_trend_stats, best_channel_score

def fetch_crypto_market_data(crypto_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/ohlc"
    params = {
        "vs_currency": "usd",
        "days": days
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        ohlc = data
        if ohlc:
            df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        else:
            print("No data found.")
    else:
        print("Error fetching data:", data.get("error"))
    return None

WINDOW_SIZE = 80
PLOT_FUTURE_PERIODS = 40

last_plotted_i = -999

import streamlit as st

st.title("OHLCV Graph")
crypto_id = st.text_input("enter the cryptocurrency's name. Ex: bitcoin, ethereum: ")
#crypto_id = input("enter the cryptocurrency's name. Ex: bitcoin, ethereum: ")
days = st.text_input("enter the time frame (in days). Ex: 30, 60: ")
#days=input("enter the time frame (in days). Ex: 30, 60: ")
crypto_curr = st.text_input("Enter cryptocurrency code. Ex: BTC-USD: ")

ohlcv = fetch_crypto_market_data(crypto_id,days)

for i in tqdm(range(WINDOW_SIZE, ohlcv.shape[0])):
    o = ohlcv.iloc[i-WINDOW_SIZE+1:i+1, 0].values
    h = ohlcv.iloc[i-WINDOW_SIZE+1:i+1, 1].values
    l = ohlcv.iloc[i-WINDOW_SIZE+1:i+1, 2].values
    c = ohlcv.iloc[i-WINDOW_SIZE+1:i+1, 3].values

    best_lower_trend_line, best_lower_trend_stats, best_upper_trend_line, best_upper_trend_stats, best_channel_score = get_best_channel(
        h=h, l=l, c=c,
        max_rank_possible=20,
        min_rank_for_detection=3,
        min_pivot_point_count=3,
        max_age_latest_pivot_point=20,
        min_r_squared=0.9,
        max_p_value=0.5,
        max_crossing_perc=0.05,
        min_pivot_points_inbetween=1
    )

    if best_channel_score < 1000:
        continue

    # This is just to prevent plotting the same channels a lot times in a row,
    # so the very first appearance of the channel is shown
    if i - last_plotted_i < 10:
        continue

    print(" best_lower_trend_line: ", best_lower_trend_line)
    print("best_lower_trend_stats: ", best_lower_trend_stats)
    print(" best_upper_trend_line: ", best_upper_trend_line)
    print("best_upper_trend_stats: ", best_upper_trend_stats)
    print("    best_channel_score: ", best_channel_score)

    plot_future_periods = min(i+PLOT_FUTURE_PERIODS+1, ohlcv.shape[0])
    o1 = ohlcv.iloc[i-WINDOW_SIZE+1:plot_future_periods, 0].values
    h1 = ohlcv.iloc[i-WINDOW_SIZE+1:plot_future_periods, 1].values
    l1 = ohlcv.iloc[i-WINDOW_SIZE+1:plot_future_periods, 2].values
    c1 = ohlcv.iloc[i-WINDOW_SIZE+1:plot_future_periods, 3].values

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.07, subplot_titles=('OHLC',""),
                row_width=[0.15, 0.7], )

    fig.add_trace(go.Candlestick(
        x=np.arange(o1.shape[0]),
        open=o1,
        high=h1,
        low=l1,
        close=c1,
        name="OHLC"),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=best_upper_trend_line["points_x"], y=best_upper_trend_line["points_y"], text=['Rank: {}'.format(r) for r in best_upper_trend_line["points_weights"]], mode="markers",opacity=0.7, marker=dict(
            color='Purple',
            size=best_upper_trend_line["points_weights"]/2+10
        ), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_lower_trend_line["points_x"], y=best_lower_trend_line["points_y"], text=['Rank: {}'.format(r) for r in best_lower_trend_line["points_weights"]], mode="markers", opacity=0.7, marker=dict(
            color='Blue',
            size=best_lower_trend_line["points_weights"]/2+10
        ), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_upper_trend_line["line_x"], y=best_upper_trend_line["line_y"], opacity=0.7, showlegend=False, text=f"Error: {best_upper_trend_stats['err']}<br>R: {best_upper_trend_stats['r']}<br>P: {best_upper_trend_stats['p']}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=best_lower_trend_line["line_x"], y=best_lower_trend_line["line_y"], opacity=0.7, showlegend=False, text=f"Error: {best_lower_trend_stats['err']}<br>R: {best_lower_trend_stats['r']}<br>P: {best_lower_trend_stats['p']}"), row=1, col=1)

    fig.add_vline(x=WINDOW_SIZE-1, line_width=1, line_dash="dash", line_color="black", row=1, col=1, annotation_text="now")

    fig.update_layout(height=500, width=800)
    fig.update(layout_xaxis_rangeslider_visible=False)

    last_plotted_i = i

    break

st.plotly_chart(fig)

import yfinance as yf
import plotly.graph_objects as go

def fetch_crypto_volume_data(crypto_curr, period="200d"):
    # Fetch data using yfinance
    crypto_mapping = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "ripple": "XRP-USD",
    "litecoin": "LTC-USD",
    "cardano": "ADA-USD",
    "polkadot": "DOT1-USD",
    "chainlink": "LINK-USD",
    "stellar": "XLM-USD",
    "bitcoin-cash": "BCH-USD",
    "binancecoin": "BNB-USD",
    "eos": "EOS-USD",
    "tezos": "XTZ-USD",
    "monero": "XMR-USD",
    "tron": "TRX-USD",
    "cosmos": "ATOM1-USD",
    "filecoin": "FIL-USD",
    "aave": "AAVE-USD",
    "uniswap": "UNI-USD",
    "solana": "SOL1-USD",
    "stellar-lumens": "XLM-USD",
    "neo": "NEO-USD",
    "vechain": "VET-USD",
    "theta-token": "THETA-USD",
    "dash": "DASH-USD",
    "compound": "COMP-USD",
    "algorand": "ALGO-USD",
    "maker": "MKR-USD",
    "yearn-finance": "YFI-USD",
    "avalanche": "AVAX-USD",
    "dogecoin": "DOGE-USD",
    "nem": "XEM-USD",
    "matic-network": "MATIC-USD",
    "ethereum-classic": "ETC-USD",
    "decred": "DCR-USD",
    "zilliqa": "ZIL-USD",
    "huobi-token": "HT-USD",
    "waves": "WAVES-USD",
    "the-graph": "GRT2-USD",
    "kava": "KAVA-USD",
    "ren": "REN-USD",
    "yearn-finance-ii": "YFII-USD",
    "ontology": "ONT-USD",
    "basic-attention-token": "BAT-USD",
    "digibyte": "DGB-USD",
    "0x": "ZRX-USD",
    "siacoin": "SC-USD",
    "qtum": "QTUM-USD",
    "icon": "ICX-USD",
    "bitcoin-gold": "BTG-USD",
    "elrond-egld": "EGLD-USD",
    "enjincoin": "ENJ-USD",
    "reserve-rights": "RSR-USD",
    "bitcoin-sv": "BSV-USD",
    "ravencoin": "RVN-USD",
    "decentraland": "MANA-USD",
    "horizen": "ZEN-USD",
    "omisego": "OMG-USD",
    "sushi": "SUSHI-USD",
    "pancakeswap": "CAKE-USD",
    "band-protocol": "BAND-USD",
    "helium": "HNT-USD",
    "holotoken": "HOT-USD",
    "bitcoin-diamond": "BCD-USD",
    "nervos-network": "CKB-USD",
    "zencash": "ZEN-USD",
    "terra-luna": "LUNA1-USD",
    "havven": "SNX-USD",
    "bitcoin-cash-abc": "BCHA-USD",
    "trueusd": "TUSD-USD",
    "pax-gold": "PAXG-USD",
    "compound-governance-token": "COMP-USD",
    "uma": "UMA-USD",
    "decentraland-mana": "MANA-USD",
    "matic-network-matic": "MATIC-USD",
    "loopring": "LRC-USD",
    "audius": "AUDIO-USD",
    "augur": "REP-USD",
    "bitcoin-cash-sv": "BSV-USD",
    "hathor": "HTR-USD",
    "ankr": "ANKR-USD",
    "sia": "SC-USD",
    "harmony": "ONE1-USD",
    "energy-web-token": "EWT1-USD",
    "maidsafecoin": "MAID-USD",
    "stakecube-coin": "SCC-USD",
    "kin": "KIN-USD",
    "algorand-algo": "ALGO-USD",
    "quantstamp": "QSP-USD",
    "fantom": "FTM-USD",
    "iotex": "IOTX-USD",
    "wazirx": "WRX-USD",
    "trueusd-tusd": "TUSD-USD",
    "ethlend": "LEND-USD",
    "wanchain": "WAN-USD",
    "siacoin-sc": "SC-USD",
    "telcoin": "TEL-USD",
    "nxt": "NXT-USD",
    "mithril": "MITH-USD",
    "contentos": "COS-USD",
    "chromia": "CHR-USD",
    "ocean-protocol": "OCEAN-USD",
    "rsr": "RSR-USD",
    "wazirx-wrx": "WRX-USD",
    "the-sandbox": "SAND-USD",
    "kava-iovs": "KAVA-USD",
    "iost": "IOST-USD",
    "decentraland-mana": "MANA-USD",
    "digibyte-dgb": "DGB-USD",
    "vethor-token": "VTHO-USD",
    "curve-dao-token": "CRV-USD",
    "wazirx-wrx": "WRX-USD",
    "status": "SNT-USD",
    "wazirx": "WRX-USD",
    "orchid": "OXT-USD",
    "stakecube-coin": "SCC-USD",
    "stakenet": "XSN-USD",
    "ankr": "ANKR-USD",
    "adx-net": "ADX-USD",
    "stratis": "STRAX-USD",
    "stratis": "STRAX-USD",
    "troy": "TROY-USD",
    "beam": "BEAM-USD",
    "origin-protocol": "OGN-USD",
    "revain": "REV-USD",
    "blockstack": "STX-USD",
    "binance-usd": "BUSD-USD",
    "bitshares": "BTS-USD",
    "dent": "DENT-USD",
    "ravencoin": "RVN-USD",
    "livepeer": "LPT-USD",
    "fetch-ai": "FET-USD",
    "celer-network": "CELR-USD",
    "republic-protocol": "REN-USD",
    "nimiq": "NIM-USD",
    "lisk": "LSK-USD",
    "tokencard": "TKN-USD",
    "constellation-labs": "DAG-USD",
    "aragon": "ANT-USD",
    "v-id-blockchain": "VIDT-USD",
    "zcoin": "XZC-USD",
    "digitalbits": "XDB-USD",
    "safecoin": "SAFE-USD",
    "sora": "XOR-USD",
    "everipedia": "IQ-USD",
    "unibright": "UBT-USD",
    "bit-torrent": "BTT-USD",
    "singularitynet": "AGI-USD",
    "te-food": "TFUEL-USD",
    "all-sports": "SOC-USD",
    "oasis-network": "ROSE-USD",
    "rari-governance-token": "RGT-USD",
    "botxcoin": "BOTX-USD",
    "frax": "FRAX-USD",
    "lido-dao": "LDO-USD",
    "keeperdao": "ROOK-USD",
    "blox": "CDT-USD",
    "dusk-network": "DUSK-USD",
    "ethlend": "LEND-USD",
    "btu-protocol": "BTU-USD",
    "xensor": "XSR-USD",
    "qash": "QASH-USD",
    "funfair": "FUN-USD",
    "oneledger": "OLT-USD",
    "bread": "BRD-USD",
    "centrality": "CENNZ-USD",
    "btc-standard-hashrate-token": "BTCST-USD",
    "yield-app": "YLD-USD",
    "ark": "ARK-USD",
    "sharering": "SHR-USD",
    "unification": "FUND-USD",
    "rightmesh": "RMESH-USD",
    "lto-network": "LTO-USD",
    "civic": "CVC-USD",
    "winding-tree": "LIF-USD",
    "akash-network": "AKT-USD",
    "telos": "TLOS-USD",
    "bluzelle": "BLZ-USD",
    "everex": "EVX-USD",
    "ugChain": "UGC-USD",
    "ambrosus": "AMB-USD",
    "truefi": "TRU-USD",
    "covesting": "COV-USD",
    "coti": "COTI-USD",
    "sport-and-leisure": "SNL-USD",
    "aergo": "AERGO-USD",
    "orchid-protocol": "OXT-USD",
    "curve-fi": "CRV-USD",
    "sonm": "SNM-USD",
    "matic-network-matic-token": "MATIC-USD",
    "radicle": "RAD-USD",
    "numeraire": "NMR-USD",
    "republic-protocol": "REN-USD",
    "metronome": "MET-USD",
    "iot-chain": "ITC-USD",
    "streamr": "DATA-USD",
    "data": "DTA-USD",
    "trust": "TRUST-USD",
    "faireum": "FAIRC-USD",
    "fusion": "FSN-USD",
    "pundix": "NPXS-USD",
    "stpt": "STPT-USD",
    "singularitynet": "AGI-USD",
    "unibright": "UBT-USD",
    "golem": "GLM-USD",
    "travala": "AVA-USD",
    "hathor": "HTR-USD",
    "unibright": "UBT-USD",
    "tokencard": "TKN-USD",
    "winding-tree": "LIF-USD",
    "nash": "NEX-USD",
    "status": "SNT-USD",
    "funfair": "FUN-USD",
    "nucleus-vision": "NCASH-USD",
    "verge": "XVG-USD",
    "orchid": "OXT-USD",
    "elastos": "ELA-USD",
    "lisk": "LSK-USD",
    "appcoins": "APPC-USD",
    "blockport": "BPT-USD",
    "coinfi": "COFI-USD",
    "dether": "DTH-USD",
    "gulden": "NLG-USD",
    "adex": "ADX-USD",
    "cloakcoin": "CLOAK-USD",
    "zebi": "ZCO-USD",
    "hive": "HIVE-USD",
    "skrumble-network": "SKM-USD",
    "decentralized-machine-learning": "DML-USD",
    "apex": "CPX-USD",
    "vexanium": "VEX-USD",
    "rate3": "RTE-USD",
    "data": "DTA-USD",
    "neblio": "NEBL-USD",
    "status": "SNT-USD",
    "wepower": "WPR-USD",
    "bitcore": "BTX-USD",
    "thorchain": "RUNE-USD",
    "digitex-futures": "DGTX-USD",
    "odyssey": "OCN-USD",
    "genesis-vision": "GVT-USD",
    "aphelion": "APH-USD",
    "eos-dac": "EOSDAC-USD",
    "wax": "WAXP-USD",
    "siacoin": "SC-USD",
    "datum": "DAT-USD",
    "veil": "VEIL-USD",
    "viberate": "VIB-USD",
    "genesis-vision": "GVT-USD",
    "blue-protocol": "BLUE-USD",
    "swftcoin": "SWFTC-USD",
    "sentinel": "DVPN-USD",
    "bankex": "BKX-USD",
    "axon": "AXN-USD",
    "expanse": "EXP-USD",
    "digitalnote": "XDN-USD",
    "digitalcoin": "DGC-USD",
    "zclassic": "ZCL-USD",
    "ultra": "UOS-USD",
    "dlive": "DLIVE-USD",
    "jibrel-network": "JNT-USD",
    "whitecoin": "XWC-USD",
    "wings": "WINGS-USD",
    "substratum": "SUB-USD",
    "wepower": "WPR-USD",
    "zeepin": "ZPT-USD",
    "prizm": "PZM-USD",
    "travala.com": "AVA-USD",
    "republic-protocol": "REN-USD",
    "oneledger": "OLT-USD",
    "stakenet": "XSN-USD",
    "civic": "CVC-USD",
    "sonm": "SNM-USD",
    "naga": "NGC-USD",
    "feathercoin": "FTC-USD",
    "lto-network": "LTO-USD",
    "certik": "CTK-USD",
    "helium": "HNT-USD",
    "unifi-protocol-dao": "UNFI-USD",
    "edgeless": "EDG-USD",
    "bancor": "BNT-USD",
    "poa-network": "POA-USD",
    "dether": "DTH-USD",
    "bluzelle": "BLZ-USD",
    "apollo-currency": "APL-USD",
    "neblio": "NEBL-USD",
    "lto-network": "LTO-USD",
    "vechain": "VET-USD",
    "pivx": "PIVX-USD",
    "lykke": "LKK-USD",
    "bancor": "BNT-USD",
    "civic": "CVC-USD",
    "quant-network": "QNT-USD",
    "potcoin": "POT-USD",
    "bread": "BRD-USD",
    "xaurum": "XAUR-USD",
    "aragon": "ANT-USD",
    "pirl": "PIRL-USD",
    "stakenet": "XSN-USD",
    "bee": "BEE-USD",
    "sia": "SC-USD",
    "cryptonex": "CNX-USD",
    "tron": "TRX-USD",
    "aelf": "ELF-USD",
    "wagerr": "WGR-USD",
    "quantstamp": "QSP-USD",
    "district0x": "DNT-USD",
    "adex": "ADX-USD",
    "augur": "REP-USD",
    "xensor": "XSR-USD",
    "telcoin": "TEL-USD",
    "zap": "ZAP-USD",
    "mooncoin": "MOON-USD",
    "gamecredits": "GAME-USD",
    "aeron": "ARN-USD",
    "loki-network": "LOKI-USD",
    "storiqa": "STQ-USD",
    "coinfi": "COFI-USD",
    "bloom": "BLT-USD",
    "viberate": "VIB-USD",
    "blockport": "BPT-USD",
    "pundix": "NPXS-USD",
    "tokencard": "TKN-USD",
    "lykke": "LKK-USD",
    "counterparty": "XCP-USD",
    "metronome": "MET-USD",
    "0chain": "ZCN-USD",
    "solaris": "XLR-USD",
    "ion": "ION-USD",
    "aelf": "ELF-USD",
    "naga": "NGC-USD",
    "district0x": "DNT-USD",
    "virtacoin": "VTA-USD",
    "covalent": "CQT-USD",
    "cloakcoin": "CLOAK-USD",
    "sonm": "SNM-USD",
    "qash": "QASH-USD",
    "blockport": "BPT-USD",
    "district0x": "DNT-USD",
    "sonm": "SNM-USD",
    "winding-tree": "LIF-USD",
    "sonm": "SNM-USD",
    "telcoin": "TEL-USD",
    "bloom": "BLT-USD",
    "gulden": "NLG-USD",
    "pundix": "NPXS-USD",
    "sonm": "SNM-USD",
    "gulden": "NLG-USD"}
    crypto_data = yf.download(crypto_curr, period=period)
    # Extract volume data
    volume_data = crypto_data['Volume']
    return volume_data

#crypto_id = "BTC-USD"  # Example: Bitcoin
d="1mo"
volume_data = fetch_crypto_volume_data(crypto_curr,d)

# Plot volume graph
fig = go.Figure(data=[go.Scatter(x=volume_data.index, y=volume_data, mode='lines', name='Volume')])
fig.update_layout(title=f'Volume of {crypto_curr} over time', xaxis_title='Date', yaxis_title='Volume')
st.plotly_chart(fig)