from sklearn.manifold import TSNE
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlite3 import connect


def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
    cumdist = np.matrix(np.ones((an + 1, bn + 1)) * np.inf)
    cumdist[0, 0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi + 1],
                                   cumdist[ai + 1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai + 1, bi + 1] = pointwise_distance[ai, bi] + minimum_cost

    return cumdist[an, bn]


def plot_clusters(sessions, metric='cosine'):
    if metric == 'dtw':
        metric = DTW

    e = TSNE(n_components=2, metric=metric, square_distances=True).fit_transform(sessions)
    plt.scatter(e[:, 0], e[:, 1])
    plt.show()


def load_data():
    cnx = connect('trade_info.sqlite3')
    df = pd.read_sql_query("SELECT * FROM chart_data", cnx)
    df_session = pd.read_sql_query("SELECT * FROM trading_session", cnx)
    return df, df_session


def merge_data(df, df_session):
    for i, deal in df.iterrows():
        id = deal['session_id']
        date = df_session[df_session['id'] == id]['date']
        if len(date.values) > 0:
            df.at[i, 'full date'] = deal['time'] + ' ' + date.values[0]
            df.at[i, 'date'] = date.values[0]
            df.at[i, 'platform_id'] = df_session[df_session['id'] == id]['platform_id'].values[0]
            df.at[i, 'trading_type'] = df_session[df_session['id'] == id]['trading_type'].values[0]
        else:
            df.at[i, 'full date'] = np.nan


def get_metrics(df):
    dates = []
    prices = []
    lots = []
    sum_prices = []
    for t in np.unique(df['date']):
        cur_df = df[df['date'] == t]
        dates.append(t)
        prices.append(cur_df['price'].mean())
        lots.append(cur_df['lot_size'].mean())
        sum_prices.append((cur_df['price'] * cur_df['lot_size']).sum())
    return dates, prices, lots, sum_prices


def load_and_process_data():
    cnx = connect('trade_info.sqlite3')
    df = pd.read_sql_query("SELECT * FROM chart_data", cnx)
    df_session = pd.read_sql_query("SELECT * FROM trading_session", cnx)
    merge_data(df, df_session)
    df = df[df['trading_type'] == 'monthly']
    for d in np.unique(df['deal_id']):
        df_con = df[df['deal_id'] == d]
        df.drop(df[df['deal_id'] == d].index.drop(pd.to_datetime(df_con['time']).idxmin()), inplace=True)
    df = df.dropna(subset=['full date'])
    df['full date'] = pd.to_datetime(df['full date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['full date'])
    return df


def get_sessions_vectors(df):
    sessions = []
    start_price = 0

    for id in np.unique(df['session_id']):
        cur_df = df[df['session_id'] == id]
        price = cur_df['price']
        lot = cur_df['lot_size']
        cur_price = start_price
        chart = []
        for m in range(60):
            m_df = cur_df[cur_df['full date'].dt.minute == m]
            m_price = m_df['price']
            m_lot = m_df['lot_size']
            if len(m_df.values) == 0:
                chart += [cur_price]
            else:
                cur_price = (m_lot * m_price).sum() / m_lot.sum()
                chart += [cur_price]

        start_price = (lot * price).sum() / lot.sum()

        sessions.append(np.array(chart))
    sessions = np.array(sessions)
    sessions = (sessions - sessions.mean(axis=1, keepdims=True)) / sessions.std(axis=1, keepdims=True)
    return sessions


def plot_session(ax, x, y, title):
    ax.plot(np.arange(60), x)
    ax.plot(np.arange(60), y)
    ax.set_title(title)


def plot_vectors(sessions, metric):
    if metric == "dtw":
        metric = DTW

    e = TSNE(n_components=2, metric=metric, square_distances=True).fit_transform(sessions)
    dist = distance.cdist(e, e)
    max1, max2 = np.unravel_index(dist.argmax(), dist.shape)
    np.fill_diagonal(dist, 1e7)
    min1, min2 = np.unravel_index(dist.argmin(), dist.shape)
    f, ax = plt.subplots(1, 2, figsize=(15, 6))
    plot_session(ax[0], sessions[max1], sessions[max2], "Furthest")
    plot_session(ax[1], sessions[min1], sessions[min2], "Closest")
    plt.show()
