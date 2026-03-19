import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Статистические тесты
from statsmodels.tsa.stattools import adfuller, kpss

# Преобразования
from scipy import stats

# Prophet (Facebook) для анализа временных рядов
from prophet import Prophet

# Визуализация
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# PyTorch
import torch

# Qiskit (квантовые вычисления)
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQR
from qiskit.primitives import StatevectorEstimator

import os
os.makedirs('results', exist_ok=True)

# Загрузка исходных данных
df = pd.read_parquet('src/moex_data/all_stocks.parquet')

# Выбор нужных колонок
long_df = df[['TRADEDATE', 'SECID', 'VALUE']].copy()
long_df['TRADEDATE'] = pd.to_datetime(long_df['TRADEDATE'])
long_df = long_df.sort_values('TRADEDATE')

# Фильтрация данных за последние 2 года
last_date = long_df['TRADEDATE'].max()
two_years_ago = last_date - timedelta(days=365*2)

# Оставляем тикеры с данными за последние 2 года
valid_tickers = long_df.groupby('SECID')['TRADEDATE'].max()[long_df.groupby('SECID')['TRADEDATE'].max() >= two_years_ago].index.tolist()
long_df_filtered = long_df[long_df['SECID'].isin(valid_tickers)]
long_df_final = long_df_filtered[long_df_filtered['TRADEDATE'] >= two_years_ago].copy()

# Оставляем только тикеры с ровно 509 днями данных
ticker_stats = long_df_final.groupby('SECID').size().reset_index(name='count')
valid_tickers_509 = ticker_stats[ticker_stats['count'] == 509]['SECID'].tolist()
long_df_final_filtered = long_df_final[long_df_final['SECID'].isin(valid_tickers_509)].copy()

print(f"Всего записей: {len(long_df_final_filtered)}")
print(f"Тикеров: {long_df_final_filtered['SECID'].nunique()}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
stats = long_df_final.groupby('SECID').size().reset_index(name='count')
stats['count'].hist(bins=30)
plt.title('Распределение количества наблюдений по тикерам')
plt.xlabel('Количество наблюдений')
plt.ylabel('Количество тикеров')
plt.subplot(1, 2, 2)
stats['count'].sort_values(ascending=False).reset_index(drop=True).plot()
plt.title('Ранжированный ряд количества наблюдений')
plt.xlabel('Тикеры (отсортированы)')
plt.ylabel('Количество наблюдений')
plt.tight_layout()
plt.savefig('results/distribution_plot.png')
plt.show()

# Удаляем тикеры с одним уникальным значением
unique_counts = long_df_final_filtered.groupby('SECID')['VALUE'].nunique()
tickers_to_remove = unique_counts[unique_counts == 1].index.tolist()
long_df_cleaned = long_df_final_filtered[~long_df_final_filtered['SECID'].isin(tickers_to_remove)]

print(f"Удалено тикеров с const значением: {len(tickers_to_remove)}")
print(f"Осталось тикеров: {long_df_cleaned['SECID'].nunique()}")

print(long_df_final)
ticker_stats = long_df_final.groupby('SECID').size().reset_index(name='count')
valid_tickers = ticker_stats[ticker_stats['count'] == 509]['SECID'].tolist()
long_df_final_filtered = long_df_final[long_df_final['SECID'].isin(valid_tickers)].copy()

# Функция для проверки одного ряда
def check_series_stationarity(series):
    adf_result = adfuller(series, autolag='AIC')
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_stationary = adf_pvalue < 0.05
    
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_stationary = kpss_pvalue >= 0.05
    
    return {
        'adf_statistic': round(adf_stat, 4),
        'adf_pvalue': round(adf_pvalue, 4),
        'adf_stationary': adf_stationary,
        'kpss_statistic': round(kpss_stat, 4),
        'kpss_pvalue': round(kpss_pvalue, 4),
        'kpss_stationary': kpss_stationary
    }

results = {}
tickers = long_df_cleaned['SECID'].unique()

for i, ticker in enumerate(tickers):
    series = long_df_cleaned[long_df_cleaned['SECID'] == ticker]['VALUE'].values
    results[ticker] = check_series_stationarity(series)

print(f"  Обработано {len(tickers)}/{len(tickers)} тикеров")

result_df = pd.DataFrame(results).T
result_df.index.name = 'ticker'

def get_conclusion(row):
    if row['adf_stationary'] and row['kpss_stationary']:
        return 'Стационарный'
    elif not row['adf_stationary'] and not row['kpss_stationary']:
        return 'Нестационарный'
    else:
        return 'Противоречивые результаты'

result_df['conclusion'] = result_df.apply(get_conclusion, axis=1)

print(result_df)
print("\nСТАТИСТИКА:")
print(result_df['conclusion'].value_counts())

result_df.to_csv('results/stationarity_results.csv')

# Разделяем
stationary_tickers = result_df[result_df['conclusion'] != 'Нестационарный'].index.tolist()
non_stationary_tickers = result_df[result_df['conclusion'] == 'Нестационарный'].index.tolist()

stationary_df = long_df_cleaned[long_df_cleaned['SECID'].isin(stationary_tickers)]
non_stationary_df = long_df_cleaned[long_df_cleaned['SECID'].isin(non_stationary_tickers)]

stationary_df.to_csv('results/stationary_df.csv', index=False)
non_stationary_df.to_csv('results/non_stationary_df.csv', index=False)

from scipy import stats
non_stationary_boxcox = non_stationary_df.copy()

def boxcox_with_shift(x):
    min_val = x.min()
    if min_val <= 0:
        x = x + abs(min_val) + 1
    return stats.boxcox(x)[0]

non_stationary_boxcox['VALUE'] = non_stationary_boxcox.groupby('SECID')['VALUE'].transform(boxcox_with_shift)
non_stationary_boxcox.to_csv('results/non_stationary_boxcox.csv', index=False)

print("СТАЦИОНАРНЫЙ ФРЕЙМ:")
print(f"Тикеров: {stationary_df['SECID'].nunique()}")
print(f"Записей: {len(stationary_df)}")

print("\nНЕСТАЦИОНАРНЫЙ ФРЕЙМ (Box-Cox):")
print(f"Тикеров: {non_stationary_boxcox['SECID'].nunique()}")
print(f"Записей: {len(non_stationary_boxcox)}")

window = 5

stationary_smooth = stationary_df.copy()
stationary_smooth = stationary_smooth.sort_values(['SECID', 'TRADEDATE'])
stationary_smooth['VALUE_smooth'] = stationary_smooth.groupby('SECID')['VALUE'].transform(lambda x: x.rolling(window, min_periods=1).mean())

non_stationary_smooth = non_stationary_boxcox.copy()
non_stationary_smooth = non_stationary_smooth.sort_values(['SECID', 'TRADEDATE'])
non_stationary_smooth['VALUE_smooth'] = non_stationary_smooth.groupby('SECID')['VALUE'].transform(lambda x: x.rolling(window, min_periods=1).mean())

print(f"\nСГЛАЖЕННЫЕ ВЕРСИИ (окно={window}):")
print(f"Стационарный сглаженный: {len(stationary_smooth)} записей")
print(f"Нестационарный сглаженный: {len(non_stationary_smooth)} записей")

stationary_smooth.to_csv('results/stationary_smooth.csv', index=False)
non_stationary_smooth.to_csv('results/non_stationary_smooth.csv', index=False)

from prophet import Prophet
results_prophet = []

for ticker in stationary_df['SECID'].unique():
    print(f"Анализ: {ticker}")
    
    df_prophet = stationary_df[stationary_df['SECID'] == ticker][['TRADEDATE', 'VALUE']].copy()
    df_prophet.columns = ['ds', 'y']
    
    best_score = -np.inf
    best_params = {}
    best_components = None
    
    for growth in ['linear', 'logistic']:
        for seasonality_mode in ['additive', 'multiplicative']:
            for yearly_seasonality in [True, False]:
                for weekly_seasonality in [True, False]:
                    
                    try:
                        if growth == 'logistic':
                            df_prophet['cap'] = df_prophet['y'].max() * 1.2
                            df_prophet['floor'] = 0
                        
                        model = Prophet(
                            growth=growth,
                            seasonality_mode=seasonality_mode,
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=False,
                            interval_width=0.95
                        )
                        
                        model.fit(df_prophet)
                        
                        future = model.make_future_dataframe(periods=0)
                        if growth == 'logistic':
                            future['cap'] = df_prophet['y'].max() * 1.2
                            future['floor'] = 0
                        
                        forecast = model.predict(future)
                        
                        fitted = forecast['yhat'].values[:len(df_prophet)]
                        resid = df_prophet['y'].values - fitted
                        resid_var = np.var(resid)
                        
                        score = 1 - resid_var / np.var(df_prophet['y']) - 0.1 * np.log(resid_var + 1)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'ticker': ticker,
                                'growth': growth,
                                'seasonality_mode': seasonality_mode,
                                'yearly': yearly_seasonality,
                                'weekly': weekly_seasonality,
                                'resid_var': resid_var,
                                'score': score
                            }
                            
                            best_components = {
                                'trend': forecast['trend'].values[:len(df_prophet)],
                                'seasonal': forecast['yearly'].values[:len(df_prophet)] + forecast['weekly'].values[:len(df_prophet)] if yearly_seasonality or weekly_seasonality else np.zeros(len(df_prophet)),
                                'resid': resid
                            }
                    except:
                        continue
    
    if best_params:
        results_prophet.append({
            **best_params,
            'trend': best_components['trend'],
            'seasonal': best_components['seasonal'],
            'resid': best_components['resid']
        })

summary_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['trend', 'seasonal', 'resid']}
    for r in results_prophet
])

print("Типы трендов:")
print(summary_df['growth'].value_counts())
print("\nТипы сезонности:")
print(summary_df['seasonality_mode'].value_counts())
print("\nНаличие сезонности:")
print(f"Годовая: {summary_df['yearly'].sum()}")
print(f"Недельная: {summary_df['weekly'].sum()}")

summary_df.to_csv('results/prophet_summary.csv', index=False)

from tsururu.dataset import Pipeline, TSDataset
from tsururu.strategies import RecursiveStrategy, DirectStrategy, MIMOStrategy
from tsururu.model_training import MLTrainer
from tsururu.model_training import KFoldCrossValidator
from tsururu.models.boost import CatBoost

long_df_sequential = long_df_cleaned.copy()
long_df_sequential = long_df_sequential.sort_values(['SECID', 'TRADEDATE'])
long_df_sequential['DAY_NUM'] = long_df_sequential.groupby('SECID').cumcount() + 1
df_sequential = long_df_sequential[['SECID', 'DAY_NUM', 'VALUE']].copy()
df_sequential = df_sequential.rename(columns={'DAY_NUM': 'TRADEDATE'})

print(f"Всего записей: {len(df_sequential)}")
print(f"Тикеров: {df_sequential['SECID'].nunique()}")

dataset_params = {
    "target": {
        "columns": ["VALUE"],
        "type": "continuous",
    },
    "date": {
        "columns": ["TRADEDATE"],
        "type": "datetime",
    },
    "id": {
        "columns": ["SECID"],
        "type": "categorical",
    }
}

dataset = TSDataset(
    data=df_sequential,
    columns_params=dataset_params,
)

pipeline_easy_params = {
    "target_lags": 5,
    "date_lags": 1,
    "target_normalizer": "standard_scaler",
    "target_normalizer_regime": "none",
}
pipeline1 = Pipeline.easy_setup(
    dataset_params, 
    pipeline_easy_params, 
    multivariate=False
)

model1 = CatBoost
model_params1 = {
    "loss_function": "MultiRMSE",
    "early_stopping_rounds": 100,
    "verbose": 500,
}

validation1 = KFoldCrossValidator
validation_params1 = {
    "n_splits": 2,
}

trainer1 = MLTrainer(
    model1,
    model_params1,
    validation1,
    validation_params1,
)

strategies1 = {
    'recursive': RecursiveStrategy(
        horizon=5, 
        history=10, 
        trainer=trainer1, 
        pipeline=pipeline1
    ),
    'direct': DirectStrategy(
        horizon=5, 
        history=10, 
        trainer=trainer1, 
        pipeline=pipeline1
    ),
    'mimo': MIMOStrategy(
        horizon=5, 
        history=10, 
        trainer=trainer1, 
        pipeline=pipeline1
    )
}

predictions_ml = {}
for name, strategy in strategies1.items():
    print(f"Стратегия: {name}")
    
    try:
        fit_time, _ = strategy.fit(dataset)
        print(f"Время обучения: {fit_time:.2f} сек")
        
        forecast_time, predictions = strategy.predict(dataset)
        print(f"Время прогноза: {forecast_time:.2f} сек")
        
        if predictions is not None:
            tickers_list = df_sequential['SECID'].unique()
            horizon = 5
            ids = []
            for ticker in tickers_list:
                ids.extend([ticker] * horizon)
            
            if len(predictions) == len(tickers_list) * horizon:
                predictions['SECID'] = ids[:len(predictions)]
            
            predictions_ml[name] = predictions
            predictions.to_csv(f'results/predictions_{name}.csv', index=False)
            print(f"Предсказания сохранены")
            
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

from tsururu.dataset import Pipeline, TSDataset
from tsururu.strategies import MIMOStrategy
from tsururu.model_training import DLTrainer
from tsururu.model_training import KFoldCrossValidator
from tsururu.model_training.torch_based.metrics import NegativeMSEMetric
from tsururu.models import DLinear_NN, PatchTST_NN, TimesNet_NN, CycleNet_NN
from tsururu.transformers import (
    DateSeasonsGenerator,
    LagTransformer,
    SequentialTransformer,
    TargetGenerator,
    UnionTransformer,
)

def negative_mse(pred, target):
    return -torch.nn.functional.mse_loss(pred, target)

long_df_sequential = long_df_cleaned.copy()
long_df_sequential = long_df_sequential.sort_values(['SECID', 'TRADEDATE'])
long_df_sequential['DAY_NUM'] = long_df_sequential.groupby('SECID').cumcount() + 1
df_sequential = long_df_sequential[['SECID', 'DAY_NUM', 'VALUE']].copy()
df_sequential = df_sequential.rename(columns={'DAY_NUM': 'TRADEDATE', 'SECID': 'id', 'VALUE': 'value'})

print(f"Всего записей: {len(df_sequential)}")
print(f"Тикеров: {df_sequential['id'].nunique()}")

dataset_params = {
    "target": {
        "columns": ["value"],
        "type": "continuous",
    },
    "date": {
        "columns": ["TRADEDATE"],
        "type": "datetime",
    },
    "id": {
        "columns": ["id"],
        "type": "categorical",
    }
}

dataset = TSDataset(
    data=df_sequential,
    columns_params=dataset_params,
)

history = 7
horizon = 7

lag = LagTransformer(lags=history)
target_generator = TargetGenerator()
date_features = DateSeasonsGenerator()
date_lags = LagTransformer(lags=7)

union_1 = UnionTransformer(transformers_list=[lag, target_generator])
seq_1 = SequentialTransformer(transformers_list=[union_1], input_features=["value"])
seq_2 = SequentialTransformer(transformers_list=[date_features, date_lags], input_features=["TRADEDATE"])
union = UnionTransformer(transformers_list=[seq_1, seq_2])

pipeline = Pipeline(union, multivariate=False)

model = DLinear_NN
model_params = {"moving_avg": 7, "individual": False}

validation = KFoldCrossValidator
validation_params = {"n_splits": 2}

trainer_params = {
    "device": "cpu",
    "num_workers": 0,
    "best_by_metric": True,
    "save_to_dir": False,
    "n_epochs": 5,
    "verbose": True,
    "metric": negative_mse,
    "optimizer_params": {"lr": 0.001}
}

trainer = DLTrainer(
    model,
    model_params,
    validation,
    validation_params,
    **trainer_params
)

strategy = MIMOStrategy(
    pipeline=pipeline,
    trainer=trainer,
    horizon=horizon,
    history=history,
)

print("Начинаем обучение...")
try:
    fit_time, _ = strategy.fit(dataset)
    print(f"Время обучения: {fit_time:.2f} сек")
    
    forecast_time, predictions = strategy.predict(dataset)
    print(f"Время прогноза: {forecast_time:.2f} сек")
    
    if predictions is not None:
        tickers = df_sequential['id'].unique()
        n_tickers = len(tickers)
        
        if len(predictions) == n_tickers * horizon:
            ids = []
            for ticker in tickers:
                ids.extend([ticker] * horizon)
            predictions['id'] = ids
            predictions.to_csv('results/predictions_dlinear.csv', index=False)
            print("Предсказания DLinear сохранены")
            print(predictions.head())
        
except Exception as e:
    print(f"ОШИБКА: {e}")
    import traceback
    traceback.print_exc()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQR
from qiskit.primitives import StatevectorEstimator

def create_features(data, lookback=3):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

lookback = 3
all_tickers = long_df_cleaned['SECID'].unique()
print(f"Всего тикеров: {len(all_tickers)}")

results_quantum = {}

for idx, ticker in enumerate(all_tickers):
    print(f"\n{'='*50}")
    print(f"Тикер {idx+1}/{len(all_tickers)}: {ticker}")
    
    try:
        data = long_df_cleaned[long_df_cleaned['SECID'] == ticker]['VALUE'].values
        
        if len(data) < 20:
            print(f"Пропущен (мало данных)")
            continue
        
        X, y = create_features(data, lookback)
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        feature_map = ZZFeatureMap(feature_dimension=lookback, reps=1)
        ansatz = RealAmplitudes(num_qubits=lookback, reps=1)
        estimator = StatevectorEstimator()
        
        vqr = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=50),
            estimator=estimator,
        )
        
        vqr.fit(X_train[:20], y_train[:20])
        
        y_pred = vqr.predict(X_test[:5])
        
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_orig = scaler_y.inverse_transform(y_test[:5].reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(np.mean((y_test_orig - y_pred_orig)**2))
        
        results_quantum[ticker] = {
            'rmse': rmse,
            'predictions': y_pred_orig.tolist(),
            'actual': y_test_orig.tolist()
        }
        
        print(f"RMSE: {rmse:.2f}")
        print(f"Прогноз: {y_pred_orig}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        results_quantum[ticker] = {'error': str(e)}

print("\n" + "="*50)
print("ИТОГИ")

success = [t for t, r in results_quantum.items() if 'rmse' in r]
print(f"Успешно: {len(success)} из {len(all_tickers)}")

results_quantum_df = pd.DataFrame([
    {'ticker': t, 'rmse': r.get('rmse', None)} 
    for t, r in results_quantum.items() if 'rmse' in r
])
print("\nТоп-10 по RMSE:")
print(results_quantum_df.nsmallest(10, 'rmse').to_string(index=False))

results_quantum_df.to_csv('results/quantum_results.csv', index=False)

# Сохраняем основные датафреймы
long_df_cleaned.to_csv('results/long_df_cleaned.csv', index=False)
result_df.to_csv('results/stationarity_detailed.csv')
df_sequential.to_csv('results/df_sequential.csv', index=False)

print("\nВсе результаты сохранены в папку results/")