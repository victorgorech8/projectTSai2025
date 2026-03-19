import requests
import pandas as pd
import os

# Создаем папку для данных
os.makedirs('src/moex_data', exist_ok=True)

def get_all_stocks():
    """Получить список всех акций"""
    print("Получаем список акций...")
    url = "https://iss.moex.com/iss/securities.json"
    params = {'engine': 'stock', 'market': 'shares', 'is_trading': 1}
    all_stocks = []
    start = 0
    #MOEX выдает информацию порциями
    while True:
        params['start'] = start
        response = requests.get(url, params=params)
        data = response.json()
        
        stocks = data['securities']['data']
        if not stocks:
            break
            
        all_stocks.extend(stocks)
        start += 100
        if len(stocks) < 100:
            break
    
    df = pd.DataFrame(all_stocks, columns=data['securities']['columns'])
    stocks_df = df[df['type'].isin(['common_share', 'preferred_share'])]
    return stocks_df

def get_stock_data(secid):
    """Получить исторические данные по акции"""
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{secid}.json"
    
    all_data = []
    start = 0
    
    while True:
        params = {'from': '1990-01-01', 'till': '2099-12-31', 'interval': 24, 'start': start}
        response = requests.get(url, params=params)
        data = response.json()
        
        history_data = data['history']['data']
        if not history_data:
            break
            
        all_data.extend(history_data)
        start += 100
        if len(history_data) < 100:
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=data['history']['columns'])
        df['ticker'] = secid  # Добавляем тикер
        return df
    return None

def main():
    print("Начинаем выгрузку данных MOEX...")
    
    # Получаем список всех акций
    stocks_df = get_all_stocks()
    print(f"Найдено акций: {len(stocks_df)}")
    
    # Сохраняем список акций в parquet
    stocks_df.to_parquet('src/moex_data/stocks_list.parquet', index=False)
    
    # Собираем все данные в один DataFrame
    all_data = []
    successful=0
    # Выгружаем данные по каждой акции
    for i, row in stocks_df.iterrows():
        secid = row['secid']
        name = row.get('shortname', 'N/A')
        data = get_stock_data(secid)
        if data is not None:
            successful += 1
            print(f"[{successful}/{len(stocks_df)}]  {secid} - {name}")
            all_data.append(data)
        else:
            successful += 1
            print(f"[{i+1}/{len(stocks_df)}]  {secid} - {name} (нет данных)")
    
    if all_data:
        
        combined_df = pd.concat(all_data, ignore_index=True)
        # Сохраняем в parquet
        combined_df.to_parquet('src/moex_data/all_stocks.parquet', index=False)
        
        print(f"\nОбъединенный файл создан: {len(combined_df):,} строк")
        print(f"Тикеров: {combined_df['ticker'].nunique()}")
        print(f"Период: {combined_df['TRADEDATE'].min()} - {combined_df['TRADEDATE'].max()}")
    
    print("Выгрузка завершена! Данные в папке 'src/moex_data'")

if __name__ == "__main__":
    main()

