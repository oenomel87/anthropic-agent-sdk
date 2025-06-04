from typing import List
import pandas as pd
import numpy as np
import math
from datetime import datetime

from app.tools.upbit import get_current_ticker, get_candles_for_daily, get_candles_for_minutes, get_candles_for_weekly
from app.schemas.ticker import Ticker
from app.schemas.candle import MinuteCandleStick, DailyCandleStick, WeeklyCandleStick

async def get_market_info() -> dict:
    """
    비트코인의 시장 정보(market_info)를 구하는 함수
    market_info는 비트코인 가격, 변동성, 거래량 변화율 등을 포함합니다.
    market_info 데이터
    - symbol: 비트코인 심볼 (KRW-BTC)
    - current_price: 현재 비트코인 가격
    - day_change_pct: 24시간 변동률
    - timestamp: 현재 시간 (UTC)
    - 24h_volume: 24시간 거래량
    - 24h_volume_change_pct: 24시간 거래량 변화율

    Returns:
        market_info: 비트코인 시장 정보
    """
    # 1. 현재 티커 정보 가져오기
    current_ticker: Ticker = await get_current_ticker()
    
    # 2. 일별 캔들스틱 데이터 가져오기 (최소 30일 - 변동성 계산용)
    daily_candles: List[DailyCandleStick] = await get_candles_for_daily(count=30)
    
    # 3. 분 단위 캔들스틱 데이터 가져오기 (48개 30분 캔들 = 24시간)
    minute_candles: List[MinuteCandleStick] = await get_candles_for_minutes(minutes=30, count=48)
    
    # 4. 24시간 거래량 변화율 계산
    volume_change_pct = calculate_volume_change(minute_candles)
    
    # 5. 30일 변동성 계산
    volatility = calculate_daily_volatility(daily_candles)
    
    # 6. market_info 객체 구성
    market_info = {
        "symbol": "BTC-KRW",  # 변환된 심볼
        "current_price": current_ticker.trade_price,
        "day_change_pct": round(current_ticker.signed_change_rate * 100, 2),
        "timestamp": datetime.fromtimestamp(current_ticker.timestamp / 1000).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "24h_volume": round(current_ticker.acc_trade_volume_24h, 2),
        "24h_volume_change_pct": volume_change_pct,
        "volatility_30d_annualized": volatility
    }
    
    return market_info

def calculate_volume_change(minute_candles: List[MinuteCandleStick]) -> float:
    """
    24시간 거래량 변화율 계산
    """
    # 최근 24개의 캔들(12시간)과 이전 24개의 캔들(12시간) 비교
    recent_volume = sum(candle.candle_acc_trade_volume for candle in minute_candles[:24])
    previous_volume = sum(candle.candle_acc_trade_volume for candle in minute_candles[24:])
    
    # 변화율 계산
    if previous_volume > 0:
        volume_change_pct = ((recent_volume - previous_volume) / previous_volume) * 100
        return round(volume_change_pct, 2)
    return 0.0

def calculate_daily_volatility(daily_candles: List[DailyCandleStick]) -> float:
    """
    30일 연간화 변동성 계산
    """
    # 일별 수익률 계산
    returns = []
    for i in range(1, len(daily_candles)):
        today_price = daily_candles[i-1].trade_price
        yesterday_price = daily_candles[i].trade_price
        daily_return = math.log(today_price / yesterday_price)
        returns.append(daily_return)
    
    # 표준편차 계산
    if returns:
        std_dev = np.std(returns)
        # 연간화 (252 거래일 기준)
        annualized_volatility = std_dev * math.sqrt(252) * 100
        return round(annualized_volatility, 1)
    return 0.0

async def get_trend_analysis() -> dict:
    """
    비트코인의 추세 분석(trend_analysis) 정보를 구하는 함수
    trend_analysis는 단기, 중기, 장기 추세 분석 결과를 포함합니다.
    trend_analysis 데이터
    - short_term: 단기 추세 (1h-4h)
    - medium_term: 중기 추세 (1d-1w)
    - long_term: 장기 추세 (1w-1m)
    - trend_strength: 추세 강도 (0-100)
    - trend_duration_days: 현재 추세 지속 기간 (일수)
    """
    # 1. 시간별 캔들스틱 데이터 가져오기 (단기 추세 분석용)
    hourly_candles = await get_candles_for_minutes(minutes=60, count=60)
    
    # 2. 일별 캔들스틱 데이터 가져오기 (중기 추세 분석용)
    daily_candles = await get_candles_for_daily(count=14)
    
    # 3. 주별 캔들스틱 데이터 가져오기 (장기 추세 분석용)
    weekly_candles = await get_candles_for_weekly(count=8)
    
    # 4. 단기 추세 분석 (1-4시간)
    short_term_trend = analyze_short_term_trend(hourly_candles[:4])
    
    # 5. 중기 추세 분석 (1일-1주)
    medium_term_trend = analyze_medium_term_trend(daily_candles[:7])
    
    # 6. 장기 추세 분석 (1주-1개월)
    long_term_trend = analyze_long_term_trend(weekly_candles)
    
    # 7. 추세 강도 계산
    trend_strength = calculate_trend_strength(short_term_trend, medium_term_trend, long_term_trend)
    
    # 8. 추세 지속 기간 계산
    trend_duration = calculate_trend_duration(daily_candles)
    
    # 9. trend_analysis 객체 구성
    trend_analysis = {
        "short_term": short_term_trend["trend"],  # 단기 추세 (1h-4h)
        "medium_term": medium_term_trend["trend"],  # 중기 추세 (1d-1w)
        "long_term": long_term_trend["trend"],  # 장기 추세 (1w-1m)
        "trend_strength": round(trend_strength),  # 추세 강도 (0-100)
        "trend_duration_days": trend_duration  # 현재 추세 지속 기간
    }
    
    return trend_analysis

def analyze_short_term_trend(hourly_candles: List[MinuteCandleStick]) -> dict:
    """
    단기 추세 분석 (1-4시간)
    """
    # 상승/하락 캔들 수 계산
    up_count = 0
    down_count = 0
    
    for candle in hourly_candles:
        if candle.trade_price > candle.opening_price:
            up_count += 1
        elif candle.trade_price < candle.opening_price:
            down_count += 1
    
    # 시작 가격과 종료 가격의 차이 계산
    start_price = hourly_candles[-1].opening_price
    end_price = hourly_candles[0].trade_price
    price_change = ((end_price - start_price) / start_price) * 100
    
    # 추세 결정
    trend = "neutral"
    strength = 50 - abs(price_change * 10)
    
    if price_change > 1.0 or (up_count >= 3 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 15 + up_count * 10)
    elif price_change < -1.0 or (down_count >= 3 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 15 + down_count * 10)
    
    return {"trend": trend, "strength": strength, "price_change": price_change}

def analyze_medium_term_trend(daily_candles: List[DailyCandleStick]) -> dict:
    """
    중기 추세 분석 (1일-1주)
    """
    # 상승/하락일 계산
    up_days = 0
    down_days = 0
    
    for candle in daily_candles:
        if candle.change_rate > 0:
            up_days += 1
        elif candle.change_rate < 0:
            down_days += 1
    
    # 시작 가격과 종료 가격의 차이 계산
    start_price = daily_candles[-1].trade_price
    end_price = daily_candles[0].trade_price
    price_change = ((end_price - start_price) / start_price) * 100
    
    # 추세 결정
    trend = "neutral"
    strength = 50 - abs(price_change * 3)
    
    if price_change > 4 or (up_days >= 4 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 5 + up_days * 5)
    elif price_change < -4 or (down_days >= 4 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 5 + down_days * 5)
    
    return {"trend": trend, "strength": strength, "price_change": price_change}

def analyze_long_term_trend(weekly_candles: List[WeeklyCandleStick]) -> dict:
    """
    장기 추세 분석 (1주-1개월)
    """
    # 상승/하락 주 계산
    up_weeks = 0
    down_weeks = 0
    
    for i in range(len(weekly_candles) - 1):
        current_price = weekly_candles[i].trade_price
        prev_price = weekly_candles[i + 1].trade_price
        
        if current_price > prev_price:
            up_weeks += 1
        elif current_price < prev_price:
            down_weeks += 1
    
    # 시작 가격과 종료 가격의 차이 계산
    start_price = weekly_candles[-1].trade_price
    end_price = weekly_candles[0].trade_price
    price_change = ((end_price - start_price) / start_price) * 100
    
    # 추세 결정
    trend = "neutral"
    strength = 50 - abs(price_change)
    
    if price_change > 8 or (up_weeks >= 3 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 2 + up_weeks * 10)
    elif price_change < -8 or (down_weeks >= 3 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 2 + down_weeks * 10)
    
    return {"trend": trend, "strength": strength, "price_change": price_change}

def calculate_trend_strength(short_term, medium_term, long_term) -> float:
    """
    추세 강도 계산 - 단기, 중기, 장기 추세의 가중 평균
    """
    # 가중 평균 계산
    weighted_strength = (
        (short_term["strength"] * 0.2) + 
        (medium_term["strength"] * 0.3) + 
        (long_term["strength"] * 0.5)
    )
    
    # 추세 일치 보너스
    if (short_term["trend"] == medium_term["trend"] and 
        medium_term["trend"] == long_term["trend"]):
        return min(100, weighted_strength + 15)
    
    # 중장기 추세 일치 보너스
    if medium_term["trend"] == long_term["trend"]:
        return min(100, weighted_strength + 8)
    
    return weighted_strength

def calculate_trend_duration(daily_candles: List[DailyCandleStick]) -> int:
    """
    현재 추세가 지속된 일수 계산
    """
    # 최근 추세 방향 결정
    current_direction = None
    
    # 최근 2일 종가 비교로 추세 방향 결정
    if daily_candles[0].trade_price > daily_candles[1].trade_price:
        current_direction = "up"
    elif daily_candles[0].trade_price < daily_candles[1].trade_price:
        current_direction = "down"
    else:
        # 같은 가격일 경우 이전 추세 확인
        for i in range(2, len(daily_candles)):
            if daily_candles[1].trade_price != daily_candles[i].trade_price:
                current_direction = "up" if daily_candles[1].trade_price > daily_candles[i].trade_price else "down"
                break
    
    # 추세 지속 일수 계산
    duration_days = 1  # 최소 하루부터 시작
    
    if current_direction:
        for i in range(1, len(daily_candles) - 1):
            comparison = False
            if current_direction == "up":
                comparison = daily_candles[i].trade_price > daily_candles[i + 1].trade_price
            else:
                comparison = daily_candles[i].trade_price < daily_candles[i + 1].trade_price
            
            if comparison:
                duration_days += 1
            else:
                break
    
    return duration_days

async def get_price_levels() -> dict:
    """
    비트코인의 가격 레벨(price_levels) 정보를 구하는 함수
    price_levels는 주요 지지선과 저항선, 최근 테스트된 레벨, 저항선과 지지선까지의 거리 등을 포함합니다.
    price_levels 데이터
    - key_resistance: 주요 저항선 (상위 2개)
    - key_support: 주요 지지선 (상위 2개)
    - last_tested: 최근 테스트된 레벨 (지지선/저항선)
    - distance_to_resistance_pct: 저항선까지의 거리 (퍼센트)
    - distance_to_support_pct: 지지선까지의 거리 (퍼센트)
    """
    # 1. 현재 티커 정보 가져오기
    current_ticker = await get_current_ticker()
    current_price = current_ticker.trade_price
    
    # 2. 일별 캔들스틱 데이터 가져오기 (분석용)
    daily_candles = await get_candles_for_daily(count=30)
    
    # 3. 가격 클러스터 식별
    price_clusters = identify_price_clusters(daily_candles)
    
    # 4. 주요 지지선과 저항선 결정
    support_resistance = determine_support_and_resistance(price_clusters, current_price)
    
    # 5. 최근 테스트된 레벨 확인
    last_tested = identify_last_tested_level(daily_candles, 
                                             support_resistance["support"], 
                                             support_resistance["resistance"])
    
    # 6. 저항선과 지지선까지의 거리 계산
    distance_to_resistance = calculate_distance_to_level(
        current_price, support_resistance["resistance"][0])
    distance_to_support = calculate_distance_to_level(
        support_resistance["support"][0], current_price)
    
    # 7. price_levels 객체 구성
    price_levels = {
        "key_resistance": support_resistance["resistance"],
        "key_support": support_resistance["support"],
        "last_tested": last_tested,
        "distance_to_resistance_pct": round(distance_to_resistance, 2),
        "distance_to_support_pct": round(distance_to_support, 2)
    }
    
    return price_levels

def identify_price_clusters(candles: List[DailyCandleStick]) -> List[dict]:
    """
    가격 클러스터 식별 - 중요한 가격 구간 찾기
    """
    # 고점과 저점 데이터 수집
    high_prices = [candle.high_price for candle in candles]
    low_prices = [candle.low_price for candle in candles]
    all_prices = high_prices + low_prices
    
    # 가격 범위 정의 (최대 0.5% 편차)
    price_tolerance = 0.005
    clusters = []
    
    # 처리된 가격을 추적하기 위한 집합
    processed_prices = set()
    
    # 각 가격에 대해 클러스터링
    for i, price in enumerate(all_prices):
        # 이미 처리된 가격이면 건너뜀
        if price in processed_prices:
            continue
        
        # 이 가격을 중심으로 하는 클러스터 찾기
        cluster = {
            "central_price": price,
            "prices": [price],
            "touch_count": 1,
            "type": "resistance" if i < len(high_prices) else "support"
        }
        
        # 비슷한 가격 범위에 있는 다른 가격 찾기
        for j in range(i + 1, len(all_prices)):
            other_price = all_prices[j]
            deviation = abs(price - other_price) / price
            
            if deviation <= price_tolerance:
                cluster["prices"].append(other_price)
                cluster["touch_count"] += 1
                processed_prices.add(other_price)
        
        # 클러스터 중심 가격을 평균으로 업데이트
        if len(cluster["prices"]) > 1:
            cluster["central_price"] = sum(cluster["prices"]) / len(cluster["prices"])
        
        # 클러스터 유형 결정 (고점이 더 많으면 저항선, 저점이 더 많으면 지지선)
        high_count = sum(1 for p in cluster["prices"] if p in high_prices)
        low_count = sum(1 for p in cluster["prices"] if p in low_prices)
        
        if high_count > low_count:
            cluster["type"] = "resistance"
        else:
            cluster["type"] = "support"
        
        # 클러스터 강도 계산
        cluster["strength"] = calculate_cluster_strength(cluster, candles)
        
        # 클러스터 추가
        clusters.append(cluster)
        
        # 이 가격도 처리된 것으로 표시
        processed_prices.add(price)
    
    # 강도 기준 내림차순 정렬
    return sorted(clusters, key=lambda x: x["strength"], reverse=True)

def calculate_cluster_strength(cluster: dict, candles: List[DailyCandleStick]) -> float:
    """
    클러스터 강도 계산
    """
    strength = cluster["touch_count"]
    
    # 최근성에 기반한 추가 강도 (최근 캔들일수록 더 중요)
    for i, candle in enumerate(candles):
        recency_factor = max(1, 5 - i * 0.5)  # 최근 캔들은 더 높은 가중치
        
        # 이 캔들이 클러스터에 터치했는지 확인
        if cluster["type"] == "resistance":
            deviation = abs(cluster["central_price"] - candle.high_price) / cluster["central_price"]
            if deviation <= 0.005:
                strength += recency_factor
        else:
            deviation = abs(cluster["central_price"] - candle.low_price) / cluster["central_price"]
            if deviation <= 0.005:
                strength += recency_factor
    
    # 최근 추세와 일치하면 추가 강도 부여
    recent_trend = "up" if candles[0].trade_price > candles[4].trade_price else "down"
    if (recent_trend == "up" and cluster["type"] == "resistance") or \
       (recent_trend == "down" and cluster["type"] == "support"):
        strength += 2
    
    return strength

def determine_support_and_resistance(clusters, current_price: float) -> dict:
    """
    주요 지지선과 저항선 결정
    """
    # 저항선 (현재 가격보다 높은 클러스터)
    resistance_clusters = sorted(
        [c for c in clusters if c["central_price"] > current_price and c["type"] == "resistance"],
        key=lambda x: x["central_price"]
    )
    
    # 지지선 (현재 가격보다 낮은 클러스터)
    support_clusters = sorted(
        [c for c in clusters if c["central_price"] < current_price and c["type"] == "support"],
        key=lambda x: x["central_price"],
        reverse=True
    )
    
    # 상위 2개의 저항선과 지지선 선택
    resistance_levels = [round(c["central_price"]) for c in resistance_clusters[:2]]
    support_levels = [round(c["central_price"]) for c in support_clusters[:2]]
    
    # 저항선이 충분하지 않은 경우, 기술적 레벨 추가
    if len(resistance_levels) < 2:
        if resistance_levels:
            # 상위 저항선이 있으면 그 위에 1% 추가
            resistance_levels.append(round(resistance_levels[0] * 1.01))
        else:
            # 저항선이 하나도 없으면 현재 가격에서 상승
            resistance_levels.append(round(current_price * 1.01))
            resistance_levels.append(round(current_price * 1.02))
    
    # 지지선이 충분하지 않은 경우, 기술적 레벨 추가
    if len(support_levels) < 2:
        if support_levels:
            # 상위 지지선이 있으면 그 아래에 1% 추가
            support_levels.append(round(support_levels[0] * 0.99))
        else:
            # 지지선이 하나도 없으면 현재 가격에서 하락
            support_levels.append(round(current_price * 0.99))
            support_levels.append(round(current_price * 0.98))
    
    return {
        "resistance": resistance_levels,
        "support": support_levels
    }

def identify_last_tested_level(candles: List[DailyCandleStick], support_levels, resistance_levels) -> str:
    """
    최근 테스트된 레벨 식별
    """
    # 최근 5개 캔들 확인
    recent_candles = candles[:5]
    price_tolerance = 0.005  # 0.5% 오차 허용
    
    last_tested_resistance = -1  # 가장 최근에 저항선이 테스트된 캔들 인덱스
    last_tested_support = -1  # 가장 최근에 지지선이 테스트된 캔들 인덱스
    
    # 각 캔들에 대해 레벨 테스트 확인
    for i, candle in enumerate(recent_candles):
        # 저항선 테스트 확인
        for resistance in resistance_levels:
            high_diff = abs(resistance - candle.high_price) / resistance
            if high_diff < price_tolerance:
                last_tested_resistance = i
                break
        
        # 지지선 테스트 확인
        for support in support_levels:
            low_diff = abs(support - candle.low_price) / support
            if low_diff < price_tolerance:
                last_tested_support = i
                break
        
        # 둘 다 테스트된 경우, 가장 최근 것을 반환
        if last_tested_resistance != -1 and last_tested_support != -1:
            return "resistance" if last_tested_resistance <= last_tested_support else "support"
    
    # 하나만 테스트된 경우
    if last_tested_resistance != -1:
        return "resistance"
    if last_tested_support != -1:
        return "support"
    
    # 둘 다 테스트되지 않은 경우, 최근 가격 움직임으로 결정
    if candles[0].trade_price > candles[1].trade_price:
        return "resistance"  # 상승 중이므로 저항선 방향으로 이동 중
    else:
        return "support"  # 하락 중이므로 지지선 방향으로 이동 중

def calculate_distance_to_level(from_price, to_price) -> float:
    """
    두 가격 레벨 간의 거리를 백분율로 계산
    """
    return ((to_price - from_price) / from_price) * 100

async def calculate_moving_averages() -> dict:
    """
    비트코인의 이동평균(moving_averages) 정보를 구하는 함수
    moving_averages는 20일, 50일, 200일 이동평균을 포함합니다.
    moving_averages 데이터
    - ma_20d: 20일 이동평균
    - ma_50d: 50일 이동평균
    - ma_200d: 200일 이동평균
    - ma_crossovers: 이동평균 교차 정보 (Golden Cross, Death Cross)
    """
    # 1. 캔들 데이터 가져오기
    candles = await get_candles_for_daily(count=200)  # 200일 이동평균을 위해 충분한 데이터
    
    # 2. 데이터프레임으로 변환
    df = pd.DataFrame(candles)
    df = df.sort_values('candle_date_time_utc')  # 날짜 오름차순 정렬
    
    # 3. 이동평균 계산
    df['ma_20d'] = df['trade_price'].rolling(window=20).mean()
    df['ma_50d'] = df['trade_price'].rolling(window=50).mean()
    df['ma_200d'] = df['trade_price'].rolling(window=200).mean()
    
    # 4. 최신 값 추출
    current_price = df['trade_price'].iloc[-1]
    ma_20d = df['ma_20d'].iloc[-1]
    ma_50d = df['ma_50d'].iloc[-1]
    ma_200d = df['ma_200d'].iloc[-1]
    
    # 5. Position과 Signal 결정
    ma_20d_position = "above" if current_price > ma_20d else "below"
    ma_20d_signal = "bullish" if ma_20d_position == "above" else "bearish"
    
    ma_50d_position = "above" if current_price > ma_50d else "below"
    ma_50d_signal = "bullish" if ma_50d_position == "above" else "bearish"
    
    ma_200d_position = "above" if current_price > ma_200d else "below"
    ma_200d_signal = "bullish" if ma_200d_position == "above" else "bearish"
    
    # 6. 이동평균 교차 확인 (Golden Cross, Death Cross)
    ma_crossovers = []
    
    # 최근 30일간의 데이터에서 교차 확인
    for i in range(1, min(30, len(df))):
        # 이전 날 관계
        prev_ma20_above_ma50 = df['ma_20d'].iloc[-i-1] > df['ma_50d'].iloc[-i-1]
        # 현재 날 관계
        curr_ma20_above_ma50 = df['ma_20d'].iloc[-i] > df['ma_50d'].iloc[-i]
        
        # 관계가 바뀌었다면 교차 발생
        if prev_ma20_above_ma50 != curr_ma20_above_ma50:
            crossover_type = "golden_cross" if curr_ma20_above_ma50 else "death_cross"
            ma_crossovers.append({
                "type": crossover_type,
                "fast_ma": "20d",
                "slow_ma": "50d",
                "days_ago": i
            })
    
    # 7. 결과 반환
    moving_averages = {
        "ma_200d": {
            "value": int(ma_200d),
            "position": ma_200d_position,
            "signal": ma_200d_signal
        },
        "ma_50d": {
            "value": int(ma_50d),
            "position": ma_50d_position,
            "signal": ma_50d_signal
        },
        "ma_20d": {
            "value": int(ma_20d),
            "position": ma_20d_position,
            "signal": ma_20d_signal
        },
        "ma_crossovers": ma_crossovers
    }
    
    return moving_averages

async def calculate_momentum() -> dict:
    """
    비트코인의 모멘텀 지표를 계산하는 함수
    모멘텀 지표는 RSI, MACD, Stochastic Oscillator를 포함합니다.
    """
    # 1. 캔들 데이터 가져오기
    candles = await get_candles_for_daily(count=50)  # 충분한 데이터
    
    # 2. 데이터프레임으로 변환
    df = pd.DataFrame(candles)
    df = df.sort_values('candle_date_time_utc')  # 날짜 오름차순 정렬
    
    # 3. RSI 계산 (14일 기준)
    delta = df['trade_price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    # 최신 및 이전 RSI 값
    current_rsi = df['rsi_14d'].iloc[-1]
    previous_rsi = df['rsi_14d'].iloc[-2]
    
    # RSI zone 결정
    rsi_zone = "neutral"
    if current_rsi >= 70:
        rsi_zone = "overbought"
    elif current_rsi <= 30:
        rsi_zone = "oversold"
    
    # RSI trend 결정
    rsi_trend = "neutral"
    if current_rsi > previous_rsi:
        rsi_trend = "rising"
    elif current_rsi < previous_rsi:
        rsi_trend = "falling"
    
    # 4. MACD 계산
    # EMA 계산
    df['ema_12'] = df['trade_price'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['trade_price'].ewm(span=26, adjust=False).mean()
    
    # MACD 라인 = 12일 EMA - 26일 EMA
    df['macd_line'] = df['ema_12'] - df['ema_26']
    
    # 시그널 라인 = MACD의 9일 EMA
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    
    # 히스토그램 = MACD 라인 - 시그널 라인
    df['histogram'] = df['macd_line'] - df['signal_line']
    
    # 최신 값과 이전 값
    current_macd = df['macd_line'].iloc[-1]
    current_signal = df['signal_line'].iloc[-1]
    current_histogram = df['histogram'].iloc[-1]
    previous_histogram = df['histogram'].iloc[-2]
    
    # MACD 추세 결정
    macd_trend = "neutral"
    if abs(current_histogram) < abs(previous_histogram):
        macd_trend = "converging"
    elif abs(current_histogram) > abs(previous_histogram):
        macd_trend = "diverging"
    elif (current_histogram > 0 and previous_histogram < 0) or (current_histogram < 0 and previous_histogram > 0):
        macd_trend = "crossover"
    
    # 5. Stochastic Oscillator 계산
    k_period = 14
    d_period = 3
    
    # %K = (현재가 - 기간 내 최저가) / (기간 내 최고가 - 기간 내 최저가) * 100
    df['lowest_low'] = df['low_price'].rolling(window=k_period).min()
    df['highest_high'] = df['high_price'].rolling(window=k_period).max()
    df['stoch_k'] = ((df['trade_price'] - df['lowest_low']) / 
                     (df['highest_high'] - df['lowest_low'])) * 100
    
    # %D = %K의 3일 이동평균
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    
    # 최신 값과 이전 값
    current_k = df['stoch_k'].iloc[-1]
    current_d = df['stoch_d'].iloc[-1]
    previous_k = df['stoch_k'].iloc[-2]
    previous_d = df['stoch_d'].iloc[-2]
    
    # Stochastic zone 결정
    stoch_zone = "neutral"
    if current_k >= 80 and current_d >= 80:
        stoch_zone = "overbought"
    elif current_k <= 20 and current_d <= 20:
        stoch_zone = "oversold"
    
    # Stochastic trend 결정
    stoch_trend = "neutral"
    if current_k > previous_k and current_d > previous_d:
        stoch_trend = "bullish"
    elif current_k < previous_k and current_d < previous_d:
        stoch_trend = "bearish"
    elif current_k > current_d and previous_k <= previous_d:
        stoch_trend = "bullish_crossover"
    elif current_k < current_d and previous_k >= previous_d:
        stoch_trend = "bearish_crossover"
    
    # 6. 결과 반환
    momentum = {
        "rsi_14d": {
            "value": round(current_rsi, 1),
            "zone": rsi_zone,
            "trend": rsi_trend
        },
        "macd": {
            "line": round(current_macd / 1000000, 2),  # 단위 조정
            "signal": round(current_signal / 1000000, 2),
            "histogram": round(current_histogram / 1000000, 2),
            "trend": macd_trend
        },
        "stochastic": {
            "k": round(current_k, 1),
            "d": round(current_d, 1),
            "trend": stoch_trend,
            "zone": stoch_zone
        }
    }
    
    return momentum

async def calculate_volatility_indicators() -> dict:
    """
    변동성 분석 - Bollinger Bands, ATR
    """
    # 1. 캔들 데이터 가져오기
    candles = await get_candles_for_daily(count=200)  # 충분한 데이터
    
    # 2. 데이터프레임으로 변환
    df = pd.DataFrame(candles)
    df = df.sort_values('candle_date_time_utc')  # 날짜 오름차순 정렬
    
    # 3. Bollinger Bands 계산 (20일 SMA 기준, 2×표준편차)
    period = 20
    multiplier = 2
    
    df['sma_20'] = df['trade_price'].rolling(window=period).mean()
    df['std_20'] = df['trade_price'].rolling(window=period).std()
    df['upper_band'] = df['sma_20'] + (df['std_20'] * multiplier)
    df['lower_band'] = df['sma_20'] - (df['std_20'] * multiplier)
    
    # 밴드 폭 계산 (표준화된 백분율)
    df['band_width'] = ((df['upper_band'] - df['lower_band']) / df['sma_20']) * 100
    
    # 현재 가격의 밴드 내 위치 계산 (0: 하단, 100: 상단)
    current_price = df['trade_price'].iloc[-1]
    latest_upper = df['upper_band'].iloc[-1]
    latest_lower = df['lower_band'].iloc[-1]
    latest_band_width = df['band_width'].iloc[-1]
    
    position = min(100, max(0, ((current_price - latest_lower) / (latest_upper - latest_lower)) * 100))
    
    # 밴드 폭 백분위 계산
    band_width_history = df['band_width'].dropna().sort_values()
    width_percentile = int(band_width_history.searchsorted(latest_band_width) / len(band_width_history) * 100)
    
    # 볼린저 밴드 신호 결정
    bb_signal = "neutral"
    if position > 80:
        bb_signal = "overbought"
    elif position < 20:
        bb_signal = "oversold"
    
    # 4. ATR (Average True Range) 계산
    df['prev_close'] = df['trade_price'].shift(1)
    
    # True Range = max(고가-저가, |고가-이전 종가|, |저가-이전 종가|)
    df['tr1'] = df['high_price'] - df['low_price']
    df['tr2'] = abs(df['high_price'] - df['prev_close'])
    df['tr3'] = abs(df['low_price'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 14일 ATR
    df['atr_14d'] = df['true_range'].rolling(window=14).mean()
    
    latest_atr = df['atr_14d'].iloc[-1]
    
    # ATR 백분위 계산
    atr_history = df['atr_14d'].dropna().sort_values()
    atr_percentile = int(atr_history.searchsorted(latest_atr) / len(atr_history) * 100)
    
    # 5. 결과 반환
    volatility = {
        "bollinger_bands": {
            "width_percentile": width_percentile,
            "position": round(position),
            "signal": bb_signal
        },
        "atr_14d": int(latest_atr),
        "atr_percentile": atr_percentile
    }
    
    return volatility

async def calculate_volume() -> dict:
    """
    거래량 분석 - OBV, Volume EMA Ratio, Volume-Price Trend
    """
    # 1. 캔들 데이터 가져오기
    candles = await get_candles_for_daily(count=50)  # 충분한 데이터
    
    # 2. 데이터프레임으로 변환
    df = pd.DataFrame(candles)
    df = df.sort_values('candle_date_time_utc')  # 날짜 오름차순 정렬
    
    # 3. OBV (On-Balance Volume) 계산
    df['price_change'] = df['trade_price'].diff()
    df['obv_change'] = np.where(df['price_change'] > 0, 
                                df['candle_acc_trade_volume'], 
                               np.where(df['price_change'] < 0, 
                                       -df['candle_acc_trade_volume'], 0))
    df['obv'] = df['obv_change'].cumsum()
    
    # OBV 추세 결정 (최근 5일 변화)
    recent_obv = df['obv'].tail(5)
    obv_slope = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
    
    obv_trend = "neutral"
    if obv_slope > 0:
        obv_trend = "rising"
    elif obv_slope < 0:
        obv_trend = "falling"
    
    # 4. Volume EMA Ratio 계산
    period = 20
    df['volume_ema'] = df['candle_acc_trade_volume'].ewm(span=period, adjust=False).mean()
    
    # 최신 거래량과 EMA 비율
    current_volume = df['candle_acc_trade_volume'].iloc[-1]
    volume_ema = df['volume_ema'].iloc[-1]
    volume_ema_ratio = round(current_volume / volume_ema, 2)
    
    # 5. Volume-Price Trend 계산
    # 최근 5일 데이터로 추세 분석
    period = 5
    recent_prices = df['trade_price'].tail(period)
    recent_volumes = df['candle_acc_trade_volume'].tail(period)
    
    # 가격 추세
    price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
    price_direction = "up" if price_change > 0 else "down" if price_change < 0 else "flat"
    
    # 거래량 추세
    volume_change = recent_volumes.iloc[-1] - recent_volumes.iloc[0]
    volume_direction = "up" if volume_change > 0 else "down" if volume_change < 0 else "flat"
    
    # 가격-거래량 관계 분석
    volume_price_trend = "neutral"
    if (price_direction == "up" and volume_direction == "up") or \
       (price_direction == "down" and volume_direction == "down"):
        volume_price_trend = "confirming"
    elif (price_direction == "up" and volume_direction == "down") or \
         (price_direction == "down" and volume_direction == "up"):
        volume_price_trend = "diverging"
    
    # 6. 결과 반환
    volume_data = {
        "obv_trend": obv_trend,
        "volume_ema_ratio": volume_ema_ratio,
        "volume_price_trend": volume_price_trend
    }
    
    return volume_data

async def get_technical_signals() -> dict:
    """
    비트코인 기술적 신호(technical_signals) 정보를 구하는 함수
    technical_signals는 이동평균, 모멘텀, 변동성, 거래량 지표를 포함합니다.
    technical_signals 데이터
    - moving_averages: 이동평균 지표 (20일, 50일, 200일)
    - momentum: 모멘텀 지표 (RSI, MACD, Stochastic)
    - volatility: 변동성 지표 (Bollinger Bands, ATR)
    - volume: 거래량 지표 (OBV, Volume EMA Ratio, Volume-Price Trend)
    """
    # 각 지표 계산
    moving_averages = await calculate_moving_averages()
    momentum = await calculate_momentum()
    volatility = await calculate_volatility_indicators()
    volume = await calculate_volume()
    
    # 모든 지표 통합
    technical_signals = {
        "moving_averages": moving_averages,
        "momentum": momentum,
        "volatility": volatility,
        "volume": volume
    }
    
    return technical_signals

async def analyze_btc_mareket() -> dict:
    """
    비트코인 시장 분석 시스템 실행
    이 함수는 비트코인 시장 정보를 수집하고, 추세 분석 및 가격 레벨을 계산하여 종합적인 분석 결과를 반환합니다.

    분석 결과는 다음과 같은 항목을 포함합니다:
    1. market_info: 비트코인 시장 정보 (현재 가격, 변동성, 거래량 변화율 등)
    - symbol: 비트코인 심볼 (KRW-BTC)
    - current_price: 현재 비트코인 가격
    - day_change_pct: 24시간 변동률
    - timestamp: 현재 시간 (UTC)
    - 24h_volume: 24시간 거래량
    - 24h_volume_change_pct: 24시간 거래량 변화율
    2 trend_analysis: 비트코인 추세 분석 (단기, 중기, 장기)
    - short_term: 단기 추세 (1h-4h)
    - medium_term: 중기 추세 (1d-1w)
    - long_term: 장기 추세 (1w-1m)
    - trend_strength: 추세 강도 (0-100)
    - trend_duration_days: 현재 추세 지속 기간 (일수)
    3. price_levels: 비트코인 가격 레벨 (주요 지지선, 저항선 등)
    - key_resistance: 주요 저항선 (상위 2개)
    - key_support: 주요 지지선 (상위 2개)
    - last_tested: 최근 테스트된 레벨 (지지선/저항선)
    - distance_to_resistance_pct: 저항선까지의 거리 (퍼센트)
    - distance_to_support_pct: 지지선까지의 거리 (퍼센트)
    4. technical_signals: 비트코인 기술적 신호 (이동평균, 모멘텀, 변동성, 거래량)
    - moving_averages: 이동평균 지표 (20일, 50일, 200일)
    - momentum: 모멘텀 지표 (RSI, MACD, Stochastic)
    - volatility: 변동성 지표 (Bollinger Bands, ATR)
    - volume: 거래량 지표 (OBV, Volume EMA Ratio, Volume-Price Trend)
    """
    # 1. market_info 항목 구하기
    market_info = await get_market_info()
    print("Market Info:", market_info)
    
    # 2. trend_analysis 항목 구하기
    trend_analysis = await get_trend_analysis()
    print("Trend Analysis:", trend_analysis)
    
    # 3. price_levels 항목 구하기
    price_levels = await get_price_levels()
    print("Price Levels:", price_levels)

    # 4. technical_signals 항목 구하기
    technical_signals = await get_technical_signals()
    print("Technical Signals:", technical_signals)
    
    # 4. 전체 분석 결과 통합
    bitcoin_analysis = {
        "market_info": market_info,
        "trend_analysis": trend_analysis,
        "price_levels": price_levels,
        "technical_signals": technical_signals
    }
    
    return bitcoin_analysis
