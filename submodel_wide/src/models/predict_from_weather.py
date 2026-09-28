import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd


def load_json(file_path):
    """JSONファイルを読み込む共通関数"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_domain_features(weather_data, required_stations):
    """気象データ（site/weather_latest.json）から

    直交風成分や降水量などの計算要素（ドメイン特徴量）を算出する
    """
    stations_data = weather_data.get("data", {})
    extracted_raw = {}

    for station_id in required_stations:
        if station_id not in stations_data:
            # 該当地点のデータが存在しない場合は0で埋める
            extracted_raw[f"{station_id}_crosswind"] = 0.0
            extracted_raw[f"{station_id}_rain_1h"] = 0.0
            continue

        st = stations_data[station_id]
        wind_speed = st.get("wind_speed", 0.0)
        wind_deg = st.get("wind_deg", 0)
        rain_1h = st.get("rain_1h", 0.0)

        # 沿線の方位（山陽本線の広島エリア概ね45度-225度ライン）に対する直交風の計算例
        # 方位角差から垂直成分（sin）を算出
        track_angle = 45.0
        angle_rad = np.radians(wind_deg - track_angle)
        crosswind = abs(wind_speed * np.sin(angle_rad))

        extracted_raw[f"{station_id}_crosswind"] = round(crosswind, 2)
        extracted_raw[f"{station_id}_rain_1h"] = float(rain_1h)

    return extracted_raw


def align_features_with_selected(raw_features, selected_features):
    """学習時に Optuna / 特徴量選定で絞り込んだ

    selected_features_v2.json の並び順とカラムに厳密に一致させる
    """
    df_single = pd.DataFrame([raw_features])

    # 存在しない特徴量カラムがあれば 0 で埋めて追加
    for col in selected_features:
        if col not in df_single.columns:
            df_single[col] = 0.0

    # 必須の特徴量リストの並び順通りに並び替え
    df_aligned = df_single[selected_features]
    return df_aligned


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using latest weather data."
    )
    parser.add_argument(
        "--config", required=True, help="Path to config/config.json"
    )
    parser.add_argument(
        "--weather", required=True, help="Path to site/weather_latest.json"
    )
    parser.add_argument("--model_id", required=True, help="Target model ID")
    parser.add_argument(
        "--model_file",
        required=True,
        help="Path to downloaded .pkl model file",
    )
    parser.add_argument(
        "--selected_features",
        required=True,
        help="Path to selected_features_v2.json",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output prediction json"
    )
    args = parser.parse_args()

    # 1. 各種設定・データ・特徴量リストの読み込み
    config = load_json(args.config)
    weather_data = load_json(args.weather)
    selected_features = load_json(args.selected_features)

    # configから該当モデルの定義を検索
    model_cfg = next(
        (m for m in config["active_models"] if m["model_id"] == args.model_id),
        None,
    )
    if not model_cfg:
        raise ValueError(
            f"Config内に model_id: '{args.model_id}' が存在しません。"
        )

    # 2. 気象データから入力特徴量を抽出・生成
    required_stations = model_cfg.get("required_stations", [])
    raw_features = calculate_domain_features(weather_data, required_stations)

    # 3. 学習時と全く同じ列順・列名に整形
    X_input = align_features_with_selected(raw_features, selected_features)

    # 4. 学習済みモデル (.pkl) のロードと推論
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(
            f"モデルファイルが見つかりません: {args.model_file}"
        )

    model = joblib.load(args.model_file)

    # LightGBM Booster または Scikit-Learn Wrapper の両方に対応して確率計算
    if hasattr(model, "predict_proba"):
        delay_prob = model.predict_proba(X_input)[0][1]
    else:
        # LightGBM Booster オブジェクトの場合
        delay_prob = model.predict(X_input)[0]

    delay_prob = float(delay_prob)

    # リスクレベルの判定ルール
    if delay_prob >= 0.5:
        risk_level = "HIGH"
    elif delay_prob >= 0.2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 5. 結果JSONの書き出し
    output_data = {
        "model_id": args.model_id,
        "release_tag": model_cfg.get("release_tag", ""),
        "calculated_at": weather_data.get("fetched_at", ""),
        "delay_probability": round(delay_prob, 4),
        "risk_level": risk_level,
    }

    # 出力先ディレクトリが存在しない場合は自動作成
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(
        f"✅ 推論完了 [{args.model_id}]: 確率={delay_prob:.4f} ({risk_level}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
