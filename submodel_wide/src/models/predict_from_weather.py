import argparse
import json
import os
import joblib
import pandas as pd


def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_era5_features(weather_data, required_stations):
    """取得したリアルタイム気象データから

    学習時の特徴量名 (era5_p1_tp, era5_p1_wind 等) を生成
    """
    stations_data = weather_data.get("data", {})
    extracted_raw = {}

    for station_id in required_stations:
        st = stations_data.get(station_id, {})

        # APIの降雨量(rain_1h) -> era5_pN_tp
        # APIの風速(wind_speed) -> era5_pN_wind
        tp_value = float(st.get("rain_1h", 0.0))
        wind_value = float(st.get("wind_speed", 0.0))

        extracted_raw[f"{station_id}_tp"] = tp_value
        extracted_raw[f"{station_id}_wind"] = wind_value

    return extracted_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--selected_features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    weather_data = load_json(args.weather)
    selected_features = load_json(args.selected_features)

    model_cfg = next(
        (m for m in config["active_models"] if m["model_id"] == args.model_id),
        None,
    )
    if not model_cfg:
        raise ValueError(f"model_id: '{args.model_id}' が見つかりません。")

    # 1. 5地点の気象データ抽出（era5_p1_tp, era5_p1_wind ...）
    required_stations = model_cfg.get("required_stations", [])
    raw_features = extract_era5_features(weather_data, required_stations)

    # 2. 列順を selected_features_v2.json（学習時と完全一致）に揃える
    df_single = pd.DataFrame([raw_features])
    for col in selected_features:
        if col not in df_single.columns:
            df_single[col] = 0.0
    X_input = df_single[selected_features]

    # 3. モデル推論
    model = joblib.load(args.model_file)
    if hasattr(model, "predict_proba"):
        delay_prob = float(model.predict_proba(X_input)[0][1])
    else:
        delay_prob = float(model.predict(X_input)[0])

    # 4. 結果書き出し
    output_data = {
        "model_id": args.model_id,
        "release_tag": model_cfg.get("release_tag", ""),
        "calculated_at": weather_data.get("fetched_at", ""),
        "delay_probability": round(delay_prob, 4),
        "risk_level": (
            "HIGH"
            if delay_prob >= 0.5
            else ("MEDIUM" if delay_prob >= 0.2 else "LOW")
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(
        f"✅ モデル2 ({args.model_id}) 推論完了: 遅延確率 = {delay_prob:.4f}"
    )


if __name__ == "__main__":
    main()
