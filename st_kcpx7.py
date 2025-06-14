import streamlit as st
import pandas as pd
import json
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np # Used for np.nan for robust data handling


# --- Helper Functions ---

@st.cache_data
def load_initial_excel_data(file_path="chiller_data.xlsx"):
    """
    Loads initial chiller specs and data from a predefined Excel file.
    Used for initial app state and 'Manual Key-in' fallback.
    """
    chspec_initial = []
    data1_initial = []
    try:
        # Load Chiller Specs
        df_spec = pd.read_excel(file_path, sheet_name='Chiller_Specs')
        chspec_initial = df_spec.to_dict(orient='records')

        # Load Chiller Operational Data (Historical)
        df_data_raw = pd.read_excel(file_path, sheet_name='Chiller_Data') # Assuming 'Chiller_Data' sheet for historical
        data1_initial = _process_raw_data_df_to_data_structure(df_data_raw)
        st.success(f"Initial historical data loaded from '{file_path}'.")
    except FileNotFoundError:
        st.warning(f"Excel file '{file_path}' not found. Starting with empty data.")
    except Exception as e:
        st.error(f"Error loading initial Excel data: {e}. Starting with empty data.")
    return chspec_initial, data1_initial

def _process_raw_data_df_to_data_structure(df_raw_data):
    """
    Helper to convert a flat DataFrame (from Excel or CSV)
    into the nested list-of-dictionaries format:
    [{"name": "chX", "data": [...]}]
    Assumes df_raw_data has a 'ch_name' column.
    Calculates 'ton' and 'kw_per_ton'.
    """
    data_output = []
    if 'ch_name' not in df_raw_data.columns:
        raise ValueError("Input DataFrame must contain a 'ch_name' column.")

    for chiller_name in df_raw_data['ch_name'].unique():
        chiller_data_df = df_raw_data[df_raw_data['ch_name'] == chiller_name].copy() # Use .copy() to avoid SettingWithCopyWarning

        # --- NEW CALCULATIONS FOR 'ton' and 'kw_per_ton' ---
        # Calculate 'ton' if 'fwch', 'tchr', and 'tchs' columns exist
        if all(col in chiller_data_df.columns for col in ['fwch', 'tchr', 'tchs']):
            # Ensure calculations handle potential NaN values by converting to numeric, then filling NaN with 0 for calculation
            # Use .fillna(0) for calculations to prevent NaN results, or use .dropna() if rows with missing data should be excluded
            # For this formula, it's safer to ensure numeric types
            chiller_data_df['fwch'] = pd.to_numeric(chiller_data_df['fwch'], errors='coerce').fillna(0)
            chiller_data_df['tchr'] = pd.to_numeric(chiller_data_df['tchr'], errors='coerce').fillna(0)
            chiller_data_df['tchs'] = pd.to_numeric(chiller_data_df['tchs'], errors='coerce').fillna(0)

            # The formula is ton = fwch * (tchr - tchs) / 24
            chiller_data_df['ton'] = (chiller_data_df['fwch'] * (chiller_data_df['tchr'] - chiller_data_df['tchs'])*3) / 40

        # Calculate 'kw_per_ton' if 'kw' and 'ton' columns exist
        # 'ton' must be calculated first or already present
        if 'kw' in chiller_data_df.columns and 'ton' in chiller_data_df.columns:
            chiller_data_df['kw'] = pd.to_numeric(chiller_data_df['kw'], errors='coerce').fillna(0) # Ensure numeric
            # Avoid division by zero: if 'ton' is 0 or very small, kw_per_ton is effectively infinite or NaN
            chiller_data_df['kw_per_ton'] = chiller_data_df.apply(
                lambda row: row['kw'] / row['ton'] if pd.notna(row['ton']) and abs(row['ton']) > 1e-6 else pd.NA, axis=1
            )

        # Calculate approach temp and effectiveness 
        if 'tamb' in chiller_data_df.columns and 'rh' in chiller_data_df.columns:
                chiller_data_df['Approach Temp'] = chiller_data_df['tcdr'] - (-14.098+0.935*chiller_data_df['tamb'] +0.165*chiller_data_df['rh'] )
                chiller_data_df['Effectiveness'] = 100* (chiller_data_df['tcds']-chiller_data_df['tcdr'])/(chiller_data_df['tcds']- (-14.098+0.935*chiller_data_df['tamb'] +0.165*chiller_data_df['rh'] ))                                      

        # --- END NEW CALCULATIONS ---

        chiller_data_list = chiller_data_df.drop(columns=['ch_name']).to_dict(orient='records')
        data_output.append({"name": chiller_name, "data": chiller_data_list})
    return data_output


def save_data_to_excel(chspec, historical_data, current_data, file_name="output_chiller_data.xlsx"):
    """Saves the current chspec, historical, and current data back to an Excel file."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Save Chiller Specs
        df_specs_output = pd.DataFrame(chspec)
        df_specs_output.to_excel(writer, sheet_name='Chiller_Specs', index=False)

        # Save Historical Data (data1)
        all_hist_data_rows = []
        for entry in historical_data:
            ch_name = entry['name']
            for data_row in entry['data']:
                row = {"ch_name": ch_name}
                row.update(data_row)
                all_hist_data_rows.append(row)
        df_hist_data_output = pd.DataFrame(all_hist_data_rows)
        df_hist_data_output.to_excel(writer, sheet_name='Chiller_Data_Historical', index=False) # Renamed sheet

        # Save Current Operational Data (data3)
        if current_data:
            all_current_data_rows = []
            for entry in current_data:
                ch_name = entry['name']
                for data_row in entry['data']:
                    row = {"ch_name": ch_name}
                    row.update(data_row)
                    all_current_data_rows.append(row)
            df_current_data_output = pd.DataFrame(all_current_data_rows)
            df_current_data_output.to_excel(writer, sheet_name='Chiller_Data_Current', index=False) # New sheet

    output.seek(0)
    return output, file_name

# --- NEW HELPER FOR LOADING ALL FROM OUTPUT FILE ---
def load_all_from_output_excel(uploaded_file):
    """Loads chspec, data1, and data3 from a saved output Excel file."""
    chspec_loaded = []
    data1_loaded = []
    data3_loaded = []

    try:
        # Create a Pandas ExcelFile object to access sheets by name
        xls = pd.ExcelFile(uploaded_file)

        # Read Chiller Specs
        if 'Chiller_Specs' in xls.sheet_names:
            df_specs = xls.parse(sheet_name='Chiller_Specs')
            chspec_loaded = df_specs.to_dict(orient='records')
        else:
            st.warning("'Chiller_Specs' sheet not found in the uploaded file.")

        # Read Historical Data
        if 'Chiller_Data_Historical' in xls.sheet_names:
            df_data1_raw = xls.parse(sheet_name='Chiller_Data_Historical')
            data1_loaded = _process_raw_data_df_to_data_structure(df_data1_raw)
        else:
            st.warning("'Chiller_Data_Historical' sheet not found in the uploaded file.")

        # Read Current Data
        if 'Chiller_Data_Current' in xls.sheet_names:
            df_data3_raw = xls.parse(sheet_name='Chiller_Data_Current')
            data3_loaded = _process_raw_data_df_to_data_structure(df_data3_raw)
        else:
            st.warning("'Chiller_Data_Current' sheet not found in the uploaded file.")

        st.success("Successfully loaded data from the uploaded Excel file!")
        return chspec_loaded, data1_loaded, data3_loaded

    except Exception as e:
        st.error(f"Error loading data from Excel: {e}")
        return [], [], [] # Return empty on error


# --- Model Placeholders (Replace with your actual model logic) ---


def run_random_forest_model(chspec, historical_data, current_data):
    """
    Implements a Random Forest Regressor model for kW prediction and anomaly detection.
    Trains individual Random Forest models for each chiller on historical_data (data1)
    and predicts for current_data (data3).
    Adds 'predicted_kw' column and flags anomalies based on prediction deviation.
    """
    st.info("Running Random Forest Model (Individual Chiller Models)...")
    results = {
        "chiller_graphs_data": {},
        "anomaly_results": [],
        "general_comments": "Random Forest model executed successfully. ",
        "stat_his":{},
        "stat_cur":{},
        "kpi":{}
    }

    # Define features and target for the Random Forest model
    feature_cols = ['ton', 'tchs', 'tchr', 'tcds', 'tcdr', 'fwch']
    target_col = 'kw'

    # --- 0. Consolidate DataFrames for easier processing ---
    # Flatten historical_data (list of dicts) into a single DataFrame
    all_hist_records_flat = []
    for entry in historical_data:
        ch_name = entry['name']
        for data_row in entry['data']:
            row_with_name = {"ch_name": ch_name}
            row_with_name.update(data_row)
            all_hist_records_flat.append(row_with_name)
    df_historical_flat = pd.DataFrame(all_hist_records_flat)
    all_current_records_flat = []
    data_for_prediction_source = current_data if current_data else historical_data # Prioritize current, fallback to historical
    for entry in data_for_prediction_source:
        ch_name = entry['name']
        for data_row in entry['data']:
            row_with_name = {"ch_name": ch_name}
            row_with_name.update(data_row)
            all_current_records_flat.append(row_with_name)
    df_current_flat = pd.DataFrame(all_current_records_flat)

    if df_historical_flat.empty:
        results["general_comments"] += "No historical data available to train Random Forest models."
        st.warning("No historical data available to train Random Forest models. Please load data1.")
        return results
    if df_current_flat.empty:
        results["general_comments"] += "No current data available for prediction."
        st.warning("No current data available for prediction. Please load data3 or ensure data1 is available.")
        return results

    # Ensure all relevant columns are numeric for both dataframes
    cols_to_numeric = feature_cols + [target_col]
    for df in [df_historical_flat, df_current_flat]:
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify unique chillers from historical data (only train for chillers we have historical data for)
    unique_chillers = df_historical_flat['ch_name'].unique()
    if len(unique_chillers) == 0:
        results["general_comments"] += "No unique chillers found in historical data."
        st.warning("No unique chillers found in historical data to train Random Forest models.")
        return results

    all_chiller_predictions_df = pd.DataFrame() # To store all predicted data for all chillers

    st.write(f"Training and predicting for {len(unique_chillers)} chillers...")

    for chiller_name in unique_chillers:
        st.write(f"--- Processing Chiller: {chiller_name} ---")

        # --- 1. Prepare Training Data for current chiller ---
        chiller_df_train = df_historical_flat[df_historical_flat['ch_name'] == chiller_name].copy()
        
        # Calculate statistics, skipping NaN values
        results["stat_his"][chiller_name] = {
       "Load":f"{chiller_df_train['ton'].mean():.1f} Ton (min {chiller_df_train['ton'].min():.1f} -max {chiller_df_train['ton'].max():.1f})" if 'ton' in chiller_df_train.columns else np.nan,
       "Power(kW)": f"{chiller_df_train['kw'].mean():.1f} kW (min {chiller_df_train['kw'].min():.1f} -max {chiller_df_train['kw'].max():.1f})"if 'kw' in chiller_df_train.columns else np.nan,   
       "Performance": f"{chiller_df_train['kw_per_ton'].mean():.2f} kW/Ton (min {chiller_df_train['kw_per_ton'].min():.2f} -max {chiller_df_train['kw_per_ton'].max():.2f})" if 'kw_per_ton' in chiller_df_train.columns else np.nan,
       "Temp(CHS)": f"{chiller_df_train['tchs'].mean():.1f} C (min {chiller_df_train['tchs'].min():.1f} -max {chiller_df_train['tchs'].max():.1f})" if 'tchs' in chiller_df_train.columns else np.nan,
       "Temp(CDR)": f"{chiller_df_train['tcdr'].mean():.1f} C (min {chiller_df_train['tcdr'].min():.1f} -max {chiller_df_train['tcdr'].max():.1f})" if 'tcdr' in chiller_df_train.columns else np.nan, 
        }
 

        # Drop rows with NaN in features or target for training. Model needs complete data.
        chiller_df_train_cleaned = chiller_df_train.dropna(subset=feature_cols + [target_col])

        if chiller_df_train_cleaned.empty:
            st.warning(f"Insufficient clean historical data to train Random Forest model for {chiller_name}. Skipping.")
            # Add this chiller's current data (if any) with NaN for predicted_kw to the overall results
            chiller_df_current_for_merge = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
            if not chiller_df_current_for_merge.empty:
                chiller_df_current_for_merge['predicted_kw'] = np.nan
                all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_current_for_merge], ignore_index=True)
            continue

        X_train = chiller_df_train_cleaned[feature_cols]
        y_train = chiller_df_train_cleaned[target_col]

        # Ensure there's enough data to train. RF can train with few samples, but more is better.
        if len(X_train) < 2: # At least 2 samples to train
            st.warning(f"Insufficient data ({len(X_train)} samples) to train Random Forest model for {chiller_name}. Requires at least 2 samples. Skipping.")
            chiller_df_current_for_merge = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
            if not chiller_df_current_for_merge.empty:
                chiller_df_current_for_merge['predicted_kw'] = np.nan
                all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_current_for_merge], ignore_index=True)
            continue

        # Initialize and fit StandardScaler on this chiller's training features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Initialize and train RandomForestRegressor for this specific chiller
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        st.write(f"  Random Forest model trained for {chiller_name}.")

        # --- 2. Prepare Prediction Data for current chiller ---
        chiller_df_predict = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()

        # Calculate statistics, skipping NaN values
        results["stat_cur"][chiller_name]={
       "Load":f"{chiller_df_predict['ton'].mean():.1f} Ton (min {chiller_df_predict['ton'].min():.1f} -max {chiller_df_predict['ton'].max():.1f})" if 'ton' in chiller_df_predict.columns else np.nan,
       "Power(kW)": f"{chiller_df_predict['kw'].mean():.1f} kW (min {chiller_df_predict['kw'].min():.1f} -max {chiller_df_predict['kw'].max():.1f})"if 'kw' in chiller_df_predict.columns else np.nan,   
       "Performance": f"{chiller_df_predict['kw_per_ton'].mean():.2f} kW/Ton (min {chiller_df_predict['kw_per_ton'].min():.2f} -max {chiller_df_predict['kw_per_ton'].max():.2f})" if 'kw_per_ton' in chiller_df_predict.columns else np.nan,
       "Temp(CHS)": f"{chiller_df_predict['tchs'].mean():.1f} C (min {chiller_df_predict['tchs'].min():.1f} -max {chiller_df_predict['tchs'].max():.1f})" if 'tchs' in chiller_df_predict.columns else np.nan,
       "Temp(CDR)": f"{chiller_df_predict['tcdr'].mean():.1f} C (min {chiller_df_predict['tcdr'].min():.1f} -max {chiller_df_predict['tcdr'].max():.1f})" if 'tcdr' in chiller_df_predict.columns else np.nan, 
        }
    
        
        if chiller_df_predict.empty:
            st.write(f"  No current data to predict for {chiller_name}.")
            continue

        # Drop rows with NaN in features for prediction.
        chiller_df_predict_cleaned = chiller_df_predict.dropna(subset=feature_cols)

        if chiller_df_predict_cleaned.empty:
            st.warning(f"  No clean current data to make predictions for {chiller_name}. Skipping.")
            chiller_df_predict['predicted_kw'] = np.nan # Add NaN column if no prediction possible
            all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_predict], ignore_index=True)
            continue

        X_predict = chiller_df_predict_cleaned[feature_cols]
 
        # Transform prediction features using the SAME scaler fitted on this chiller's training data
        X_predict_scaled = scaler.transform(X_predict)

        # Make predictions
        predicted_kw_values = rf_model.predict(X_predict_scaled)
        chiller_df_predict_cleaned['predicted_kw'] = predicted_kw_values

        # Merge predictions back to the original chiller_df_predict
        chiller_df_predict = chiller_df_predict.merge(
            chiller_df_predict_cleaned[['ch_name', 'h', 'predicted_kw']],
            on=['ch_name', 'h'],
            how='left'
        )
        all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_predict], ignore_index=True)

        # --- Calculate full load and iplv value
        iplv=0
        cap=0
        rkw=0
        rkwpton=0
        for items in chspec:
            if items['name']==chiller_name:
                iplv=items['iplv']              
                cap=items['cap']
                rkw=items['rkw']
                rkwpton=rkw/cap
        input_values_4q = [cap, 7, 11, 31, 27, 2.4*cap]
        input_values_3q = [0.75*cap, 7, 11, 31, 27, 2.4*cap]
        input_values_2q = [0.5*cap, 7, 11, 31, 27, 2.4*cap]   
        input_values_1q = [0.25*cap, 7, 11, 31, 27, 2.4*cap]
        input_df_4q = pd.DataFrame([input_values_4q], columns=feature_cols)
        input_df_3q = pd.DataFrame([input_values_3q], columns=feature_cols)
        input_df_2q = pd.DataFrame([input_values_2q], columns=feature_cols)
        input_df_1q = pd.DataFrame([input_values_1q], columns=feature_cols)

        # Transform prediction features using the SAME scaler fitted on this chiller's training data
        X_predict_scaled4 = scaler.transform( input_df_4q)
        X_predict_scaled3 = scaler.transform( input_df_3q)
        X_predict_scaled2 = scaler.transform( input_df_2q)
        X_predict_scaled1 = scaler.transform( input_df_1q)
        # Make predictions
        predicted_kW4 = rf_model.predict(X_predict_scaled4)
        predicted_kW3 = rf_model.predict(X_predict_scaled3)
        predicted_kW2 = rf_model.predict(X_predict_scaled2)
        predicted_kW1 = rf_model.predict(X_predict_scaled1)
        kwpt1=predicted_kW1/(0.25*cap)
        kwpt2=predicted_kW2/(0.5*cap)
        kwpt3=predicted_kW3/(0.75*cap)
        kwpt4=predicted_kW4/(cap)
        iplv_value= 0.11*kwpt1+0.4*kwpt2+.23*kwpt3+0.25*kwpt4
        diff1= 100*(kwpt4-rkwpton)/rkwpton
        diff2= 100*(iplv_value-iplv)/iplv
        results["kpi"][chiller_name] = {
          "พิกัดทำความเย็น":cap,
          "พิกัดสมรรถนะ จากผู้ผลิต (kW/ton)":f"{rkwpton:.2f}",
          "พิกัด IPLV จากผู้ผลิต (kW/ton)":f"{iplv:.2f}",
          "ค่าสมรรถนะที่ประเมินจากโมเดลที่ Full Load (kW/ton)(%diff)":f"{float(kwpt4):.2f} ({float(diff1):.2f}"+" %)",
          "ค่า IPLV ที่ประเมินจากโมเดล (%diff)":f"{float(iplv_value):.2f} ({float(diff2):.2f}"+" %)"
        }

    # --- 3. Anomaly Detection and Graph Data Preparation for all chillers ---
    ANOMALY_THRESHOLD_PERCENT = 0.10 # 10% deviation (can be adjusted)

    # Now process the combined predictions for graphing and anomaly detection
    # Ensure all columns are numeric for plotting, fill NaNs if necessary
    for col in ['kw', 'predicted_kw', 'ton', 'kw_per_ton']:
        if col not in all_chiller_predictions_df.columns:
            all_chiller_predictions_df[col] = np.nan # Add missing columns
        all_chiller_predictions_df[col] = pd.to_numeric(all_chiller_predictions_df[col], errors='coerce')

    for chiller_name in all_chiller_predictions_df['ch_name'].unique():
        chiller_df = all_chiller_predictions_df[all_chiller_predictions_df['ch_name'] == chiller_name].copy()

        if 'h' not in chiller_df.columns:
            chiller_df['h'] = range(1, len(chiller_df) + 1)
            st.warning(f"Hour column 'h' not found for {chiller_name}. Using row index for plotting.")

        # Prepare data for 24h graphs, including 'predicted_kw'
        results["chiller_graphs_data"][chiller_name] = {
            "hours": chiller_df['h'].tolist(),
            "kw": chiller_df['kw'].tolist(),
            "predicted_kw": chiller_df['predicted_kw'].tolist(),
            "ton": chiller_df['ton'].tolist(),
            "kw_per_ton": chiller_df['kw_per_ton'].tolist()
        }

        # Anomaly detection based on actual vs. predicted kW
        if 'kw' in chiller_df.columns and 'predicted_kw' in chiller_df.columns:
            for index, row in chiller_df.iterrows():
                actual_kw = row['kw']
                predicted_kw_val = row['predicted_kw']

                if pd.notna(actual_kw) and pd.notna(predicted_kw_val):
                    if abs(predicted_kw_val) > 1e-6:
                        deviation = actual_kw - predicted_kw_val # Keep signed deviation
                        percentage_deviation = abs(deviation) / abs(predicted_kw_val)

                        if percentage_deviation > ANOMALY_THRESHOLD_PERCENT:
                            results["anomaly_results"].append({
                                "chiller": ch_name,
                                "hour": row['h'],
                                "metric": "kW (Actual vs. Predicted)",
                                "actual_value": f"{actual_kw:.1f}",
                                "predicted_value": f"{predicted_kw_val:.1f}",
                                "deviation": f"{deviation:.1f}",
                                "percentage_deviation": f"{percentage_deviation*100:.1f}%",
                                "comment": f"kW deviation from predicted: {percentage_deviation*100:.1f}%."
                            })
                    elif abs(actual_kw) > 1e-6:
                         results["anomaly_results"].append({
                            "chiller": chiller_name,
                            "hour": row['h'],
                            "metric": "kW (Actual vs. Predicted)",
                            "actual_value": f"{actual_kw:.1f}",
                            "predicted_value": f"{predicted_kw_val:.1f}",
                            "deviation": f"{actual_kw:.1f}",
                            "percentage_deviation": "N/A (Predicted kW is zero)",
                            "comment": "Actual kW is significant but predicted kW is zero/near-zero."
                        })
                elif pd.notna(actual_kw) and pd.isna(predicted_kw_val):
                     results["anomaly_results"].append({
                            "chiller": chiller_name,
                            "hour": row['h'],
                            "metric": "kW (Actual vs. Predicted)",
                            "actual_value": f"{actual_kw:.1f}",
                            "predicted_value": "N/A",
                            "deviation": "N/A",
                            "percentage_deviation": "N/A",
                            "comment": "Could not predict kW due to missing feature data for this hour."
                        })

    results["general_comments"] += "Review graphs for actual vs. predicted kW and anomaly table for deviations."
    return results

def run_gordon_ng_model(chspec, historical_data, current_data):
    """
    Implements the Gordon Ng model to find coefficients (beta0, beta1, beta2)
    and predict 'predicted_kw' for operational data.
    Trains individual models for each chiller on historical_data (data1)
    and predicts for current_data (data3).
    """
    st.info("Running Gordon Ng Model (Individual Chiller Models)...")
    results = {
        "chiller_graphs_data": {},
        "anomaly_results": [],
        "general_comments": "Gordon Ng model executed successfully. ",
        "stat_his":{},
        "stat_cur":{},
        "kpi":{}
    }
    # --- 0. Consolidate DataFrames for easier processing ---
    # Flatten historical_data (list of dicts) into a single DataFrame
    all_hist_records_flat = []
    for entry in historical_data:
        ch_name = entry['name']
        for data_row in entry['data']:
            row_with_name = {"ch_name": ch_name}
            row_with_name.update(data_row)
            all_hist_records_flat.append(row_with_name)
    df_historical_flat = pd.DataFrame(all_hist_records_flat)

    # Flatten current_data (list of dicts) into a single DataFrame
    all_current_records_flat = []
    data_for_prediction_source = current_data if current_data else historical_data # Prioritize current, fallback to historical
    for entry in data_for_prediction_source:
        ch_name = entry['name']
        for data_row in entry['data']:
            row_with_name = {"ch_name": ch_name}
            row_with_name.update(data_row)
            all_current_records_flat.append(row_with_name)
    df_current_flat = pd.DataFrame(all_current_records_flat)

    if df_historical_flat.empty:
        results["general_comments"] += "No historical data available to train Gordon Ng models."
        st.warning("No historical data available to train Gordon Ng models. Please load data1.")
        return results
    if df_current_flat.empty:
        results["general_comments"] += "No current data available for prediction."
        st.warning("No current data available for prediction. Please load data3 or ensure data1 is available.")
        return results

    # Ensure all relevant columns are numeric for both dataframes
    cols_to_numeric = ['ton', 'tchs', 'tchr', 'tcds', 'tcdr', 'fwch', 'kw']
    for df in [df_historical_flat, df_current_flat]:
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify unique chillers from historical data (only train for chillers we have historical data for)
    unique_chillers = df_historical_flat['ch_name'].unique()
    if len(unique_chillers) == 0:
        results["general_comments"] += "No unique chillers found in historical data."
        st.warning("No unique chillers found in historical data to train Gordon Ng models.")
        return results

    all_chiller_predictions_df = pd.DataFrame() # To store all predicted data for all chillers

    st.write(f"Training and predicting for {len(unique_chillers)} chillers...")

    for chiller_name in unique_chillers:
        st.write(f"--- Processing Chiller: {chiller_name} ---")

        # --- 1. Prepare Training Data for current chiller ---
        chiller_df_train = df_historical_flat[df_historical_flat['ch_name'] == chiller_name].copy()
        
        # calculate stat_his
        results["stat_his"][chiller_name] = {
       "Load":f"{chiller_df_train['ton'].mean():.1f} Ton (min {chiller_df_train['ton'].min():.1f} -max {chiller_df_train['ton'].max():.1f})" if 'ton' in chiller_df_train.columns else np.nan,
       "Power(kW)": f"{chiller_df_train['kw'].mean():.1f} kW (min {chiller_df_train['kw'].min():.1f} -max {chiller_df_train['kw'].max():.1f})"if 'kw' in chiller_df_train.columns else np.nan,   
       "Performance": f"{chiller_df_train['kw_per_ton'].mean():.2f} kW/Ton (min {chiller_df_train['kw_per_ton'].min():.2f} -max {chiller_df_train['kw_per_ton'].max():.2f})" if 'kw_per_ton' in chiller_df_train.columns else np.nan,
       "Temp(CHS)": f"{chiller_df_train['tchs'].mean():.1f} C (min {chiller_df_train['tchs'].min():.1f} -max {chiller_df_train['tchs'].max():.1f})" if 'tchs' in chiller_df_train.columns else np.nan,
       "Temp(CDR)": f"{chiller_df_train['tcdr'].mean():.1f} C (min {chiller_df_train['tcdr'].min():.1f} -max {chiller_df_train['tcdr'].max():.1f})" if 'tcdr' in chiller_df_train.columns else np.nan, 
        }

        # Define features and target for the Gordon Ng model
           # The Gordon Ng model typically uses:
        # X1 = (Tc-Tw)/TcTw                                   #Ton
        # X2 = (Q*Q/(tCtW))*(1+1/COP)                                   #(TCDR - TCDS)
        # Target Y =  ((Tw/Tc)*(1+1/cop) -1 )*(Q/Tw)                              #   kW
        gn_feature_cols = ['ton','tchr', 'tcdr', 'fwch', 'kw']            # Base features
        target_col = 'Y'

        # Drop rows where any required column is NaN for training
        chiller_df_train_cleaned = chiller_df_train.dropna(subset=gn_feature_cols )

        if chiller_df_train_cleaned.empty:
            st.warning(f"Insufficient clean historical data to train Gordon Ng model for {chiller_name}. Skipping.")
            # Add this chiller's current data (if any) with NaN for predicted_kw to the overall results
            chiller_df_current_for_merge = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
            if not chiller_df_current_for_merge.empty:
                chiller_df_current_for_merge['predicted_kw'] = np.nan
                all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_current_for_merge], ignore_index=True)
            continue


        # Feature Engineering for Gordon Ng model

        chiller_df_train_cleaned['X1'] = (chiller_df_train_cleaned['tcdr'] - chiller_df_train_cleaned['tchr'])/((chiller_df_train_cleaned['tcdr']+273.15)*(chiller_df_train_cleaned['tchr']+273.15))
        chiller_df_train_cleaned['X2'] = ((chiller_df_train_cleaned['ton']*chiller_df_train_cleaned['ton']*3510*3510 )/((chiller_df_train_cleaned['tcdr']+273.15)*(chiller_df_train_cleaned['tchr']+273.15)))*(1+(1/ ( 3.510*chiller_df_train_cleaned['ton']/chiller_df_train_cleaned['kw'])))
        chiller_df_train_cleaned['Y'] =  (((chiller_df_train_cleaned['tchr']+273.15)/(chiller_df_train_cleaned['tcdr']+273.15))*(1+(1/(3.510*chiller_df_train_cleaned['ton']/chiller_df_train_cleaned['kw'])))-1)*((3510*chiller_df_train_cleaned['ton'])/(chiller_df_train_cleaned['tchr']+273.15))       
        # Define Gordon Ng features
        final_gn_features = ['X1', 'X2']
        # Drop rows if interaction term became NaN (e.g., if original 'ton' or 'tcds' was NaN)
        chiller_df_train_cleaned.dropna(subset=final_gn_features, inplace=True)

        if chiller_df_train_cleaned.empty:
            st.warning(f"No valid features after engineering for {chiller_name}. Cannot train Gordon Ng model.")
            chiller_df_current_for_merge = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
            if not chiller_df_current_for_merge.empty:
                chiller_df_current_for_merge['predicted_kw'] = np.nan
                all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_current_for_merge], ignore_index=True)
            continue

        X_train_gn = chiller_df_train_cleaned[final_gn_features]
        y_train_gn = chiller_df_train_cleaned[target_col]

        # Ensure enough data to train Linear Regression (at least num_features + 1 samples)
        if len(X_train_gn) < len(final_gn_features) + 1:
            st.warning(f"Insufficient data ({len(X_train_gn)} samples) to train Gordon Ng model for {chiller_name}. Requires at least {len(final_gn_features) + 1} samples. Skipping.")
            chiller_df_current_for_merge = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
            if not chiller_df_current_for_merge.empty:
                chiller_df_current_for_merge['predicted_kw'] = np.nan
                all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_current_for_merge], ignore_index=True)
            continue

        # Initialize and fit Linear Regression model for this chiller
        gn_model = LinearRegression()
        gn_model.fit(X_train_gn, y_train_gn)

        beta0 = gn_model.intercept_
        beta1 = gn_model.coef_[0] if len(gn_model.coef_) > 0 else np.nan
        beta2 = gn_model.coef_[1] if len(gn_model.coef_) > 1 else np.nan
        #beta3 = gn_model.coef_[2] if len(gn_model.coef_) > 2 else np.nan # For the interaction term

        st.write(f"  Coefficients for {chiller_name}: β0={beta0:.3f}, β1={beta1:.3f}, β2={beta2:.3f} ")
        results["general_comments"] += f"Chiller {chiller_name} GN Coeffs: β0={beta0:.3f}, β1={beta1:.3f}, β2={beta2:.3f}. "

        # --- 2. Prepare Prediction Data for current chiller ---
        chiller_df_predict = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
        # Calculate statistics, skipping NaN values
        results["stat_cur"][chiller_name] = {
       "Load":f"{chiller_df_predict['ton'].mean():.1f} Ton (min {chiller_df_predict['ton'].min():.1f} -max {chiller_df_predict['ton'].max():.1f})" if 'ton' in chiller_df_predict.columns else np.nan,
       "Power(kW)": f"{chiller_df_predict['kw'].mean():.1f} kW (min {chiller_df_predict['kw'].min():.1f} -max {chiller_df_predict['kw'].max():.1f})"if 'kw' in chiller_df_predict.columns else np.nan,   
       "Performance": f"{chiller_df_predict['kw_per_ton'].mean():.2f} kW/Ton (min {chiller_df_predict['kw_per_ton'].min():.2f} -max {chiller_df_predict['kw_per_ton'].max():.2f})" if 'kw_per_ton' in chiller_df_predict.columns else np.nan,
       "Temp(CHS)": f"{chiller_df_predict['tchs'].mean():.1f} C (min {chiller_df_predict['tchs'].min():.1f} -max {chiller_df_predict['tchs'].max():.1f})" if 'tchs' in chiller_df_predict.columns else np.nan,
       "Temp(CDR)": f"{chiller_df_predict['tcdr'].mean():.1f} C (min {chiller_df_predict['tcdr'].min():.1f} -max {chiller_df_predict['tcdr'].max():.1f})" if 'tcdr' in chiller_df_predict.columns else np.nan, 
        }


        if chiller_df_predict.empty:
            st.write(f"  No current data to predict for {chiller_name}.")
            continue

        # Drop rows with NaN in base features for prediction
        chiller_df_predict_cleaned = chiller_df_predict.dropna(subset=gn_feature_cols)

        if chiller_df_predict_cleaned.empty:
            st.warning(f"  No clean current data to make predictions for {chiller_name}. Skipping.")
            chiller_df_predict['predicted_kw'] = np.nan
            all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_predict], ignore_index=True)
            continue

        # Create interaction term for prediction data
  
        #chiller_df_predict_cleaned['X1'] = ( chiller_df_predict_cleaned['tcdr'] -  chiller_df_predict_cleaned['tchr'])/(( chiller_df_predict_cleaned['tcdr']+273.15)*( chiller_df_predict_cleaned['tchr']+273.15))
        #chiller_df_predict_cleaned['X2'] = (( chiller_df_predict_cleaned['ton']* chiller_df_predict_cleaned['ton']*3510*3510 )/(( chiller_df_predict_cleaned['tcdr']+273.15)*( chiller_df_predict_cleaned['tchr']+273.15)))*(1+(1/ ( 3.510* chiller_df_predict_cleaned['ton']/ chiller_df_predict_cleaned['kw'])))
        #X_predict_gn = chiller_df_predict_cleaned[final_gn_features]
        # ab =Q/tw + beta0 + beta1* (tc-tw)/(tc*tw)
        # div = Q/tc - beta2*Q*Q/(Tc*tw)
        ab= (chiller_df_predict_cleaned['ton']*3510/(chiller_df_predict_cleaned['tchr']+273.15))+beta0+beta1*( chiller_df_predict_cleaned['tcdr'] -  chiller_df_predict_cleaned['tchr'])/(( chiller_df_predict_cleaned['tcdr']+273.15)*( chiller_df_predict_cleaned['tchr']+273.15))
        div= (chiller_df_predict_cleaned['ton']*3510/(chiller_df_predict_cleaned['tcdr']+273.15))-beta2*(chiller_df_predict_cleaned['ton']*chiller_df_predict_cleaned['ton']*3510*3510 )/(( chiller_df_predict_cleaned['tcdr']+273.15)*( chiller_df_predict_cleaned['tchr']+273.15))


        # Make predictions
        # kw =ton *3.51* (ab-div)/div
        predicted_kw_values = 3.510*chiller_df_predict_cleaned['ton']*(ab-div)/div
        #predicted_kw_values = ((gn_model.predict(X_predict_gn)*((chiller_df_predict_cleaned['tchr']+273.15)/(3510*chiller_df_predict_cleaned['ton']))+1)*((chiller_df_predict_cleaned['tcdr']+273.15)/(chiller_df_predict_cleaned['tchr']+273.15))-1)*chiller_df_predict_cleaned['ton']*3510/1000
        chiller_df_predict_cleaned['predicted_kw'] = predicted_kw_values
        st.write(chiller_df_predict_cleaned)
        # Merge predictions back to the original chiller_df_predict
        chiller_df_predict = chiller_df_predict.merge(
            chiller_df_predict_cleaned[['ch_name', 'h', 'predicted_kw']],
            on=['ch_name', 'h'],
            how='left'
        )
        all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_predict], ignore_index=True)

        # --- Calculate full load and iplv value
    
        iplv=0
        cap=0
        rkw=0
        rkwpton=0
        for items in chspec:
            if items['name']==chiller_name:
                iplv=items['iplv']              
                cap=items['cap']
                rkw=items['rkw']
                rkwpton=rkw/cap
        input_values_4q = [cap, 11, 31, 2.4*cap,100]
        input_values_3q = [0.75*cap, 11, 31, 2.4*cap,100]
        input_values_2q = [0.5*cap,11, 31, 2.4*cap,100]   
        input_values_1q = [0.25*cap, 11, 31, 2.4*cap,100]

        df_4q = pd.DataFrame([input_values_4q], columns=gn_feature_cols)
        df_3q = pd.DataFrame([input_values_3q], columns=gn_feature_cols)
        df_2q = pd.DataFrame([input_values_2q], columns=gn_feature_cols)
        df_1q = pd.DataFrame([input_values_1q], columns=gn_feature_cols)

        ab4= (df_4q['ton']*3510/(df_4q['tchr']+273.15))+beta0+beta1*( df_4q['tcdr'] -  df_4q['tchr'])/(( df_4q['tcdr']+273.15)*(df_4q['tchr']+273.15))
        div4= (df_4q['ton']*3510/(df_4q['tcdr']+273.15))-beta2*(df_4q['ton']*df_4q['ton']*3510*3510 )/(( df_4q['tcdr']+273.15)*( df_4q['tchr']+273.15))
        ab3= (df_3q['ton']*3510/(df_3q['tchr']+273.15))+beta0+beta1*( df_3q['tcdr'] -  df_3q['tchr'])/(( df_3q['tcdr']+273.15)*(df_3q['tchr']+273.15))
        div3= (df_3q['ton']*3510/(df_3q['tcdr']+273.15))-beta2*(df_3q['ton']*df_3q['ton']*3510*3510 )/(( df_3q['tcdr']+273.15)*( df_3q['tchr']+273.15))
        ab2= (df_2q['ton']*3510/(df_2q['tchr']+273.15))+beta0+beta1*( df_2q['tcdr'] -  df_2q['tchr'])/(( df_2q['tcdr']+273.15)*(df_2q['tchr']+273.15))
        div2= (df_2q['ton']*3510/(df_2q['tcdr']+273.15))-beta2*(df_2q['ton']*df_2q['ton']*3510*3510 )/(( df_2q['tcdr']+273.15)*( df_2q['tchr']+273.15))
        ab1= (df_1q['ton']*3510/(df_1q['tchr']+273.15))+beta0+beta1*( df_1q['tcdr'] -  df_1q['tchr'])/(( df_1q['tcdr']+273.15)*(df_1q['tchr']+273.15))
        div1= (df_1q['ton']*3510/(df_1q['tcdr']+273.15))-beta2*(df_1q['ton']*df_1q['ton']*3510*3510 )/(( df_1q['tcdr']+273.15)*( df_1q['tchr']+273.15))

        # Make predictions
        # kw =ton *3.51* (ab-div)/div
        predicted_kw_values = 3.510*chiller_df_predict_cleaned['ton']*(ab-div)/div

        # Make predictions
        predicted_kW4 = 3.510*df_4q['ton']*(ab4-div4)/div4
        predicted_kW3 = 3.510*df_3q['ton']*(ab3-div3)/div3
        predicted_kW2 = 3.510*df_2q['ton']*(ab2-div2)/div2
        predicted_kW1 = 3.510*df_1q['ton']*(ab1-div1)/div1
        kwpt1=predicted_kW1/(0.25*cap)
        kwpt2=predicted_kW2/(0.5*cap)
        kwpt3=predicted_kW3/(0.75*cap)
        kwpt4=predicted_kW4/(cap)
        iplv_value= 0.11*kwpt1+0.4*kwpt2+.23*kwpt3+0.25*kwpt4
        diff1= 100*(kwpt4-rkwpton)/rkwpton
        diff2= 100*(iplv_value-iplv)/iplv
        results["kpi"][chiller_name] = {
          "พิกัดทำความเย็น":cap,
          "พิกัดสมรรถนะ จากผู้ผลิต (kW/ton)":f"{rkwpton:.2f}",
          "พิกัด IPLV จากผู้ผลิต (kW/ton)":f"{iplv:.2f}",
          "ค่าสมรรถนะที่ประเมินจากโมเดลที่ Full Load (kW/ton)(%diff)":f"{float(kwpt4):.2f} ({float(diff1):.2f}"+" %)",
          "ค่า IPLV ที่ประเมินจากโมเดล (%diff)":f"{float(iplv_value):.2f} ({float(diff2):.2f}"+" %)"
        }

    # --- 3. Anomaly Detection and Graph Data Preparation for all chillers ---
    ANOMALY_THRESHOLD_PERCENT = 0.10 # 10% deviation (can be adjusted)

    # Now process the combined predictions for graphing and anomaly detection
    # Ensure all columns are numeric for plotting, fill NaNs if necessary
    for col in ['kw', 'predicted_kw', 'ton', 'kw_per_ton']:
        if col not in all_chiller_predictions_df.columns:
            all_chiller_predictions_df[col] = np.nan # Add missing columns
        all_chiller_predictions_df[col] = pd.to_numeric(all_chiller_predictions_df[col], errors='coerce')

    for chiller_name in all_chiller_predictions_df['ch_name'].unique():
        chiller_df = all_chiller_predictions_df[all_chiller_predictions_df['ch_name'] == chiller_name].copy()

        if 'h' not in chiller_df.columns:
            chiller_df['h'] = range(1, len(chiller_df) + 1)
            st.warning(f"Hour column 'h' not found for {chiller_name}. Using row index for plotting.")

        # Prepare data for 24h graphs, including 'predicted_kw'
        results["chiller_graphs_data"][chiller_name] = {
            "hours": chiller_df['h'].tolist(),
            "kw": chiller_df['kw'].tolist(),
            "predicted_kw": chiller_df['predicted_kw'].tolist(),
            "ton": chiller_df['ton'].tolist(),
            "kw_per_ton": chiller_df['kw_per_ton'].tolist()
        }

        # Anomaly detection based on actual vs. predicted kW
        if 'kw' in chiller_df.columns and 'predicted_kw' in chiller_df.columns:
            for index, row in chiller_df.iterrows():
                actual_kw = row['kw']
                predicted_kw_val = row['predicted_kw']

                if pd.notna(actual_kw) and pd.notna(predicted_kw_val):
                    if abs(predicted_kw_val) > 1e-6: # Avoid division by zero
                        deviation = actual_kw - predicted_kw_val # Keep signed deviation
                        percentage_deviation = abs(deviation) / abs(predicted_kw_val)

                        if percentage_deviation > ANOMALY_THRESHOLD_PERCENT:
                            results["anomaly_results"].append({
                                "chiller": ch_name,
                                "hour": row['h'],
                                "metric": "kW (Actual vs. Predicted)",
                                "actual_value": f"{actual_kw:.1f}",
                                "predicted_value": f"{predicted_kw_val:.1f}",
                                "deviation": f"{deviation:.1f}",
                                "percentage_deviation": f"{percentage_deviation*100:.1f}%",
                                "comment": f"kW deviation from predicted: {percentage_deviation*100:.1f}%."
                            })
                    elif abs(actual_kw) > 1e-6: # Predicted is zero/near-zero but actual is significant
                         results["anomaly_results"].append({
                            "chiller": chiller_name,
                            "hour": row['h'],
                            "metric": "kW (Actual vs. Predicted)",
                            "actual_value": f"{actual_kw:.1f}",
                            "predicted_value": f"{predicted_kw_val:.1f}",
                            "deviation": f"{actual_kw:.1f}",
                            "percentage_deviation": "N/A (Predicted kW is zero)",
                            "comment": "Actual kW is significant but predicted kW is zero/near-zero."
                        })
                elif pd.notna(actual_kw) and pd.isna(predicted_kw_val):
                     results["anomaly_results"].append({
                            "chiller": ch_name,
                            "hour": row['h'],
                            "metric": "kW (Actual vs. Predicted)",
                            "actual_value": f"{actual_kw:.1f}",
                            "predicted_value": "N/A",
                            "deviation": "N/A",
                            "percentage_deviation": "N/A",
                            "comment": "Could not predict kW due to missing feature data for this hour."
                        })

    results["general_comments"] += "Review graphs for actual vs. predicted kW and anomaly table for deviations."
    return results

def run_isolation_forest_model(chspec, historical_data, current_data):
    """
    Implements a Random Forest Regressor model for kW prediction and anomaly detection.
    Trains individual Random Forest models for each chiller on historical_data (data1)
    and predicts for current_data (data3).
    Adds 'predicted_kw' column and flags anomalies based on prediction deviation.
    """
    st.info("Running Isolation Forest Model (Individual Chiller Models)...")
    results = {
        "chiller_graphs_data": {},
        "anomaly_results": [],
        "general_comments": "Random Forest model executed successfully. "
    }

    # Define features and target for the Random Forest model
    #feature_cols = ['ton', 'tchs', 'tchr', 'tcds', 'tcdr', 'fwch']
    #target_col = 'kw'

    # --- 0. Consolidate DataFrames for easier processing ---
    # Flatten historical_data (list of dicts) into a single DataFrame

    # Flatten current_data (list of dicts) into a single DataFrame
    all_current_records_flat = []
    data_for_prediction_source = current_data if current_data else historical_data # Prioritize current, fallback to historical
    for entry in data_for_prediction_source:
        ch_name = entry['name']
        for data_row in entry['data']:
            row_with_name = {"ch_name": ch_name}
            row_with_name.update(data_row)
            all_current_records_flat.append(row_with_name)
    df_current_flat = pd.DataFrame(all_current_records_flat)

    if df_current_flat.empty:
        results["general_comments"] += "No current data available for prediction."
        st.warning("No current data available for prediction. Please load data3 or ensure data1 is available.")
        return results

    # Ensure all relevant columns are numeric for both dataframes

    # Identify unique chillers from historical data (only train for chillers we have historical data for)
    unique_chillers = df_current_flat['ch_name'].unique()
    if len(unique_chillers) == 0:
        results["general_comments"] += "No unique chillers found in historical data."
        st.warning("No unique chillers found in historical data to train Random Forest models.")
        return results

    all_chiller_predictions_df = pd.DataFrame() # To store all predicted data for all chillers

    st.write(f"Training and predicting for {len(unique_chillers)} chillers...")

    for chiller_name in unique_chillers:
        st.write(f"--- Processing Chiller: {chiller_name} ---")

        # --- 1. Prepare Training Data for current chiller ---
        df_operational  = df_current_flat[df_current_flat['ch_name'] == chiller_name].copy()
        
        # Ensure required columns are numeric and handle missing values
        required_columns = ['kw', 'ton', 'tchs', 'tchr', 'tcds', 'tcdr', 'fwch'] # Added fwch as it's used for ton calc
        for col in required_columns:
            if col in df_operational.columns:
                df_operational[col] = pd.to_numeric(df_operational[col], errors='coerce')
            else:
                st.warning(f"Missing column for Isolation Forest: '{col}'. Filling with 0 or considering its impact.")
                df_operational[col] = 0 # Or handle as per your needs, e.g., drop rows or use mean/median

        # Drop rows where critical columns for IF are NaN after numeric conversion
        df_operational.dropna(subset=required_columns, inplace=True)

        if df_operational.empty:
            results["general_comments"] += "No valid numerical data after cleaning for Isolation Forest."
            return results

        # Select features for Isolation Forest
        X = df_operational[required_columns]

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply Isolation Forest
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        df_operational['anomaly'] = iso.fit_predict(X_scaled)
        df_operational['anomaly_label'] = df_operational['anomaly'].map({1: 'normal', -1: 'anomaly'})

        # Prepare data for 24h graphs (ensure 'h' column for plotting)
        if 'h' not in df_operational.columns:
            df_operational['h'] = range(1, len(df_operational) + 1) # Dummy hour if not present

        results["chiller_graphs_data"][chiller_name] = {
            "hours": df_operational['h'].tolist(),
            "kw": df_operational['kw'].tolist(),
            "ton": df_operational['ton'].tolist(),
            "kw_per_ton": df_operational['kw_per_ton'].tolist() if 'kw_per_ton' in df_operational.columns else [np.nan]*len(df_operational) # Ensure kw_per_ton is included
        }

        # Populate anomaly results
        anomalies_for_chiller = df_operational[df_operational['anomaly'] == -1]
        for index, row in anomalies_for_chiller.iterrows():
            results["anomaly_results"].append({
                "chiller": row['ch_name'],
                "hour": row['h'],
                "method": "Isolation Forest",
                #"actual_values": {col: f"{row[col]:.1f}" for col in required_columns if pd.notna(row[col])},# Show all values that contributed
                "Detected": f"Suspected : {', '.join([f'{col}={row[col]:.1f}' for col in required_columns if pd.notna(row[col])])}"
            })
        
    results["general_comments"] += "Review anomaly table for detected outliers across combined features."
    return results

def get_plant_data(chiller_operational_data):
    """
    Combines 'ton', 'kw', and 'kw_per_ton' data for all chillers
    from the provided operational data (e.g., st.session_state.data1 or data3)
    to calculate plant-level total kW, total Ton, and overall kW/Ton.

    Args:
        chiller_operational_data (list): A list of dictionaries, where each dictionary
                                         represents a chiller and has a 'name' key
                                         and a 'data' key (list of operational records).
                                         Example structure:
                                         [
                                             {"name": "chillerA", "data": [{"h": 1, "kw": 100, "ton": 80, "kw_per_ton": 1.25}, ...]},
                                             {"name": "chillerB", "data": [{"h": 1, "kw": 120, "ton": 90, "kw_per_ton": 1.33}, ...]},
                                         ]

    Returns:
        pandas.DataFrame: A DataFrame containing the combined plant-level data with
                          'h', 'plant_kw', 'plant_ton', and 'plant_kw_per_ton' columns.
                          Returns an empty DataFrame if no valid chiller data is provided.
    """
    if not chiller_operational_data:
        print("No chiller operational data provided to combine for plant data.")
        return pd.DataFrame()

    all_chiller_records = []
    # Iterate through each chiller's data
    for chiller_entry in chiller_operational_data:
        ch_name = chiller_entry.get('name')
        ch_data = chiller_entry.get('data', [])

        # Skip if chiller name or data is missing/empty
        if not ch_name or not ch_data:
            continue

        # Convert each chiller's list of dictionaries into a DataFrame
        df_chiller = pd.DataFrame(ch_data)

        # --- Data Cleaning and Preparation for Aggregation ---
        # Ensure 'h' (hour) column exists. If not, create a simple index.
        if 'h' not in df_chiller.columns:
            df_chiller['h'] = range(1, len(df_chiller) + 1)
            print(f"Warning: 'h' (hour) column not found for {ch_name}. Using row index as hour for this chiller.")

        # Ensure 'kw' and 'ton' columns exist and are numeric. Coerce errors to NaN.
        for col in ['kw', 'ton']:
            if col not in df_chiller.columns:
                df_chiller[col] = np.nan # Add column with NaN if missing
                print(f"Warning: '{col}' column not found for {ch_name}. Filling with NaN.")
            df_chiller[col] = pd.to_numeric(df_chiller[col], errors='coerce') # Convert to numeric

        # Add chiller name to each row before concatenating
        df_chiller['ch_name'] = ch_name
        all_chiller_records.append(df_chiller)

    # If no valid chiller records were found after processing, return empty DataFrame
    if not all_chiller_records:
        print("No valid chiller records found after initial processing for plant data.")
        return pd.DataFrame()

    # Concatenate all individual chiller DataFrames into one large DataFrame
    df_all_data = pd.concat(all_chiller_records, ignore_index=True)

    # Group by 'h' (hour) and sum the 'kw' and 'ton' values across all chillers
    plant_summary_df = df_all_data.groupby('h').agg(
        plant_kw=('kw', 'sum'),
        plant_ton=('ton', 'sum')
    ).reset_index()

    # Calculate plant_kw_per_ton.
    # Handle potential division by zero or very small 'plant_ton' values to avoid errors.
    plant_summary_df['plant_kw_per_ton'] = plant_summary_df.apply(
        lambda row: row['plant_kw'] / row['plant_ton']
        if pd.notna(row['plant_ton']) and abs(row['plant_ton']) > 1e-6 # Check for non-NaN and non-zero/near-zero ton
        else np.nan, # Assign NaN if ton is problematic
        axis=1
    )
     # Calculate statistics, skipping NaN values
    plant_stat = {
        "kw_min": plant_summary_df['plant_kw'].min() if 'plant_kw' in plant_summary_df.columns else np.nan,
        "kw_max": plant_summary_df['plant_kw'].max() if 'plant_kw' in plant_summary_df.columns else np.nan,
        "kw_avg": plant_summary_df['plant_kw'].mean() if 'plant_kw' in plant_summary_df.columns else np.nan,
        "ton_min": plant_summary_df['plant_ton'].min() if 'plant_ton' in plant_summary_df.columns else np.nan,
        "ton_max":plant_summary_df['plant_ton'].max() if 'plant_ton' in plant_summary_df.columns else np.nan,
        "ton_avg": plant_summary_df['plant_ton'].mean() if 'plant_ton' in plant_summary_df.columns else np.nan,
        "kw_per_ton_min": plant_summary_df['plant_kw_per_ton'].min() if 'plant_kw_per_ton' in plant_summary_df.columns else np.nan,
        "kw_per_ton_max": plant_summary_df['plant_kw_per_ton'].max() if 'plant_kw_per_ton' in plant_summary_df.columns else np.nan,
        "kw_per_ton_avg": plant_summary_df['plant_kw_per_ton'].mean() if 'plant_kw_per_ton' in plant_summary_df.columns else np.nan,
    }

    results=[plant_summary_df,plant_stat]    
    return results

def coolingperformance(df_raw_data):
    """
    Helper to convert a flat DataFrame (from Excel or CSV)
    into the nested list-of-dictionaries format:
    [{"name": "chX", "data": [...]}]
    Assumes df_raw_data has a 'ch_name' column.
    Calculates 'ton' and 'kw_per_ton'.
    """
    data_output = []
    if 'ch_name' not in df_raw_data.columns:
        raise ValueError("Input DataFrame must contain a 'ch_name' column.")

    for chiller_name in df_raw_data['ch_name'].unique():
        chiller_data_df = df_raw_data[df_raw_data['ch_name'] == chiller_name].copy() # Use .copy() to avoid SettingWithCopyWarning

        # --- NEW CALCULATIONS FOR 'ton' and 'kw_per_ton' ---
        # Calculate 'ton' if 'fwch', 'tchr', and 'tchs' columns exist
        if all(col in chiller_data_df.columns for col in ['fwch', 'tchr', 'tchs']):
            # Ensure calculations handle potential NaN values by converting to numeric, then filling NaN with 0 for calculation
            # Use .fillna(0) for calculations to prevent NaN results, or use .dropna() if rows with missing data should be excluded
            # For this formula, it's safer to ensure numeric types
            chiller_data_df['fwch'] = pd.to_numeric(chiller_data_df['fwch'], errors='coerce').fillna(0)
            chiller_data_df['tchr'] = pd.to_numeric(chiller_data_df['tchr'], errors='coerce').fillna(0)
            chiller_data_df['tchs'] = pd.to_numeric(chiller_data_df['tchs'], errors='coerce').fillna(0)

            # The formula is ton = fwch * (tchr - tchs) / 24
            chiller_data_df['ton'] = (chiller_data_df['fwch'] * (chiller_data_df['tchr'] - chiller_data_df['tchs'])*3) / 40

        # Calculate 'kw_per_ton' if 'kw' and 'ton' columns exist
        # 'ton' must be calculated first or already present
        if 'kw' in chiller_data_df.columns and 'ton' in chiller_data_df.columns:
            chiller_data_df['kw'] = pd.to_numeric(chiller_data_df['kw'], errors='coerce').fillna(0) # Ensure numeric
            # Avoid division by zero: if 'ton' is 0 or very small, kw_per_ton is effectively infinite or NaN
            chiller_data_df['kw_per_ton'] = chiller_data_df.apply(
                lambda row: row['kw'] / row['ton'] if pd.notna(row['ton']) and abs(row['ton']) > 1e-6 else pd.NA, axis=1
            )
        # --- END NEW CALCULATIONS ---

        chiller_data_list = chiller_data_df.drop(columns=['ch_name']).to_dict(orient='records')
        data_output.append({"name": chiller_name, "data": chiller_data_list})
    return data_output



# --- Streamlit App Configuration ---

st.set_page_config(layout="wide", page_title="Chiller System Analysis")
st.title(":red[KCP]:orange[X] Chiller System Analysis & Performance")

# --- Initialize Session State ---
if 'chspec' not in st.session_state:
    st.session_state.chspec, st.session_state.data1 = load_initial_excel_data()

if 'input_method' not in st.session_state:
    st.session_state.input_method = "Manual Key-in"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Random Forest"
if 'model_output' not in st.session_state:
    st.session_state.model_output = None
if 'csv_df_loaded_data1' not in st.session_state:
    st.session_state.csv_df_loaded_data1 = None
if 'data3' not in st.session_state:
    st.session_state.data3 = []
if 'csv_df_loaded_data3' not in st.session_state:
    st.session_state.csv_df_loaded_data3 = None


# --- Sidebar ---
try:
    st.sidebar.image("kcps-logo.png", use_container_width=False) # Add your logo here
except FileNotFoundError:
    st.sidebar.warning("KCPX Logo (kcpx_logo.png) not found. Please place it in the app directory.")

st.sidebar.header("App Controls")

# --- Data Input Choice for Historical Data ---
st.sidebar.subheader("1. Historical Data Input")
input_method = st.sidebar.radio(
    "Select how to input historical data:",
    ("Manual Key-in", "Upload Excel File"),
    key="input_method_radio"
)
st.session_state.input_method = input_method

# --- Model Selection ---
st.sidebar.subheader("2. Select Analysis Model")
selected_model = st.sidebar.selectbox(
    "Which model would you like to run?",
    ("Random Forest", "Gordon Ng"),
    key="model_select"
)
st.session_state.selected_model = selected_model

st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ using Streamlit")
st.sidebar.write("EnConLab ,KMUTT")

# --- Main Content Area ---

# --- NEW SECTION: About This App & Methodology ---
with st.expander("About This App & Methodology", expanded=True):
    st.markdown("""
    แอพลิเคชั่นนี้ช่วยวิเคราะห์ข้อมูลการทำงานของเครื่องทำนำ้เย็น ว่าเครื่องทำงานต่างจากสเปคหรือไม่ และในปัจจุบันยังทำงานปกติหรื่อไม่ การระบายความร้อนปกติหรือไม่ โดยใช้โมเดลด้านข้อมูลในการวิเคราะห์ สามารถกรอกข้อมูลหรือโหลดจาก Excel file

    **Key Capabilities:**
    * อ่านข้อมูล Logsheet และเปรียบเทียบกับสเปคว่าทำงานปกติหรือไม่ (rated capacity, rated power, IPLV).
    * อ่านข้อมูล Logsheet ในอดีตและเปรียบเทียบกับการทำงานปัจจุบันว่าปกติหรือไม่               
    * Upload and manage historical and current operational data in Excel (temperatures, flow, power).
    * Run analysis using different models to assess chiller performance.
    * Visualize key performance indicators (kW, Ton, kW/Ton) over time.
    * Identify potential anomalies or deviations in chiller operation.
    * Export all input and generated data for further use.

    **Methodology & Key Calculations:**
    * **Compare kW/ton of models and actual operation
    * **Check Anomaly data
    * **Check Cooling Tower efficiency
    
    **Analytical Models:**
    * **Random Forest Model:**
        This model employs a Random Forest algorithm, an ensemble machine learning technique. It is typically used for predictive analysis and anomaly detection by learning complex patterns from historical data to predict expected behavior and identify deviations from those patterns.
    * **Gordon Ng Model:**
        The Gordon Ng model (also known as the Gordon-Ng Thermodynamic Efficiency Model) is a physics-based approach for evaluating chiller performance. It provides a theoretical benchmark of efficiency based on thermodynamic principles and operating conditions, allowing for comparison against actual performance to identify degradation or opportunities for improvement.
    * **IsolationForest Model:**
        ตรวจหาค่าที่ผิดปกติด้วย Isolation Forest Model และแจ้งว่าเกิดที่ข้อมูลเวลาใด
                
    **Important Considerations:**
    * ความถูกต้องของผลวิเคราะห์ขึ้นกับคุณภาพของข้อมูล
    * การปรับตั้งการทำงานของระบบต้องดำเนินการโดยผู้ควบคุมที่มีความรู้ความเข้าใจ
    * การประเมินว่าข้อมูลผิดปกติ (Anomaly detection) ใช้เกณฑ์ตรวจจับค่าที่เบี่ยงเบนเกิน 10% 
    """)

st.header("Download Excel Template (option)")
with st.expander("Click the button below to download the Excel template file.", expanded=False):
    st.write("หากใช้งานในครั้งแรก และต้องการป้อนข้อมูลด้วยไฟล์ Excel แต่ไม่มีไฟล์ หรืออยากทดสอบการใช้งาน โหลดไฟล์ตัวอย่างตรงนี้ไปใช้ได้ โดยนำไปกรอกเป็นข้อมูลของท่าน แล้วโหลดไฟล์นี้เข้าระบบในหัวข้อถัดไป")
    # --- Replace with your actual GitHub raw URL ---
    # Example: "https://raw.githubusercontent.com/myuser/myproject/main/template.xlsx"
    github_template_url = "https://raw.githubusercontent.com/damrongbouyom/my_streamlit_app/main/template01.xlsx"


    # --- Custom CSS for a consistent button style ---
    st.markdown("""
    <style>
    .download-button-container {
        background-color:white;
        text-align: left; /* Center the button if desired */
        margin-top: 0px;
        margin-bottom: 0px;
    }

    .download-button {
    
        color: black !important; /* Ensures text color is white, !important overrides Streamlit's default link style */
        padding: 5px 5px;
        text-align: center;
        text-decoration: none; /* Remove underline from link */
        display: inline-block;
        font-size: 16px;
        margin: 5px 5px;
        cursor: pointer;
        border: solid;
        border-width: 1px;
        border-color: red;
        border-radius: 5px;
        transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth hover effects */
    }

    .download-button:hover {
        background-color: white;              /*#45a049; /* Slightly darker green on hover */
        font-color:red;
        transform: translateY(-2px); /* Slight lift effect */
    }

    /* If you have other buttons, you can define a general .app-button style */
    /* For instance:
    .app-button {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        cursor: pointer;
        border: none;
    }
    */
    </style>
    """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="download-button-container">
        <a href="{github_template_url}" download="template.xlsx">
            <button class="download-button">
                Download Excel Template
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- NEW SECTION: Load Previously Saved Data ---
st.header("Load Previously Saved Data (Shortcut)")
st.info("โหลดไฟล์ข้อมูลที่บันทึกข้อมูลสเปค (data0) ข้อมูลการทำงานที่ผ่านมา (data1) และข้อมูลการใช้งานปัจจุบันที่ต้องการตรวจสอบ (data2) จากไฟล์ Template หนเดียว ไม่ต้องป้อนทีละข้อ โหลดแล้วไปรันโมเดลข้อ 4 ได้เลย")
with st.form("load_saved_data_form"):
    uploaded_saved_file = st.file_uploader(
        "Upload your 'output_chiller_data.xlsx' or a similar saved Excel file",
        type=["xlsx"],
        key="load_saved_file_uploader"
    )
    load_saved_data_button = st.form_submit_button("Load All Saved Data")

    if load_saved_data_button:
        if uploaded_saved_file is not None:
            # Call the new helper function to load all data
            st.session_state.chspec, st.session_state.data1, st.session_state.data3 = load_all_from_output_excel(uploaded_saved_file)
            st.success("All data (Chiller Specs, Historical, Current) has been loaded from your file! Goto No.4 to execute Models.")
            # Optional: st.rerun() to immediately update the UI, though often not strictly necessary if data is used in subsequent forms
            st.rerun()
        else:
            st.warning("Please upload a file to load.")

st.markdown("---") # Separator after the new section

st.header("1. Chiller Specifications")
st.info("Enter or edit chiller specifications. Click 'Confirm Specs' to apply changes.")
with st.form("chiller_specs_form"):
    # Using a standard button to add a row, which will then be reflected in the data_editor
    # but actual session_state update only happens on form submit.
    if st.form_submit_button("Add New Chiller Spec Row", help="Click this to add an empty row, then edit and 'Confirm Specs'"):
        st.session_state.chspec.append({"name": "", "type": "", "cap": 0, "rkw": 0, "iplv": 0.0})

    chspec_df_display = pd.DataFrame(st.session_state.chspec)
    edited_chspec_df = st.data_editor(
        chspec_df_display,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("Chiller Name", required=True),
            "type": st.column_config.SelectboxColumn("Type", options=["Water Coolled-screw","Water Coolled-centrifute","Water Coolled-VSD","Water Cooled-Magnetic", "Air Cooled-standard","Air Cooled-VSD", "Other"]),
             "cap": st.column_config.NumberColumn("Capacity (TR)", format="%d"),
            "rkw": st.column_config.NumberColumn("Rated kW", format="%d"),
            "iplv": st.column_config.NumberColumn("IPLV", format="%.2f")
        },
        key="chiller_specs_editor",
        height=200 # Adjust height as needed
    )
    confirm_specs_button = st.form_submit_button("Confirm Chiller Specs")
    if confirm_specs_button:
        st.session_state.chspec = edited_chspec_df.to_dict(orient='records')
        st.success("Chiller specifications confirmed!")


st.markdown("---")
st.header("2.โหลดข้อมูลการทำงานปกติ Historical Operational Data (data1)")

# --- Conditional Data Input Section for Historical Data ---
if st.session_state.input_method == "Upload Excel File":
    st.info("Upload your historical operational data Excel. Click 'Confirm Upload' to process. Data should be on the first sheet or named 'Chiller_Data_Historical'.")
    with st.form("historical_data_excel_form"):
        uploaded_file_data1 = st.file_uploader("Upload historical data Excel file", type=["xlsx"], key="uploader_data1_in_form")
        confirm_upload_data1_button = st.form_submit_button("Confirm Historical Data Upload")

        if confirm_upload_data1_button:
            if uploaded_file_data1 is not None:
                try:
                    # Try to read the specific sheet from the output file first, then fallback to first sheet
                    try:
                        df_uploaded_data1 = pd.read_excel(uploaded_file_data1, sheet_name='Chiller_Data_Historical')
                        st.info("Reading 'Chiller_Data_Historical' sheet from the uploaded Excel.")
                    except ValueError:
                        df_uploaded_data1 = pd.read_excel(uploaded_file_data1, sheet_name=0)
                        st.warning("Could not find 'Chiller_Data_Historical' sheet. Reading the first sheet instead.")

                    st.session_state.csv_df_loaded_data1 = df_uploaded_data1 # Store the raw DataFrame
                    st.success("Historical Excel file uploaded and processed successfully!")
                    st.dataframe(df_uploaded_data1.head()) # Show preview after processing

                    # Process and update data1 from Excel
                    st.session_state.data1 = _process_raw_data_df_to_data_structure(df_uploaded_data1)
                except pd.errors.EmptyDataError:
                    st.error("The uploaded Excel file is empty.")
                    st.session_state.csv_df_loaded_data1 = None
                    st.session_state.data1 = []
                except ValueError as ve: # Catch the specific ValueError from _process...
                    st.error(f"Error processing Excel: {ve}")
                    st.session_state.csv_df_loaded_data1 = None
                    st.session_state.data1 = []
                except Exception as e:
                    st.error(f"An unexpected error occurred reading Excel file: {e}")
                    st.session_state.csv_df_loaded_data1 = None
                    st.session_state.data1 = []
            else:
                st.warning("Please select an Excel file to upload before confirming.")
        elif st.session_state.csv_df_loaded_data1 is not None:
             st.info("Previously uploaded historical Excel data:")
             st.dataframe(st.session_state.csv_df_loaded_data1.head())
        else:
            st.info("No historical Excel uploaded yet. Upload a file and click 'Confirm Upload'.")


elif st.session_state.input_method == "Manual Key-in":
    st.info("Manually enter or edit historical operational data. Click 'Confirm Historical Data' to apply changes.")

    tab_titles_data1 = [spec['name'] for spec in st.session_state.chspec]

    with st.form("historical_data_manual_form"):
        if tab_titles_data1:
            tabs_data1 = st.tabs(tab_titles_data1)
            edited_dfs_for_data1 = {} # Use a dictionary to store the edited dataframes from each tab temporarily

            for i, tab in enumerate(tabs_data1):
                chiller_name = tab_titles_data1[i]
                with tab:
                    st.subheader(f"Historical Data for {chiller_name}")

                    chiller_data_entry = next((item for item in st.session_state.data1 if item["name"] == chiller_name), None)
                    if chiller_data_entry:
                        current_chiller_data_list = chiller_data_entry['data']
                    else:
                        current_chiller_data_list = []
                        # If this chiller name is new, add an empty entry to data1
                        st.session_state.data1.append({"name": chiller_name, "data": []})
                        chiller_data_entry = st.session_state.data1[-1]

                    chiller_data_df = pd.DataFrame(current_chiller_data_list)
                    if chiller_data_df.empty:
                        empty_data_schema = {
                            "h": pd.Series(dtype='int'), "tchs": pd.Series(dtype='float'), "tchr": pd.Series(dtype='float'),
                            "tcds": pd.Series(dtype='float'), "tcdr": pd.Series(dtype='float'), "fwch": pd.Series(dtype='float'),
                            "kw": pd.Series(dtype='float'), "trfe": pd.Series(dtype='float'), "trfc": pd.Series(dtype='float'),
                        }
                        chiller_data_df = pd.DataFrame(empty_data_schema)

                    edited_chiller_data_df = st.data_editor(
                        chiller_data_df,
                        num_rows="dynamic",
                        key=f"chiller_data_editor_data1_{chiller_name}", # Unique key per tab
                        column_config={
                            "h": st.column_config.NumberColumn("Hour", format="%d", required=True),
                            "tchs": st.column_config.NumberColumn("TCHS (°C)", format="%.1f"),
                            "tchr": st.column_config.NumberColumn("TCHR (°C)", format="%.1f"),
                            "tcds": st.column_config.NumberColumn("TCDS (°C)", format="%.1f"),
                            "tcdr": st.column_config.NumberColumn("TCDR (°C)", format="%.1f"),
                            "fwch": st.column_config.NumberColumn("FWCH (m³/h)", format="%.1f"),
                            "kw": st.column_config.NumberColumn("kW", format="%.1f"),
                            "trfe": st.column_config.NumberColumn("TRfe (Ton)", format="%.1f"),
                            "trfc": st.column_config.NumberColumn("TRfc (°C)", format="%.1f"),
                            "ton": st.column_config.NumberColumn("Ton Calculated", format="%.1f", disabled=True), # Display calculated Ton
                            "kw_per_ton": st.column_config.NumberColumn("kW/Ton Calculated", format="%.2f", disabled=True), # Display calculated kW/Ton
                        }
                    )
                    edited_dfs_for_data1[chiller_name] = edited_chiller_data_df
        else:
            st.info("Please add chiller specifications in section 1 to enable historical data entry for each chiller.")

        confirm_manual_data1_button = st.form_submit_button("Confirm Historical Data")

        if confirm_manual_data1_button:
            updated_data1_list = []
            for chiller_name in tab_titles_data1:
                if chiller_name in edited_dfs_for_data1:
                    temp_df = edited_dfs_for_data1[chiller_name]
                    if not temp_df.empty:
                        temp_df_with_name = temp_df.copy()
                        temp_df_with_name['ch_name'] = chiller_name
                        processed_chiller_data = _process_raw_data_df_to_data_structure(temp_df_with_name)
                        updated_data1_list.append({"name": chiller_name, "data": processed_chiller_data[0]['data']})
                    else:
                        updated_data1_list.append({"name": chiller_name, "data": []})

            st.session_state.data1 = updated_data1_list
            st.success("Historical operational data confirmed!")


st.markdown("---")
st.header("3.โหลดข้อมูลการทำงานปัจจุบัน Current Operational Data (data2)")
st.info("Upload current operational data (e.g., today's or last 24 hours). Click 'Confirm Upload' to process. Data should be on the first sheet or named 'Chiller_Data_Current'.")

with st.form("current_data_excel_form"):
    uploaded_file_data3 = st.file_uploader("Upload current operational data Excel file", type=["xlsx"], key="uploader_data3_in_form")
    confirm_upload_data3_button = st.form_submit_button("Confirm Current Data Upload")

    if confirm_upload_data3_button:
        if uploaded_file_data3 is not None:
            try:
                # Try to read the specific sheet from the output file first, then fallback to first sheet
                try:
                    df_uploaded_data3 = pd.read_excel(uploaded_file_data3, sheet_name='Chiller_Data_Current')
                    st.info("Reading 'Chiller_Data_Current' sheet from the uploaded Excel.")
                except ValueError: # Sheet not found
                    df_uploaded_data3 = pd.read_excel(uploaded_file_data3, sheet_name=0)
                    st.warning("Could not find 'Chiller_Data_Current' sheet. Reading the first sheet instead.")

                st.session_state.csv_df_loaded_data3 = df_uploaded_data3 # Store the raw DataFrame
                st.success("Current operational data Excel file uploaded and processed successfully!")
                st.dataframe(df_uploaded_data3.head()) # Show preview after processing

                # Process and update data3 from Excel
                st.session_state.data3 = _process_raw_data_df_to_data_structure(df_uploaded_data3)
            except pd.errors.EmptyDataError:
                st.error("The uploaded Excel file is empty.")
                st.session_state.csv_df_loaded_data3 = None
                st.session_state.data3 = []
            except ValueError as ve: # Catch the specific ValueError from _process...
                st.error(f"Error processing Excel: {ve}")
                st.session_state.csv_df_loaded_data3 = None
                st.session_state.data3 = []
            except Exception as e:
                st.error(f"An unexpected error occurred reading Excel file: {e}")
                st.session_state.csv_df_loaded_data3 = None
                st.session_state.data3 = []
        else:
            st.warning("Please select an Excel file to upload before confirming.")
    elif st.session_state.csv_df_loaded_data3 is not None:
        st.info("Previously uploaded current operational data:")
        st.dataframe(st.session_state.csv_df_loaded_data3.head())
    else:
        st.info("No current operational data uploaded yet. Upload a file and click 'Confirm Upload'.")


st.markdown("---")
st.header("Check Points:ตรวจข้อมูลที่ป้อน")
st.info("Click the buttons below to review the currently loaded Chiller Specifications and Operational Data.")

col1, col2, col3 = st.columns(3) # Added a third column for data3

with col1:
    if st.button("Show Chiller Specs (data0)", key="show_specs_btn"):
        if st.session_state.chspec:
            st.subheader("Current Chiller Specifications")
            st.dataframe(pd.DataFrame(st.session_state.chspec))
        else:
            st.warning("No chiller specifications loaded yet.")

with col2:
    if st.button("Show Historical Data (data1)", key="show_data1_btn"):
        if st.session_state.data1:
            st.subheader("Current Historical Data (data1)")
            all_data1_rows = []
            for entry in st.session_state.data1:
                ch_name = entry['name']
                for data_row in entry['data']:
                    row = {"ch_name": ch_name}
                    row.update(data_row)
                    all_data1_rows.append(row)
            if all_data1_rows:
                st.dataframe(pd.DataFrame(all_data1_rows))
            else:
                st.warning("Historical data (data1) is loaded but appears empty.")
        else:
            st.warning("No historical data (data1) loaded yet.")

with col3:
    if st.button("Show Current Data (data2)", key="show_data3_btn"):
        if st.session_state.data3:
            st.subheader("Current Operational Data (data3)")
            all_data3_rows = []
            for entry in st.session_state.data3:
                ch_name = entry['name']
                for data_row in entry['data']:
                    row = {"ch_name": ch_name}
                    row.update(data_row)
                    all_data3_rows.append(row)
            if all_data3_rows:
                st.dataframe(pd.DataFrame(all_data3_rows))
            else:
                st.warning("Current operational data (data2) is loaded but appears empty.")
        else:
            st.warning("No current operational data (data2) loaded yet.")


st.markdown("---")
st.header("4.Execute Model")
st.write("พร้อมรันโมเดลที่คุณเลือก")
if st.button("Execute Model", key="execute_model_btn", help="Run the selected model with the current data."):
    if not st.session_state.chspec:
        st.warning("Please define chiller specifications (Section 1) and 'Confirm Specs' before executing the model.")
    elif not st.session_state.data1 and not st.session_state.data3:
        st.warning("Please provide operational data (historical in Section 2, or current in Section 3) and confirm it before executing the model.")
    else:
        st.info(f"Executing {st.session_state.selected_model} model...")
        st.session_state.model_output = None # Clear previous output

        data_for_model = st.session_state.data3 if st.session_state.data3 else st.session_state.data1

        if not data_for_model:
             st.warning("No valid operational data available to run the model.")
        else:
            # Check if data_for_model has actual data dictionaries within channels
            is_data_valid = False
            for channel in data_for_model:
                if channel['data']:
                    is_data_valid = True
                    break

            if not is_data_valid:
                st.warning("Operational data seems empty. Please ensure data is correctly loaded or entered and confirmed.")
            else:
                with st.spinner(f"Running {st.session_state.selected_model}..."):
                    if st.session_state.selected_model == "Random Forest":
                        st.session_state.model_output = run_random_forest_model(st.session_state.chspec, st.session_state.data1, st.session_state.data3)
                    elif st.session_state.selected_model == "Gordon Ng":
                        st.session_state.model_output = run_gordon_ng_model(st.session_state.chspec, st.session_state.data1, st.session_state.data3)
st.markdown("---")
st.header("5. Model Output & Analysis Results")

if st.session_state.model_output:
    st.write(st.session_state.model_output.get("general_comments", "No general comments provided."))
    st.subheader("24-Hour Performance Graphs")
    chiller_graphs_data = st.session_state.model_output.get("chiller_graphs_data", {})
    if chiller_graphs_data:
        for ch_name, data in chiller_graphs_data.items():
            if data and data.get('hours'): # Ensure there's data to plot
                st.markdown(f"#### Chiller: {ch_name}")
                df_plot = pd.DataFrame({
                    'Hour': data['hours'],
                    'kW': data['kw'],
                    'Predicted_kW': data['predicted_kw'],         
                    'Ton': data['ton'],
                    'kW/Ton': data['kw_per_ton']
                })

                # Create separate figures for each metric
                fig_combined = go.Figure()
                # Add kW trace (primary Y-axis)
                fig_combined.add_trace(
                    go.Scatter(x=df_plot['Hour'], y=df_plot['kW'], mode='lines+markers', name='Power (kW)', yaxis='y1'),
                )

                # Add Predicted kW trace (primary Y-axis)
                fig_combined.add_trace(
                    go.Scatter(x=df_plot['Hour'], y=df_plot['Predicted_kW'], mode='lines+markers', name='Predicted_kW', yaxis='y1'),
                )
                # Add Ton trace (primary Y-axis)
                fig_combined.add_trace(
                    go.Scatter(x=df_plot['Hour'], y=df_plot['Ton'], mode='lines+markers', name='Ton', yaxis='y2'),
                )

                # Update layout for dual Y-axes
                fig_combined.update_layout(
                    title=f'{ch_name} - Power Consumption (kW) over 24h',
                    xaxis_title='Hour',
                    yaxis=dict(
                        title='Power (kW)',
                        #titlefont=dict(color='blue'),
                        tickfont=dict(color='blue')
                    ),
                    yaxis2=dict(
                        title='Ton',
                        #titlefont=dict(color='red'),
                        tickfont=dict(color='red'),
                        overlaying='y', # Overlay on the primary y-axis
                        side='right'    # Place on the right side
                    ),
                    legend=dict(x=0.01, y=0.99), # Adjust legend position
                    hovermode="x unified" # For better hover experience
                )
                st.plotly_chart(fig_combined, use_container_width=True)

                fig_kwh_ton = px.line(df_plot, x='Hour', y='kW/Ton', title=f'{ch_name} - Efficiency (kW/Ton) over 24h', markers=True)
                fig_kwh_ton.update_layout(hovermode="x unified")
                st.plotly_chart(fig_kwh_ton, use_container_width=True)

        # Graph of Total plant
        df1=get_plant_data(st.session_state.data3)
        df_t=df1[0]
        statov_cur=df1[1]
        df2=get_plant_data(st.session_state.data1)   
        statov_his=df2[1]     
        # Prepare data for 24h graphs
        total= {
            "hours": df_t['h'].tolist(),
            "kw": df_t['plant_kw'].tolist() if 'plant_kw' in df_t.columns else [0]*len(df_t),
            "ton": df_t['plant_ton'].tolist() if 'plant_ton' in df_t.columns else [0]*len(df_t), # Use the new 'ton'
            "kw_per_ton": df_t['plant_kw_per_ton'].tolist() if 'plant_kw_per_ton' in df_t.columns else [0]*len(df_t)
        }

        if total : # Ensure there's data to plot
            df_plot = pd.DataFrame({
                'Hour': total['hours'],
                'kW': total['kw'],
                'Ton': total['ton'],
                'kW/Ton': total['kw_per_ton']
            })

            # --- NEW: Combined kW and Ton Graph ---
            fig_combined = go.Figure()

            # Add kW trace (primary Y-axis)
            fig_combined.add_trace(
                go.Scatter(x=df_plot['Hour'], y=df_plot['kW'], mode='lines+markers', name='Power (kW)', yaxis='y1'),
            )

            # Add Ton trace (primary Y-axis)
            fig_combined.add_trace(
                go.Scatter(x=df_plot['Hour'], y=df_plot['Ton'], mode='lines+markers', name='Ton', yaxis='y1'),
            )
            # Add Ton trace (primary Y-axis)
            fig_combined.add_trace(
                go.Scatter(x=df_plot['Hour'], y=df_plot['kW/Ton'], mode='lines+markers', name='kW/Ton', yaxis='y2'),
            )

            # Update layout for dual Y-axes
            fig_combined.update_layout(
                title='Total Pant Operation (Power,Ton,kW/Ton)',
                xaxis_title='Hour',
                yaxis=dict(
                    title='Power (kW)',
                    #titlefont=dict(color='blue'),
                    tickfont=dict(color='blue')
                ),
                yaxis2=dict(
                    title='kW/Ton',
                    #titlefont=dict(color='red'),
                    tickfont=dict(color='red'),
                    overlaying='y', # Overlay on the primary y-axis
                    side='right'    # Place on the right side
                ),
                legend=dict(x=0.01, y=0.99), # Adjust legend position
                hovermode="x unified" # For better hover experience
            )
            st.plotly_chart(fig_combined, use_container_width=True)
    else:
        st.info("No graph data available. Run the model to generate graphs.")
    
    st.subheader("General Comments")
    ##prepare data for comments
     #Table 1
    a=statov_cur["ton_avg"]
    b=statov_cur["ton_min"]
    c=statov_cur["ton_max"]
    d=statov_cur["kw_avg"]
    e=statov_cur["kw_min"]
    f=statov_cur["kw_max"]
    i=statov_cur["kw_per_ton_avg"]
    j=statov_cur["kw_per_ton_min"]
    k=statov_cur["kw_per_ton_max"]
    a1=statov_his["ton_avg"]
    b1=statov_his["ton_min"]
    c1=statov_his["ton_max"]
    d1=statov_his["kw_avg"]
    e1=statov_his["kw_min"]
    f1=statov_his["kw_max"]
    i1=statov_his["kw_per_ton_avg"]
    j1=statov_his["kw_per_ton_min"]
    k1=statov_his["kw_per_ton_max"]

    datastructure =pd.DataFrame({"Current run":{"Load":f" {a:.1f} Ton (min {b:.1f}- max {c:.1f})","Power(kW)":f" {d:.1f} kW (min {e:.1f}- max {f:.1f})","Performance":f" {(d/a):.2f} kW/ton (min {j:.2f}- max {k:.2f})"},"Previous run":{"Load":f" {a1:.1f} Ton (min {b1:.1f}- max {c1:.1f})","Power(kW)":f" {d1:.1f} kW (min {e1:.1f}- max {f1:.1f})","Performance":f" {(d1/a1):.2f} kW/ton (min {j1:.2f}- max {k1:.2f})"}})
    # Center align all text
    styled_df = datastructure.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    #Table 2
    data_b=st.session_state.model_output.get("stat_cur")
    datastructure2=pd.DataFrame(data_b)

    # Center align all text
    styled_df2 = datastructure2.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    #Table 3
    data_c=st.session_state.model_output.get("stat_his")
    datastructure3=pd.DataFrame(data_c)
    # Center align all text
    styled_df3 = datastructure3.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )

 # 1 #111111111111111111111111111111111111111111111111111111111111111111111111111111111111
    st.info("1.สรุปภาพรวมการทำงาน")
    #condition check
    #load compare
    str1=""
    str2=""
    str3=""
    strname=""
    str4=""
    if a/a1 < 0.90 :
        str1=" ภาระปรับอากาศลดลง"
    elif a/a1 <1.05 :
        str1=" ภาระปรับอากาศใกล้เคียงเดิม"
    elif a/a1 <1.1 :
         str1=" ภาระปรับอากาศเพิ่มขึ้นเล็กน้อย"
    else:
        str1=" ภาระปรับอากาศเพิ่มขึ้นมาก" 
    # kw/ton compare
    if d*a1/(d1*a) < 0.90 :
        str2=" และสมรรถนะดีขึ้น"
    elif d*a1/(d1*a) <1.05 :
        str2=" และสมรรถนะใกล้เคียงเดิม"
    elif d*a1/(d1*a) <1.1 :
         str2=" และสมรรถนะตกเล็กน้อย อาจเนื่องจากสภาวะการทำงานที่ต่างกัน"
    else:
        str1=" และสมรรถนะตกลง ควรตรวจสอบสาเหตุ"     
    # tchs
    flag1=0
    tcdsum=0
    n=0
    for chiller_name,inner_dict in data_c.items():
        tchvalue= data_c[chiller_name]["Temp(CHS)"].split(" ",1)
        if float(tchvalue[0]) <7:
            str3=str3+ tchvalue[0]+ " C  "
            strname=strname+chiller_name+" "
            flag1=1
        tcdvalue= data_c[chiller_name]["Temp(CDR)"].split(" ",1)      
        if float(tcdvalue[0]) > 32:
            str4= "อุณหภูมินำ้ระบายความร้อนสูงมาก ควรแก้ไข"  
        tcdsum=tcdsum+float(tcdvalue[0])
        n=n+1
    tcd_avg=tcdsum/n
    str4="อุณหภูมินำ้ระบายความร้อนเฉลี่ยเท่ากับ "+f"**{tcd_avg:.2f}**"+" C"+str4
    if flag1==1:
        str3= "อุณหภูมินำ้เย็นของเครื่องทำนำ้เย็นบางชุดตำ่ เช่น "+str3+" สามารถประหยัดพลังงาน โดยปรับอุณหภูมินำ้เย็นชุด "+strname+ " สูงขึ้นได้"
    else:
        str3= "การตั้งค่าอุณหภูมินำ้เย็นอยู่ในช่วงปกติ"
    
    st.write(f"ภาระปรับอากาศเฉลี่ย **{a:.1f}** ตัน ใช้ไฟฟ้าเฉลี่ย **{d:.1f}** kW และสมรรถนะ **{(d/a):.2f}** kW/ton"+str1+str2+"  "+str3+"  "+str4)
    st.write("สรุปข้อมูลการทำงานในภาพรวม และเครื่องทำนำ้เย็นแต่ละชุดแสดงได้ดังนี้")
    #Table 1
    # Render as HTML
    st.write(f"**ภาพรวม**")
    st.write(styled_df.to_html(), unsafe_allow_html=True)
    st.write(f"**เครื่องทำนำ้เย็น**")  
    st.write("การทำงานปัจจุบัน (Current)") 
    #Table 2
    # Render as HTML
    st.write(styled_df2.to_html(), unsafe_allow_html=True)
    #Table 3
    st.write("การทำงานที่ผ่านมา (Previous)")
    # Render as HTML    
    st.write(styled_df3.to_html(), unsafe_allow_html=True)

 # 2 #2222222222222222222222222222222222222222222222222222222222222222222222222222222222222
    st.info("2.เปรียบเทียบค่า Spec (data0) กับ ข้อมูลการใช้งาน (data1)")
    df_kpi=st.session_state.model_output.get("kpi")
    data_kpi=pd.DataFrame(df_kpi)
    chkwpt_nok=[]
    chiplv_nok=[]
    str8="การเปรียบเทียบค่าสมรรถนะ และค่า IPLV ของผู้ผลิตกับค่าที่ประเมินจากการทำงานพบว่า สำหรับค่าสมรรถนะ "
    str9="สำหรับค่า IPLV"


    for key,value in df_kpi.items():
        lrkwpt = float(df_kpi[key]["พิกัดสมรรถนะ จากผู้ผลิต (kW/ton)"]) 
        lriplv = float(df_kpi[key]["พิกัด IPLV จากผู้ผลิต (kW/ton)"])
        lkwpt = (df_kpi[key]["ค่าสมรรถนะที่ประเมินจากโมเดลที่ Full Load (kW/ton)(%diff)"]).split(" ",1)        
        liplv = df_kpi[key]["ค่า IPLV ที่ประเมินจากโมเดล (%diff)"].split(" ",1)  
        
        difkwpt =100* (float(lkwpt[0])-lrkwpt)/lrkwpt
        difiplv= 100*(float(liplv[0])-lriplv)/ lriplv
        if difkwpt> 10 or difkwpt < -10 :
            str8=str8+f" เครื่องทำนำ้เย็น {key} ค่าสมรรถนะที่ full load ที่ประเมินสูงกว่าสเปคมาก (diff  {difkwpt:.2f} %) ควรตรวจสอบ" 
            chkwpt_nok.append(key)
        elif difkwpt>5:
            str8=str8+f" เครื่องทำนำ้เย็น {key} ค่าสมรรถนะที่ full load ที่ประเมินสูงกว่าสเปคเล็กน้อย (diff  {difkwpt:.2f} %)" 
        elif difkwpt>-5:
            str8=str8+f" เครื่องทำนำ้เย็น {key} ค่าสมรรถนะที่ full load ที่ประเมินใกล้เคียงสเปคมาก (diff  {difkwpt:.2f} %)" 
        else:
            str8=str8+f" เครื่องทำนำ้เย็น {key} ค่าสมรรถนะที่ full load ที่ประเมินตำ่กว่าสเปค (diff  {difkwpt:.2f} %) อาจตรวจสอบ" 
        
        if difiplv> 10 or difiplv< -10:
            str9=str9+f" เครื่องทำนำ้เย็น {key} ค่า IPLV ที่ประเมินสูงกว่าสเปคพอสมควร (diff  {difkwpt:.2f} %) " 
            chiplv_nok.append(key)
        else :
            str9=str9+f" เครื่องทำนำ้เย็น {key} ค่า IPLV ที่ประเมินใกล้เคียงสเปค (diff  {difkwpt:.2f} %) " 

    if len(chkwpt_nok) >0:
        str8=str8+f" เครื่องทำนำ้เย็นที่มีค่า สมรรถนะแตกต่างเกิน 10% ได้แก่ {chkwpt_nok}"
    if len(chiplv_nok) >0:
        str9=str9+f" เครื่องทำนำ้เย็นที่มีค่า IPLV แตกต่างเกิน 10% ได้แก่ {chkwpt_nok}"
    st.write(str8)
    st.write(str9)

    # Center align all text
    styled_kpi = data_kpi.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    # Render as HTML    
    st.write(styled_kpi.to_html(), unsafe_allow_html=True)

 # 3 #3333333333333333333333333333333333333333333333333333333333333333333333333333333333333   
    st.info("3.เปรียบเทียบค่า ข้อมูลปัจจุบัน (data2) กับ ข้อมูลการทำงานปกติ (data1)")
    n=0
    sum=0
    sump=0
    df_model={}
    str6=" หากเปรียบเทียบการใช้พลังงานปัจจุบัน (data2) กับโมเดลที่จำลองให้เครื่องในช่วงปกติ (data1)มาทำงานในสภาวะเดียวกับปัจจุบัน พบว่า"
    str06=""
    for chiller_name,inner_dict in chiller_graphs_data.items():
        inner1=inner_dict["kw"]
        inner2=inner_dict["predicted_kw"]
        for value in inner1:
            n=n+1
            sum=sum+value
        for value in inner2:
            sump=sump+value
        skw=sum/n
        spkw=sump/n
        dif=100*(skw-spkw)/spkw
        df_model[chiller_name]={"Actual kW":f"{skw:.1f}","Model kW":f"{spkw:.1f}","%Diviation":f"{dif:.2f}"}
        if dif >10:
            str06=f"  เครื่องทำนำ้เย็น {chiller_name} มีการใช้พลังงานมากกว่าโมเดลถึงร้อยละ {dif:.2f}"
        elif dif >5:
            str06=f"  เครื่องทำนำ้เย็น {chiller_name} มีการใช้พลังงานมากกว่าโมเดล ร้อยละ {dif:.2f}"
        elif dif >0:
            str06=f"  เครื่องทำนำ้เย็น {chiller_name} มีการใช้พลังงานมากกว่าโมเดลเล็กน้อย ร้อยละ {dif:.2f}"
        elif dif >-5:
          str06=f"  เครื่องทำนำ้เย็น {chiller_name} มีการใช้พลังงานน้อยกว่าโมเดลเล็กน้อย ร้อยละ {dif:.2f}"  
        else:
          str06=f"  เครื่องทำนำ้เย็น {chiller_name} มีการใช้พลังงานน้อยกว่าโมเดลถึงร้อยละ {dif:.2f}"
        str6=str6+str06
    st.write(str6)
    datastructure4=pd.DataFrame(df_model)
    # Center align all text
    styled_df4 = datastructure4.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center'),('width','5cm')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    # Render as HTML    
    st.write(styled_df4.to_html(), unsafe_allow_html=True)

 # 4 #4444444444444444444444444444444444444444444444444444444444444444444444444444444444444
    st.info("4.สมรรถนะการระบายความร้อน(Cooling Tower Plant Performance)")

    aplist={}
    eflist={}
    apmean={}
    efmean={}
    hourlist={}
    list_ch_ap=[]
    list_ch_ef=[]
    for set in st.session_state.data1:
        #st.write(set['data'])
        ap=[]
        ef=[]
        hour=[]
        sap=0
        sef=0
        n=0
        for set2 in set["data"]:
           n=n+1
           #st.write(set2)
           hour.append(set2["h"])
           ap.append(set2["Approach Temp"])
           ef.append(set2["Effectiveness"])
           sap=sap+set2["Approach Temp"]
           sef=sef+set2["Effectiveness"] 
        hourlist[set["name"]] = hour
        aplist[set["name"]] = ap
        eflist[set["name"]] = ef  
        apmean[set["name"]] = {"Approach temp (C)":f"{sap/n:.1f}"}
        efmean[set["name"]] = {"Effectiveness (%)":f"{sef/n:.1f}"}
        if (sap/n) > 3:
            list_ch_ap.append(set["name"])
        if (sef/n) < 60:
            list_ch_ef.append(set["name"])
    # --- NEW: Combined all Approach temp Graph ---
    fig_combined = go.Figure()
    for name1,value in aplist.items():
        # --- NEW: Combined kW and Ton Graph ---
        # Add kW trace (primary Y-axis)
        fig_combined.add_trace(
        go.Scatter(x=hourlist[name1], y=value, mode='lines+markers', name=name1, yaxis='y1'),
        )
    # Update layout for dual Y-axes
    fig_combined.update_layout(
        title='Approach Temperature (C)',
        xaxis_title='Hour',
        yaxis=dict(
            title='C',
            #titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        legend=dict(x=0.01, y=0.99), # Adjust legend position
        hovermode="x unified" # For better hover experience
    )
    st.plotly_chart(fig_combined, use_container_width=True)
    str6=""
    str7=""
    if len(list_ch_ap) == 0:
        str6="หอผึ่งนำ้ที่จ่ายนำ้ระบายความร้อนให้เครื่องทำนำ้เย็นทุกชุดมีอุณหภูมิเข้าถึงเป็นไปตามเกณฑ์ (<3 C)"
    else :
        str6=f"เครื่องทำนำ้เย็นบางชุดที่หอผึ่งนำ้มีอุณหภูมิสูงเกินกว่าเกณฑ์ที่เหมาะสม (> 3C) ได้แก่ {list_ch_ap}"

    if len(list_ch_ef) == 0:
        str7="หอผึ่งนำ้ที่จ่ายนำ้ระบายความร้อนให้เครื่องทำนำ้เย็นทุกชุดมีค่าประสิทธิผลเป็นไปตามเกณฑ์ (ุ60-75%)"
    else :
        str7=f"เครื่องทำนำ้เย็นบางชุดที่หอผึ่งนำ้มีประสิทธิผลตำ่ ( < 60%) ได้แก่ {list_ch_ef}"

    st.write("**อุณหภูมิเข้าถึง (Approach Temperature)**")
    st.write("อุณหภูมิเข้าถึงวัดความแตกต่างของอุณหภูมินำ้จากหอผึ่งนำ้กับอุณหภูมิกระเปาะเปียก (Wetbulb) ยิ่งสูงยิ่งหมายถึงมีปัญหา พบว่า"+str6)
    datastructure5=pd.DataFrame.from_dict(apmean, orient='index')
    # Center align all text
    styled_df4 = datastructure5.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center'),('width','5cm')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    # Render as HTML    
    st.write(styled_df4.to_html(), unsafe_allow_html=True)

    # --- NEW: Combined all Effectiveness Graph ---
    fig_combined = go.Figure()
    for name1,value in eflist.items():
        # --- NEW: Combined kW and Ton Graph ---
        # Add kW trace (primary Y-axis)
        fig_combined.add_trace(
        go.Scatter(x=hourlist[name1], y=value, mode='lines+markers', name=name1, yaxis='y1'),
        )
    # Update layout for dual Y-axes
    fig_combined.update_layout(
        title='Effectiveness of CT which supply CHW to Chiller (%) ',
        xaxis_title='Hour',
        yaxis=dict(
            title='C',
            #titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        legend=dict(x=0.01, y=0.99), # Adjust legend position
        hovermode="x unified" # For better hover experience
    )
    st.plotly_chart(fig_combined, use_container_width=True) 
    st.write("**ค่าประสิทธิผลการระบายความร้อน (Effectiveness)**")
    st.write("ค่าประสิทธิผลเป็นการเปรียบเทียบอุณหภูมินำ้ที่ลดลงได้ เปรียบเทียบกับการลดลงจนถึงอุณหภูมิกระเปาะเปียกในรูปร้อยละ  ยิ่งสูงยิ่งดี พบว่า"+str7)
    datastructure6=pd.DataFrame.from_dict(efmean, orient='index')
    # Center align all text
    styled_df6 = datastructure6.style.set_table_styles(
    [{'selector': 'th', 'props': [('text-align', 'center'),('width','5cm')]},
     {'selector': 'td', 'props': [('text-align', 'center')]}]
    )
    # Render as HTML    
    st.write(styled_df6.to_html(), unsafe_allow_html=True)


 # # ########################################################################################
    st.subheader("ความผิดปกติ :Anomaly Detection")
    st.write("1.ความคลาดเคลื่อนจากโมเดล: Deviation from Historical trend")
    anomaly_results = st.session_state.model_output.get("anomaly_results", [])
    if anomaly_results:
        df_anomaly = pd.DataFrame(anomaly_results)
        st.dataframe(df_anomaly)
    else:
        st.info("No anomalies detected or no data available.")
    st.markdown("---")
    st.write("2.ความผิดปกติของข้อมูล: Deviation of Data")
    st.session_state.model_output2 = run_isolation_forest_model(st.session_state.chspec, st.session_state.data1, st.session_state.data3)
    anomaly_results2 = st.session_state.model_output2.get("anomaly_results", [])
    df_anomaly2 = pd.DataFrame(anomaly_results2)
    st.dataframe(df_anomaly2)
else:
    st.info("Model output will appear here after execution. Please ensure data is loaded and model is executed.")


st.markdown("---")
st.header("6. Save All Current Data")

# --- Save Option for User ---
if st.button("Save All Current Data to Excel", key="save_excel_btn"):
    try:
        excel_buffer, file_name = save_data_to_excel(
            st.session_state.chspec,
            st.session_state.data1, # Historical data
            st.session_state.data3  # Current operational data
        )
        st.success(f"Data prepared for download: '{file_name}'.")
        st.download_button(
            label="Download Saved Excel File",
            data=excel_buffer,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_button"
        )
    except Exception as e:
        st.error(f"Error saving data to Excel: {e}")
