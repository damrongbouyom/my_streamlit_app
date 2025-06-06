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
        "general_comments": "Random Forest model executed successfully. "
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
        "general_comments": "Gordon Ng model executed successfully. "
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

        # Define features and target for the Gordon Ng model
           # The Gordon Ng model typically uses:
        # X1 = (Tc-Tw)/TcTw                                   #Ton
        # X2 = (Q*Q/(tCtW))*(1+1/COP)                                   #(TCDR - TCDS)
        # Target Y =  ((Tw/Tc)*(1+1/cop) -1 )*(Q/Tw)                              #   kW
        gn_feature_cols = ['ton', 'tcds', 'tcdr', 'fwch', 'kw']            # Base features
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
  
        chiller_df_predict_cleaned['X1'] = ( chiller_df_predict_cleaned['tcdr'] -  chiller_df_predict_cleaned['tchr'])/(( chiller_df_predict_cleaned['tcdr']+273.15)*( chiller_df_predict_cleaned['tchr']+273.15))
        chiller_df_predict_cleaned['X2'] = (( chiller_df_predict_cleaned['ton']* chiller_df_predict_cleaned['ton']*3510*3510 )/(( chiller_df_predict_cleaned['tcdr']+273.15)*( chiller_df_predict_cleaned['tchr']+273.15)))*(1+(1/ ( 3.510* chiller_df_predict_cleaned['ton']/ chiller_df_predict_cleaned['kw'])))
        X_predict_gn = chiller_df_predict_cleaned[final_gn_features]

        # Make predictions
        # ((y *(Tw/Q) +1)*(Tc/Tw) -1 )*Q/1000
        predicted_kw_values = ((gn_model.predict(X_predict_gn)*((chiller_df_predict_cleaned['tchr']+273.15)/(3510*chiller_df_predict_cleaned['ton']))+1)*((chiller_df_predict_cleaned['tcdr']+273.15)/(chiller_df_predict_cleaned['tchr']+273.15))-1)*chiller_df_predict_cleaned['ton']*3510/1000
        chiller_df_predict_cleaned['predicted_kw'] = predicted_kw_values
        st.write(chiller_df_predict_cleaned)
        # Merge predictions back to the original chiller_df_predict
        chiller_df_predict = chiller_df_predict.merge(
            chiller_df_predict_cleaned[['ch_name', 'h', 'predicted_kw']],
            on=['ch_name', 'h'],
            how='left'
        )
        all_chiller_predictions_df = pd.concat([all_chiller_predictions_df, chiller_df_predict], ignore_index=True)

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

def run_isolation_forest_model2(chspec, historical_data, current_data):
    """
    Executes the Isolation Forest model for anomaly detection.
    Analyzes current_data if available, otherwise historical_data.
    """
    #st.info("Running Isolation Forest Model...")
    results = {
        "chiller_graphs_data": {},
        "anomaly_results": [],
        "general_comments": "Isolation Forest model executed successfully. "
    }

    # Decide which data to analyze (prioritize current over historical)
    data_to_analyze = current_data if current_data else historical_data
    if not data_to_analyze:
        results["general_comments"] += "No data provided for analysis."
        return results

    all_chiller_data_for_if = []
    # Flatten the data_to_analyze structure into a single list of dictionaries
    for entry in data_to_analyze:
        ch_name = entry['name']
        for data_row in entry['data']:
            # Add chiller name to each row's data
            row_with_name = {"ch_name": ch_name}
            row_with_name.update(data_row)
            all_chiller_data_for_if.append(row_with_name)

    if not all_chiller_data_for_if:
        results["general_comments"] += "Operational data is empty for analysis."
        return results

    # Convert to DataFrame
    df_operational = pd.DataFrame(all_chiller_data_for_if)

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

    # Process results for Streamlit output
    for chiller_name in df_operational['ch_name'].unique():
        chiller_df = df_operational[df_operational['ch_name'] == chiller_name].copy()

        # Prepare data for 24h graphs (ensure 'h' column for plotting)
        if 'h' not in chiller_df.columns:
            chiller_df['h'] = range(1, len(chiller_df) + 1) # Dummy hour if not present

        results["chiller_graphs_data"][chiller_name] = {
            "hours": chiller_df['h'].tolist(),
            "kw": chiller_df['kw'].tolist(),
            "ton": chiller_df['ton'].tolist(),
            "kw_per_ton": chiller_df['kw_per_ton'].tolist() if 'kw_per_ton' in chiller_df.columns else [np.nan]*len(chiller_df) # Ensure kw_per_ton is included
        }

        # Populate anomaly results
        anomalies_for_chiller = chiller_df[chiller_df['anomaly'] == -1]
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

    return plant_summary_df

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


# --- Main Content Area ---

# --- NEW SECTION: About This App & Methodology ---
with st.expander("About This App & Methodology", expanded=True):
    st.markdown("""
    This application provides a tool for analyzing the performance of chiller systems. Users can input chiller specifications and operational data to gain insights into efficiency, identify potential anomalies, and benchmark performance using different analytical models.

    **Key Capabilities:**
    * อ่านข้อมูล Logsheet และเปรียบเทียบกับสเปคว่าทำงานปกติหรือไม่ (rated capacity, rated power, IPLV).
    * อ่านข้อมูล Logsheet ในอดีตและเปรียบเทียบกับการทำงานปัจจุบันว่าปกติหรือไม่               
    * Upload and manage historical and current operational data in Excel (temperatures, flow, power).
    * Run analysis using different models to assess chiller performance.
    * Visualize key performance indicators (kW, Ton, kW/Ton) over time.
    * Identify potential anomalies or deviations in chiller operation.
    * Export all input and generated data for further use.

    **Methodology & Key Calculations:**
    The application performs several critical calculations to derive chiller performance metrics:
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
    * The models currently implemented are **placeholders** to demonstrate functionality. Actual production-ready models would require extensive training on diverse and high-quality datasets specific to your chillers.
    * The accuracy of the analysis heavily relies on the quality, accuracy, and completeness of the input operational data.
    * Anomaly detection thresholds and rules are illustrative and should be fine-tuned based on specific system characteristics, operational tolerances, and expert knowledge.
    """)

# --- NEW SECTION: Load Previously Saved Data ---
st.header("Load Previously Saved Data")
st.info("If you have previously saved your chiller specifications and operational data using this app (output_chiller_data.xlsx), you can upload that file here to quickly restore your session.")
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
            st.success("All data (Chiller Specs, Historical, Current) has been loaded from your file!")
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
        df_t=get_plant_data(st.session_state.data3)
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
    st.write(st.session_state.model_output.get("general_comments", "No general comments provided."))
    st.info("1.สรุปภาพรวมการทำงาน")
    st.info("2.เปรียบเทียบค่า Spec กับ ข้อมูลการใช้งาน (data1)")
    st.info("3.เปรียบเทียบค่า ข้อมูลปัจจุบัน (data2) กับ ข้อมูลการทำงานปกติ (data1)")
    st.info("4.สมรรถนะการระบายความร้อน(Cooling Tower Plant Preformance)")

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