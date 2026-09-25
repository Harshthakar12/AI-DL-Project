import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=2):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        prediction = self.fc(last_step)
        return prediction

class WeightTrajectoryPredictor:
    def __init__(self, seq_len=7, hidden_size=32, num_layers=2):
        self.seq_len = seq_len
        self.feature_cols = ['Weight_kg', 'Net_Calories', 'Protein_g', 'Carbs_g', 'Fat_g', 'Steps']
        self.num_features = len(self.feature_cols)
        self.model = TrajectoryLSTM(input_size=self.num_features, hidden_size=hidden_size, num_layers=num_layers)
        self.scaler_min = None
        self.scaler_max = None
        self.is_trained = False
        
    def _fit_scaler(self, data):
        self.scaler_min = np.min(data, axis=0)
        self.scaler_max = np.max(data, axis=0)
        # Avoid division by zero
        diff = self.scaler_max - self.scaler_min
        diff[diff == 0] = 1.0
        self.scaler_diff = diff
        
    def _transform(self, data):
        return (data - self.scaler_min) / self.scaler_diff
        
    def _inverse_transform_weight(self, scaled_weight):
        w_min = self.scaler_min[0]
        w_diff = self.scaler_diff[0]
        return scaled_weight * w_diff + w_min
        
    def train(self, df: pd.DataFrame, epochs=75, lr=0.008):
        data = df[self.feature_cols].values
        self._fit_scaler(data)
        scaled_data = self._transform(data)
        
        # Prepare sliding windows
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_len):
            X.append(scaled_data[i:i + self.seq_len])
            # Target is the weight at the next step (index 0 is Weight_kg)
            y.append(scaled_data[i + self.seq_len, 0])
            
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        self.is_trained = True
        return loss.item()

    def forecast_future(
        self,
        historical_df: pd.DataFrame,
        future_days=30,
        planned_daily_intake=1900,
        planned_tdee=2400,
        planned_protein=160,
        planned_carbs=180,
        planned_fat=55,
        planned_steps=9500
    ):
        """
        Rolls the LSTM model forward into the future given the user's target nutrition plan.
        """
        if not self.is_trained:
            self.train(historical_df)
            
        self.model.eval()
        
        # Last known sequence
        recent_data = historical_df[self.feature_cols].values[-self.seq_len:]
        current_seq = self._transform(recent_data).tolist()
        
        planned_net = planned_daily_intake - planned_tdee
        
        last_date_str = historical_df['Date'].iloc[-1]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        
        forecast_dates = []
        forecast_weights = []
        lower_bounds = []
        upper_bounds = []
        
        with torch.no_grad():
            for day in range(1, future_days + 1):
                input_tensor = torch.tensor([current_seq[-self.seq_len:]], dtype=torch.float32)
                pred_scaled_weight = self.model(input_tensor).item()
                
                # Inverse scale weight
                pred_weight = self._inverse_transform_weight(pred_scaled_weight)
                
                # Future date
                future_date = last_date + timedelta(days=day)
                forecast_dates.append(future_date.strftime("%Y-%m-%d"))
                forecast_weights.append(round(pred_weight, 2))
                
                # Confidence / water weight bounds (+/- 0.6kg expanding slightly over time)
                water_band = 0.5 + (day * 0.015)
                lower_bounds.append(round(pred_weight - water_band, 2))
                upper_bounds.append(round(pred_weight + water_band, 2))
                
                # Create the synthetic feature vector for this forecasted day
                raw_next_step = np.array([
                    pred_weight,
                    planned_net,
                    planned_protein,
                    planned_carbs,
                    planned_fat,
                    planned_steps
                ])
                scaled_next_step = self._transform(raw_next_step.reshape(1, -1))[0]
                current_seq.append(scaled_next_step.tolist())
                
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted_Weight_kg": forecast_weights,
            "Lower_Bound_kg": lower_bounds,
            "Upper_Bound_kg": upper_bounds,
            "Planned_Intake": planned_daily_intake,
            "Planned_Net": planned_net
        })
        
        return forecast_df
