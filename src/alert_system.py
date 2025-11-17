"""
EquiRide Surge AI - Forecast-Based Alert System (Multi-Recipient Support)
Reads ONLY datasets/forecast_15min_predictions.csv for surge alerts.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ---------------- FIX IMPORT PATH ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------------------------------------

from config.logging_config import setup_logging, log_stage

# Load environment variables
load_dotenv()

class AlertSystem:
    def __init__(self, config, logger=None):
        """Initialize Alert System"""
        self.config = config
        self.logger = logger or setup_logging()
        self.twilio_enabled = config.get('twilio', {}).get('enabled', False)
        self.alert_cooldown = config.get('alerts', {}).get('alert_cooldown_minutes', 30)
        self.last_alert_time = {}

        if self.twilio_enabled:
            self._initialize_twilio()
        else:
            self.logger.info("Twilio alerts disabled in config")

    def _initialize_twilio(self):
        """Initialize Twilio client for multiple recipients"""
        try:
            from twilio.rest import Client
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
            
            # --- MULTI-RECIPIENT LOGIC ---
            # Read the string, split by comma, and strip whitespace
            phones_env = os.getenv('ALERT_PHONE_NUMBER', "")
            self.alert_phones = [p.strip() for p in phones_env.split(',') if p.strip()]

            if not all([account_sid, auth_token, self.twilio_phone, self.alert_phones]):
                self.logger.warning("Twilio credentials incomplete or no alert numbers found. SMS disabled.")
                self.twilio_enabled = False
                return

            self.twilio_client = Client(account_sid, auth_token)
            self.logger.info(f"SUCCESS Twilio initialized for {len(self.alert_phones)} recipients.")

        except Exception as e:
            self.logger.error(f"FAILED Twilio initialization failed: {str(e)}")
            self.twilio_enabled = False

    # ===================================================================
    # COOLDOWN CHECK
    # ===================================================================
    def check_alert_cooldown(self, area_name):
        if area_name not in self.last_alert_time:
            return True
        time_since_last = datetime.now() - self.last_alert_time[area_name]
        return time_since_last.total_seconds() > (self.alert_cooldown * 60)

    # ===================================================================
    # SEND SMS
    # ===================================================================
    def send_sms_alert(self, message, area_name):
        """Send SMS via Twilio to ALL recipients"""
        if not self.twilio_enabled:
            safe_msg = message.encode("ascii", "ignore").decode()
            self.logger.info(f"[ALERT - SMS DISABLED] {safe_msg}")
            return False

        if not self.check_alert_cooldown(area_name):
            self.logger.info(f"Cooldown active for {area_name}. Skipping SMS.")
            return False

        success_count = 0
        try:
            # Loop through every number in the list
            for phone in self.alert_phones:
                try:
                    msg = self.twilio_client.messages.create(
                        body=message,
                        from_=self.twilio_phone,
                        to=phone
                    )
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed sending to {phone}: {str(e)}")

            if success_count > 0:
                self.last_alert_time[area_name] = datetime.now()
                self.logger.info(f"SUCCESS SMS sent to {success_count}/{len(self.alert_phones)} recipients.")
                return True
            return False

        except Exception as e:
            self.logger.error(f"FAILED SMS send error: {str(e)}")
            return False

    # ===================================================================
    # FORECAST CSV LOADING
    # ===================================================================
    def load_forecast_data(self, csv_path="datasets/forecast_15min_predictions.csv"):
        try:
            df = pd.read_csv(csv_path)
            # Normalize columns just in case
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            
            # Check for normalized names
            required = ["h3_index", "next_time", "pred_bookings_15min"]
            if not all(col in df.columns for col in required):
                # Fallback check for old names
                rename_map = {'h3': 'h3_index', 'pred_bookings': 'pred_bookings_15min'}
                df = df.rename(columns=rename_map)
                
            return df
        except Exception as e:
            self.logger.error(f"FAILED to load forecast CSV: {str(e)}")
            return pd.DataFrame()

    # ===================================================================
    # FORECAST-BASED SURGE DETECTION (TOP 5 DEMO LOGIC)
    # ===================================================================
    def calculate_forecast_surges(self, df):
        try:
            log_stage(self.logger, "CHECK_FORECAST_SURGE", "START")
            
            # Sort by demand to find the biggest hotspots
            if "pred_bookings_15min" not in df.columns:
                return []
                
            surge_rows = df.sort_values(by="pred_bookings_15min", ascending=False).head(5)
            
            # Basic filter to avoid alerting on 0 demand
            surge_rows = surge_rows[surge_rows["pred_bookings_15min"] > 20]

            alerts = []
            max_val = surge_rows["pred_bookings_15min"].max() if not surge_rows.empty else 100

            for _, row in surge_rows.iterrows():
                alerts.append({
                    "h3_index": row["h3_index"],
                    "area_name": row.get("h3_index", "Unknown Zone"),
                    "severity": "CRITICAL" if row["pred_bookings_15min"] >= max_val * 0.9 else "HIGH",
                    "pred_bookings": row["pred_bookings_15min"],
                    "next_time": row["next_time"],
                    "timestamp": datetime.now().isoformat()
                })

            log_stage(self.logger, "CHECK_FORECAST_SURGE", "SUCCESS", alerts_found=len(alerts))
            return alerts

        except Exception as e:
            log_stage(self.logger, "CHECK_FORECAST_SURGE", "FAILURE", error=str(e))
            return []

    # ===================================================================
    # MESSAGE FORMATTING
    # ===================================================================
    def format_forecast_alert_message(self, alert):
        msg = f"""
🚨 EQUI-RIDE SURGE ALERT 🚨

📍 Zone: {alert['h3_index']}
⚠️ Level: {alert['severity']}
📈 Demand: {alert['pred_bookings']:.0f} bookings
⏰ Time: {alert['next_time']}

Dispatch Action Required immediately.
"""
        return msg.strip()

    # ===================================================================
    # PROCESS FORECAST ALERTS
    # ===================================================================
    def process_forecast_alerts(self, csv_path="datasets/forecast_15min_predictions.csv"):
        try:
            df = self.load_forecast_data(csv_path)

            if df.empty:
                self.logger.info("No forecast data → No alerts.")
                return 0

            alerts = self.calculate_forecast_surges(df)

            if len(alerts) == 0:
                self.logger.info("No forecast surge detected.")
                return 0

            sent = 0
            for alert in alerts:
                msg = self.format_forecast_alert_message(alert)
                # This returns True if at least one SMS was sent successfully
                if self.send_sms_alert(msg, alert["h3_index"]):
                    sent += 1

            log_stage(self.logger, "PROCESS_FORECAST_ALERTS", "SUCCESS", sent=sent)
            return sent

        except Exception as e:
            log_stage(self.logger, "PROCESS_FORECAST_ALERTS", "FAILURE", error=str(e))
            return 0
