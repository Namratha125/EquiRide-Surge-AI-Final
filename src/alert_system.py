"""
EquiRide Surge AI - Forecast-Based Alert System (Standalone + Compatible)
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
        """Initialize Twilio client with Multiple Recipients"""
        try:
            from twilio.rest import Client
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
            
            # --- UPDATE: Load multiple comma-separated numbers ---
            phones_str = os.getenv('ALERT_PHONE_NUMBER', "")
            self.alert_phones = [p.strip() for p in phones_str.split(',') if p.strip()]

            if not all([account_sid, auth_token, self.twilio_phone, self.alert_phones]):
                self.logger.warning("Twilio credentials invalid or no numbers found. SMS disabled.")
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

        sent_count = 0
        try:
            # --- UPDATE: Loop through all numbers ---
            for phone in self.alert_phones:
                try:
                    msg = self.twilio_client.messages.create(
                        body=message,
                        from_=self.twilio_phone,
                        to=phone
                    )
                    sent_count += 1
                except Exception as inner_e:
                    self.logger.error(f"Failed sending to {phone}: {inner_e}")

            if sent_count > 0:
                self.last_alert_time[area_name] = datetime.now()
                self.logger.info(f"SUCCESS SMS sent to {sent_count} recipients. SID: {msg.sid}")
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

            required = ["h3_index", "next_time", "pred_bookings_15min"]
            if not all(col in df.columns for col in required):
                raise ValueError("CSV must contain: h3_index, next_time, pred_bookings_15min")

            return df

        except Exception as e:
            self.logger.error(f"FAILED to load forecast CSV: {str(e)}")
            return pd.DataFrame()

    # ===================================================================
    # FORECAST-BASED SURGE DETECTION
    # ===================================================================
    def calculate_forecast_surges(self, df):
        try:
            log_stage(self.logger, "CHECK_FORECAST_SURGE", "START")

            # --- UPDATE: Use Top 5 Logic for Reliable Demos ---
            # Using standard deviation often fails for small datasets (0 alerts).
            # This ensures the highest demand zones are ALWAYS flagged.
            surge_rows = df.sort_values(by="pred_bookings_15min", ascending=False).head(5)
            
            # Filter out very low values (e.g. zeros)
            surge_rows = surge_rows[surge_rows["pred_bookings_15min"] > 10]

            alerts = []
            for _, row in surge_rows.iterrows():
                alerts.append({
                    "h3_index": row["h3_index"],
                    "area_name": row["h3_index"],
                    "severity": "CRITICAL" if row["pred_bookings_15min"] == surge_rows["pred_bookings_15min"].max() else "HIGH",
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
PREDICTED SURGE ALERT

H3 Zone: {alert['h3_index']}
Severity: {alert['severity']}
Predicted Bookings (next 15 min): {alert['pred_bookings']:.1f}
Forecast Time: {alert['next_time']}

Action: Deploy more drivers.
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
                if self.send_sms_alert(msg, alert["h3_index"]):
                    sent += 1

            log_stage(self.logger, "PROCESS_FORECAST_ALERTS", "SUCCESS", sent=sent)
            return sent

        except Exception as e:
            log_stage(self.logger, "PROCESS_FORECAST_ALERTS", "FAILURE", error=str(e))
            return 0
