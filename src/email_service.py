import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from pathlib import Path
from .logging_config import logger

class EmailService:
    def __init__(self, config_path='config/email_config.json'):
        """Initialize email service with configuration."""
        self.config = self._load_config(config_path)
        self.smtp_server = self.config.get('smtp_server')
        self.smtp_port = self.config.get('smtp_port')
        self.sender_email = self.config.get('sender_email')
        self.sender_password = self.config.get('sender_password')
        
    def _load_config(self, config_path):
        """Load email configuration from file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Email config file not found: {config_path}")
                return {}
            with open(config_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading email config: {str(e)}", exc_info=True)
            return {}
    
    def send_email(self, recipient, subject, body, html_body=None):
        """Send an email to the specified recipient."""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add plain text version
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML version if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}", exc_info=True)
            return False
    
    def send_model_training_notification(self, recipient, model_type, performance_metrics):
        """Send notification about model training completion."""
        subject = f"Model Training Complete - {model_type}"
        body = f"""
        Model training has been completed successfully.
        
        Model Type: {model_type}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Performance Metrics:
        {json.dumps(performance_metrics, indent=2)}
        """
        
        html_body = f"""
        <html>
            <body>
                <h2>Model Training Complete</h2>
                <p><strong>Model Type:</strong> {model_type}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h3>Performance Metrics:</h3>
                <pre>{json.dumps(performance_metrics, indent=2)}</pre>
            </body>
        </html>
        """
        
        return self.send_email(recipient, subject, body, html_body)
    
    def send_model_drift_alert(self, recipient, model_type, drift_metrics):
        """Send alert about model drift detection."""
        subject = f"Model Drift Alert - {model_type}"
        body = f"""
        Model drift has been detected.
        
        Model Type: {model_type}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Drift Metrics:
        {json.dumps(drift_metrics, indent=2)}
        """
        
        html_body = f"""
        <html>
            <body>
                <h2 style="color: red;">Model Drift Alert</h2>
                <p><strong>Model Type:</strong> {model_type}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h3>Drift Metrics:</h3>
                <pre>{json.dumps(drift_metrics, indent=2)}</pre>
            </body>
        </html>
        """
        
        return self.send_email(recipient, subject, body, html_body)
    
    def send_system_alert(self, recipient, alert_type, message):
        """Send system alert email."""
        subject = f"System Alert - {alert_type}"
        body = f"""
        System Alert
        
        Type: {alert_type}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Message:
        {message}
        """
        
        html_body = f"""
        <html>
            <body>
                <h2 style="color: orange;">System Alert</h2>
                <p><strong>Type:</strong> {alert_type}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h3>Message:</h3>
                <p>{message}</p>
            </body>
        </html>
        """
        
        return self.send_email(recipient, subject, body, html_body) 