import smtplib
from email.message import EmailMessage
import streamlit as st

class EmailSender:
    def __init__(self, smtp_server=st.secrets["SMTP_SERVER"], smtp_port=int(st.secrets["SMTP_PORT"]), username=st.secrets["USERNAME"], password=st.secrets["PASSWORD"]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_email(self, to_email, subject, content):
        msg = EmailMessage()
        msg.set_content(content)
        msg['Subject'] = subject
        msg['From'] = self.username  # Consider specifying a more descriptive 'From' address if needed
        msg['To'] = to_email

        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.username, self.password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")