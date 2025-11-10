# app/mailer.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Iterable, Optional
from datetime import datetime

from .config import settings


def _to_list(v: Optional[str | Iterable[str]]) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


class Mailer:
    def __init__(self):
        self.host = settings.MAIL_HOST
        self.port = settings.MAIL_PORT
        self.username = settings.MAIL_USERNAME
        self.password = settings.MAIL_PASSWORD
        self.from_addr = settings.MAIL_FROM
        self.from_name = settings.MAIL_FROM_NAME
        self.reply_to = settings.MAIL_REPLY_TO
        self.use_tls = settings.MAIL_TLS
        self.use_ssl = settings.MAIL_SSL
        self.department_email = settings.DEPARTMENT_EMAIL

    def send_email(
        self,
        to: str | Iterable[str],
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        cc: Optional[str | Iterable[str]] = None,
        bcc: Optional[str | Iterable[str]] = None,
    ) -> dict:
        """Send an email. Raises RuntimeError on SMTP auth or send errors."""
        to_list = _to_list(to)
        cc_list = _to_list(cc)
        bcc_list = _to_list(bcc)
        recipients = to_list + cc_list + bcc_list

        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self.from_name} <{self.from_addr}>"
        msg["To"] = ", ".join(to_list)
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        msg["Subject"] = subject
        if self.reply_to:
            msg.add_header("Reply-To", self.reply_to)

        if text_body:
            msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        server = None
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.host, self.port)
            else:
                server = smtplib.SMTP(self.host, self.port)
                if self.use_tls:
                    server.starttls()

            if self.username and self.password:
                try:
                    server.login(self.username, self.password)
                except smtplib.SMTPAuthenticationError as e:
                    raise RuntimeError(
                        "SMTP authentication failed. If using Gmail, enable 2-Step Verification and use an App Password."
                    ) from e

            server.sendmail(self.from_addr, recipients, msg.as_string())

        except smtplib.SMTPException as e:
            raise RuntimeError(f"SMTP error: {e}") from e
        finally:
            try:
                if server:
                    server.quit()
            except Exception:
                pass

        return {
            "to": recipients,
            "subject": subject,
            "sent_at": datetime.utcnow().isoformat() + "Z",
        }


# ------------------------------------------------
# HTML templates (English) – your 6 notifications
# ------------------------------------------------

def tpl_ticket_submitted_user(ticket_id: int, user_name: str | None):
    s = f"[Ticket #{ticket_id}] Submitted Successfully"
    h = f"""
    <html><body>
      <h2>Ticket Submitted</h2>
      <p>Dear {user_name or "User"},</p>
      <p>Your ticket <strong>#{ticket_id}</strong> has been submitted successfully.</p>
      <p>Our helpdesk team will review it and assign it shortly.</p>
      <p>Best regards,<br/>Smart Ticket Management System</p>
    </body></html>
    """
    return s, h


def tpl_new_ticket_received_hd(ticket_id: int):
    s = f"[Ticket #{ticket_id}] New Ticket Received"
    h = f"""
    <html><body>
      <h2>New Ticket Received</h2>
      <p>Dear Help Desk,</p>
      <p>A new ticket <strong>#{ticket_id}</strong> has been submitted via the Smart Ticket Management System.</p>
      <p>Please review the details and assign it to the appropriate support team.</p>
      <p>Best regards,<br/>Smart Ticket Management System</p>
    </body></html>
    """
    return s, h


def tpl_ticket_assigned_user(ticket_id: int, user_name: str | None, team_name: str):
    s = f"[Ticket #{ticket_id}] Assigned to {team_name}"
    h = f"""
    <html><body>
      <h2>Ticket Assigned</h2>
      <p>Dear {user_name or "User"},</p>
      <p>Your ticket <strong>#{ticket_id}</strong> has been assigned to the <strong>{team_name}</strong> team.</p>
      <p>They will review and update you shortly.</p>
      <p>Best regards,<br/>Smart Ticket Management System</p>
    </body></html>
    """
    return s, h


def tpl_ticket_assigned_team(ticket_id: int, team_name: str):
    s = f"[Ticket #{ticket_id}] Assigned to Your Team"
    h = f"""
    <html><body>
      <h2>New Ticket Assigned</h2>
      <p>Dear {team_name} Team,</p>
      <p>A new ticket <strong>#{ticket_id}</strong> has been assigned to your queue.</p>
      <p>Please review the ticket details and proceed with the required actions to resolve the issue.</p>
      <p>Best regards,<br/>Smart Ticket Management System</p>
    </body></html>
    """
    return s, h


def tpl_ticket_resolved_user(ticket_id: int, user_name: str | None):
    s = f"[Ticket #{ticket_id}] Resolved"
    h = f"""
    <html><body>
      <h2>Ticket Resolved</h2>
      <p>Dear {user_name or "User"},</p>
      <p>Your ticket <strong>#{ticket_id}</strong> has been resolved successfully.</p>
      <p>If the issue persists, reply to this email and we will investigate further.</p>
      <p>Best regards,<br/>Smart Ticket Management System</p>
    </body></html>
    """
    return s, h


def tpl_ticket_canceled_user(ticket_id: int, user_name: str | None):
    s = f"[Ticket #{ticket_id}] Canceled"
    h = f"""
    <html><body>
      <h2>Ticket Canceled</h2>
      <p>Dear {user_name or "User"},</p>
      <p>Your ticket <strong>#{ticket_id}</strong> has been canceled.</p>
      <p>If cancellation was not intended or you need further assistance, reply to this email.</p>
      <p>Best regards,<br/>Smart Ticket Management System</p>
    </body></html>
    """
    return s, h
