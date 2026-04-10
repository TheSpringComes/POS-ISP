
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

class SecurityAlarm(BaseSolution):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email, password, to_email):
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com: 587")
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records=5):
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()

        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"

        message_body = f"Ultralytics ALERT!!! {records} objects have been detected!!"
        message.attach(MIMEText(message_body))

        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)

        try:
            self.server.send_message(message)
            LOGGER.info("✅ Email sent successfully!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

    def monitor(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        for box, cls in zip(self.boxes, self.clss):

            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det > self.records and not self.email_sent:
            self.send_email(im0, total_det)
            self.email_sent = True

        self.display_output(im0)

        return im0
