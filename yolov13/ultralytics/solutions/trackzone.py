
import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class TrackZone(BaseSolution):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        default_region = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))

    def trackzone(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)

        masked_frame = cv2.bitwise_and(im0, im0, mask=cv2.fillPoly(np.zeros_like(im0[:, :, 0]), [self.region], 255))
        self.extract_tracks(masked_frame)

        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=f"{self.names[cls]}:{track_id}", color=colors(track_id, True))

        self.display_output(im0)

        return im0
