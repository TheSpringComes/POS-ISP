
from time import time

import numpy as np

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class SpeedEstimator(BaseSolution):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.initialize_region()

        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}

    def estimate_speed(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))

            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"

            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        self.display_output(im0)

        return im0
