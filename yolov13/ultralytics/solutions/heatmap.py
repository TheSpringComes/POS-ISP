
import cv2
import numpy as np

from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.utils.plotting import Annotator

class Heatmap(ObjectCounter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.initialized = False
        if self.region is not None:
            self.initialize_region()

        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]
        self.heatmap = None

    def heatmap_effect(self, box):
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        within_radius = dist_squared <= radius_squared

        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def generate_heatmap(self, im0):
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99
        self.initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):

            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)
                self.store_classwise_counts(cls)
                current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(current_centroid, track_id, prev_position, cls)

        if self.region is not None:
            self.display_counts(im0)

        if self.track_data.id is not None:
            im0 = cv2.addWeighted(
                im0,
                0.5,
                cv2.applyColorMap(
                    cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.colormap
                ),
                0.5,
                0,
            )

        self.display_output(im0)
        return im0
