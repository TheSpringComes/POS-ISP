
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class QueueManager(BaseSolution):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0
        self.rect_color = (255, 255, 255)
        self.region_length = len(self.region)

    def process_queue(self, im0):
        self.counts = 0
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):

            self.annotator.box_label(box, label=self.names[cls], color=colors(track_id, True))
            self.store_tracking_history(track_id, box)

            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            track_history = self.track_history.get(track_id, [])

            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        self.annotator.queue_counts_display(
            f"Queue Counts : {str(self.counts)}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        self.display_output(im0)

        return im0
