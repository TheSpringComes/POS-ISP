
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator

class AIGym(BaseSolution):

    def __init__(self, **kwargs):

        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"

        super().__init__(**kwargs)
        self.count = []
        self.angle = []
        self.stage = []

        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])
        self.down_angle = float(self.CFG["down_angle"])
        self.kpts = self.CFG["kpts"]

    def monitor(self, im0):

        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)[0]

        if tracks.boxes.id is not None:

            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            self.annotator = Annotator(im0, line_width=self.line_width)

            for ind, k in enumerate(reversed(tracks.keypoints.data)):

                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)
                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.line_width * 3)

                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts[1])],
                )

        self.display_output(im0)
        return im0
