
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

class RegionCounter(BaseSolution):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "dragging": False,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.counting_regions = []

    def add_region(self, name, polygon_points, region_color, text_color):
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)

    def count(self, im0):
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        if self.region is None:
            self.initialize_region()
            regions = {"Region#01": self.region}
        else:
            regions = self.region if isinstance(self.region, dict) else {"Region#01": self.region}

        for idx, (region_name, reg_pts) in enumerate(regions.items(), start=1):
            if not isinstance(reg_pts, list) or not all(isinstance(pt, tuple) for pt in reg_pts):
                LOGGER.warning(f"Invalid region points for {region_name}: {reg_pts}")
                continue
            color = colors(idx, True)
            self.annotator.draw_region(reg_pts=reg_pts, color=color, thickness=self.line_width * 2)
            self.add_region(region_name, reg_pts, color, self.annotator.get_txt_color())

        for region in self.counting_regions:
            region["prepared_polygon"] = self.prep(region["polygon"])

        for box, cls in zip(self.boxes, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            for region in self.counting_regions:
                if region["prepared_polygon"].contains(self.Point(bbox_center)):
                    region["counts"] += 1

        for region in self.counting_regions:
            self.annotator.text_label(
                region["polygon"].bounds,
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
            )
            region["counts"] = 0

        self.display_output(im0)
        return im0
