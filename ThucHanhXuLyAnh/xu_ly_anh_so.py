import tkinter as tk
from tkinter.filedialog import Open
from tkinter.filedialog import asksaveasfilename

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import Chapter3 as c3


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Xử lý ảnh số")
        self.geometry("250x320")
        self.resizable(False, False)
        self.imgin = None
        self.imgout = None
        self.filename = None
        self.model = YOLO("yolov8n_trai_cay.pt", task="detect")

        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.mnu_open_image_click)
        file_menu.add_command(
            label="Open Color Image", command=self.mnu_open_color_image_click
        )

        file_menu.add_command(label="Save Image", command=self.mnu_save_image_click)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        yolo_menu = tk.Menu(menu, tearoff=0)
        yolo_menu.add_command(label="Predict", command=self.mnu_yolo_predict_click)
        menu.add_cascade(label="Yolo", menu=yolo_menu)

        chapter3_menu = tk.Menu(menu, tearoff=0)
        chapter3_menu.add_command(label="Negative", command=self.mnu_c3_negative_click)
        chapter3_menu.add_command(
            label="Negative Color", command=self.mnu_c3_negative_color_click
        )
        chapter3_menu.add_command(label="Logarit", command=self.mnu_c3_logarit_click)
        chapter3_menu.add_command(label="Power", command=self.mnu_c3_power_click)
        chapter3_menu.add_command(
            label="Piecewise Line", command=self.mnu_c3_piecewise_line_click
        )
        chapter3_menu.add_command(
            label="Histogram", command=self.mnu_c3_histogram_click
        )
        chapter3_menu.add_command(
            label="Hist Equal", command=self.mnu_c3_hist_equal_click
        )
        chapter3_menu.add_command(
            label="Hist Equal Color", command=self.mnu_c3_hist_equal_color_click
        )
        chapter3_menu.add_command(
            label="Local Hist", command=self.mnu_c3_local_hist_click
        )
        chapter3_menu.add_command(
            label="Hist Stat", command=self.mnu_c3_hist_stat_click
        )
        chapter3_menu.add_command(
            label="Smooth Box", command=self.mnu_c3_smooth_box_click
        )
        chapter3_menu.add_command(
            label="Smooth Gauss", command=self.mnu_c3_smooth_gauss_click
        )
        chapter3_menu.add_command(
            label="Median Filter", command=self.mnu_c3_median_filter_click
        )
        chapter3_menu.add_command(
            label="Create Impulse Noise", command=self.mnu_c3_create_impulse_noise_click
        )

        chapter3_menu.add_command(label="Sharp", command=self.mnu_c3_sharp_click)

        menu.add_cascade(label="Chapter3", menu=chapter3_menu)

        self.config(menu=menu)

    def mnu_open_image_click(self):
        ftypes = [("Images", "*.jpg *.tif *.bmp *.png *.jpeg *.webp")]
        dlg = Open(self, filetypes=ftypes, title="Open Image")
        self.filename = dlg.show()

        if self.filename != "":
            self.imgin = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            cv2.imshow("ImageIn", self.imgin)

    def mnu_open_color_image_click(self):
        ftypes = [("Images", "*.jpg *.tif *.bmp *.png *.jpeg *.webp")]
        dlg = Open(self, filetypes=ftypes, title="Open Image")
        self.filename = dlg.show()

        if self.filename != "":
            self.imgin = cv2.imread(self.filename, cv2.IMREAD_COLOR)
            cv2.imshow("ImageIn", self.imgin)

    def mnu_save_image_click(self):
        ftypes = [("Images", "*.jpg *.tif *.bmp *.png")]
        filenameout = asksaveasfilename(
            title="Image Save", filetypes=ftypes, initialfile=self.filename
        )
        if filenameout is not None:
            cv2.imwrite(filenameout, self.imgout)

    def mnu_yolo_predict_click(self):
        names = self.model.names
        self.imgout = self.imgin.copy()
        annotator = Annotator(self.imgout)
        results = self.model.predict(self.imgin, conf=0.5, verbose=False)

        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()
        for box, cls, conf in zip(boxes, clss, confs):
            annotator.box_label(
                box,
                label=names[int(cls)] + " %4.2f" % conf,
                txt_color=(255, 0, 0),
                color=(255, 255, 255),
            )
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_negative_click(self):
        self.imgout = c3.Negative(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_negative_color_click(self):
        self.imgout = c3.NegativeColor(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_logarit_click(self):
        self.imgout = c3.Logarit(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_power_click(self):
        self.imgout = c3.Power(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_piecewise_line_click(self):
        self.imgout = c3.PiecewiseLine(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_histogram_click(self):
        self.imgout = c3.Histogram(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_hist_equal_click(self):
        # self.imgout = c3.HistEqual(self.imgin)
        self.imgout = cv2.equalizeHist(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_hist_equal_color_click(self):
        self.imgout = c3.HistEqualColor(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_local_hist_click(self):
        self.imgout = c3.LocalHist(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_hist_stat_click(self):
        self.imgout = c3.HistStat(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_smooth_box_click(self):
        self.imgout = cv2.boxFilter(self.imgin, cv2.CV_8UC1, (21, 21))
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_smooth_gauss_click(self):
        # self.imgout = c3.SmoothGauss(self.imgin)
        self.imgout = cv2.GaussianBlur(self.imgin, (43, 43), 7.0)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_median_filter_click(self):
        self.imgout = cv2.medianBlur(self.imgin, 5)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_create_impulse_noise_click(self):
        self.imgout = c3.CreateImpulseNoise(self.imgin)
        cv2.imshow("ImageOut", self.imgout)

    def mnu_c3_sharp_click(self):
        self.imgout = c3.Sharp(self.imgin)
        cv2.imshow("ImageOut", self.imgout)


if __name__ == "__main__":
    app = App()
    app.mainloop()
