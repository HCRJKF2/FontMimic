import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, opt):
        super(BaseDataset, self).__init__()

        self.letter_image_size = opt["letter_image_size"]
        self.letter_size = opt["letter_size"]

        self.font_root_path = opt["font_root_path"]
        font_dir = os.path.join(self.font_root_path, "font")
        if not os.path.isdir(font_dir):
            raise FileNotFoundError(f"Font directory does not exist: {font_dir}")
        self.fonts = sorted(
            os.path.join(font_dir, font)
            for font in os.listdir(font_dir)
            if os.path.isfile(os.path.join(font_dir, font))
        )
        if not self.fonts:
            raise RuntimeError(f"No font files found in: {font_dir}")

    def draw_letter(self, ch, font=None, font_path=None):
        if font is None:
            if font_path is None:
                font = ImageFont.load_default(size=self.letter_size)
            else:
                font = ImageFont.truetype(font_path, self.letter_size)
        ascent, descent = font.getmetrics()

        cur_w = max(self.letter_image_size, int((ascent + descent) * 1.1))
        cur_h = cur_w

        bbox = font.getbbox(ch)
        w = bbox[2] - bbox[0]
        x = (cur_w - w) // 2 - bbox[0]
        y = cur_h // 2 - (ascent + descent) // 2

        image = np.ones((cur_h, cur_w, 3), dtype=np.uint8) * 255

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((x, y), ch, fill=(0, 0, 0), font=font)

        image = image.resize((self.letter_image_size, self.letter_image_size), Image.Resampling.LANCZOS)
        return image

    def __len__(self):
        return len(self.fonts)
