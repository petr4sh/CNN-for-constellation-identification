import numpy as np
from PIL import Image
import pylab

from cntk.ops.functions import load_model
import cntk


z = load_model("star_nn.model")


def star(file):
    img = Image.open(file)
    pixels = img.load()
    w, h = img.size
    braight = []
    for x in range(w):
        for y in range(h):
            r, g, b = pixels[x, y]
            braight.append((r + g + b) // 3)
    return braight


def plot(x):
    fig = pylab.figure()
    ax = fig.add_subplot(1,2,1)
    pylab.imshow(x.reshape(300,300))
    p = cntk.softmax(z)
    hist = p.eval(x)
    ax = fig.add_subplot(1,2,2)
    pylab.bar(np.arange(85), hist[0])
    pylab.xticks(np.arange(85))
    pylab.show()


plot((np.array(star('тест_орион.jpg')).astype(np.float32) / 256.0).reshape(-1,1,300,300))

