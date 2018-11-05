import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool, ZoomInTool, ZoomOutTool
from io import BytesIO
import base64
import skimage

from PIL import Image


def getmmfile():
    mh = np.memmap('snail5_phago.mm', mode='r',
                   dtype=np.int32, shape=(4,))

    shape = tuple(mh)
    del mh
    mm = np.memmap('snail5_phago.mm', mode='r',
                   dtype=np.float32, shape=shape)

    return mm

def maketooltip():
    f = open("/media/cjw/PythonLib/cjwdeeplearning/autoencoder/tooltip.html",'r')
    t = f.read()
    return t

def to_png(image):
    image = image.astype(np.uint8)
    out = BytesIO()
    ia = Image.fromarray(image)
    ia.save(out, format='png')
    return out.getvalue()

    
def encode_images(images):
    urls=[]
    for im in images:
        png = to_png(im)
        url = 'data:image/png;base64,'
        url += base64.b64encode(png).decode('utf-8')
        urls.append(url)
    return urls

def bokehplot(images, df):
    fimages = 255*images
    bimages = fimages.astype(np.int32)
    urls = encode_images(bimages)

 #   x = np.random.randn(mm.shape[0])
 #   y = np.random.randn(mm.shape[0])

 #   data = {'x':x, 'y':y}
#    df = pd.DataFrame(data)
#    ts = df['x'].astype(str)
    df['source_text']=df['agc']
    df['image_urls'] = urls

    hv = HoverTool(tooltips=maketooltip())

    src = ColumnDataSource(data=df)
    TOOLS = [ZoomInTool(), ZoomOutTool(), hv]
    p = figure(plot_width=600, plot_height=600,
               title="Hey, that's nice", tools=TOOLS)

    p.circle('x','y', size=4, source=src)
    return p
