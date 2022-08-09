import cv2
import numpy as np

class Component():
    """
    The base Component interface defines operations that can be altered by
    decorators.
    """

    def __call__(self,img):
        pass


class PreProcessImg(Component):
    """
    Concrete Components provide default implementations of the operations. There
    might be several variations of these classes.
    """

    def __call__(self,img):

        return img

class Decorator(Component):
    """
    The base Decorator class follows the same interface as the other components.
    The primary purpose of this class is to define the wrapping interface for
    all concrete decorators. The default implementation of the wrapping code
    might include a field for storing a wrapped component and the means to
    initialize it.
    """

    _component: Component = None

    def __init__(self, component: Component) -> None:
        self._component = component

    @property
    def component(self):

        return self._component

    def __call__(self,img) -> str:
        return self._component(img)


class HistogramEqualization (Decorator):

    def __call__(self,img):
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.equalizeHist(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return self.component(final)

class ExposeImage(Decorator):
    def __call__(self,img):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.3) * 255.0, 0.4, 255)
        img = cv2.LUT(img, lookUpTable)
        
        return self.component(img)

class EnhanceImage(Decorator):
    def __call__(self,img):
        img = cv2.detailEnhance(img, sigma_r=10, sigma_s=0.15)
        return self.component(img)

class ConvertScaleAbs(Decorator):
    def __call__(self,img):
        img = cv2.convertScaleAbs(img, alpha=1, beta=45)
        return self.component(img)
