
class Component():
    """
    The base Component interface defines operations that can be altered by
    decorators.
    """
    def get_damage2carpart(self,image,pred_json,final_output):
        pass


class Damage(Component):
    """
    Concrete Components provide default implementations of the operations. There
    might be several variations of these classes.
    """
    def get_damage2carpart(self,image,pred_json,final_output):

        return pred_json,final_output

class DamageDecorator(Component):
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

    def get_damage2carpart(self,image,pred_json,final_output) -> str:
        return self._component.get_damage2carpart(image,pred_json,final_output)



class Component_Filter():
    """
    The base Component interface defines operations that can be altered by
    decorators.
    """

    def __call__(self,img):
        pass


class FilterDamage(Component_Filter):
    """
    Concrete Components provide default implementations of the operations. There
    might be several variations of these classes.
    """

    def __call__(self,image,pred_json):

        return pred_json

class FilterDecorator(Component_Filter):
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

    def __call__(self,image,pred_json) -> str:
        return self._component(image,pred_json)