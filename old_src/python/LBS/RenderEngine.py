from Renderer import Renderer
from Scene import Scene
from WindowManager import WindowManager


class RenderEngine:
    """ This class represents the entire render engine. """

    TITLE = "Our LBS Display BITCH!"
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    FPS_CAP = 100

    def __init__(self):
        ''' Initializes a new render engine. Creates the display and inits the renderers. '''
        # has to be the first call because it calls glfw.init()
        self.wm = WindowManager(RenderEngine.WINDOW_WIDTH, RenderEngine.WINDOW_HEIGHT, RenderEngine.FPS_CAP)
        self.renderer = Renderer()

    def update(self):
        ''' Updates the display. '''
        self.wm.update()

    def render_scene(self, scene: Scene):
        '''
        Renders the scene to the screen.
        scene - the game scene.
        '''
        self.renderer.render_scene(scene)

    def close(self):
        ''' Cleans up the renderers and closes the display. '''
        self.renderer.clean_up()
        self.wm.close_window()      # has to be the last call, because it calls glfw.terminate()

    def window_should_close(self):
        return self.wm.window_should_close()
