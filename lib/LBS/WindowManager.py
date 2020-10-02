import glfw


class WindowManager:

    def __init__(self, width, height, FPS_CAP, title="Our LBS Window BITCH!"):
        try:
            print("creating window...")
            glfw.init()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            # TODO: you must be wondering why I chose version 3.3 even though our pyopengl is 3.1.
            # it's based on this comment from another project:
            # "If we are planning to use anything above 2.1 we must at least
            # request a 3.3 core context to make this work across platforms"
            # anyway, it works!
            self.window = glfw.create_window(
                width, height, title, monitor=None, share=None)
            glfw.set_window_pos(self.window, 400, 200)
            glfw.make_context_current(self.window)

        except Exception as e:
            print(e)
            print("Couldn't create window!")
            raise SystemExit

        # gl.GL11.glViewport(0, 0, width, height)
        print("window created")

    def update(self):
        glfw.swap_buffers(self.window)
        return

    def close_window(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
        return

    def window_should_close(self):
        return glfw.window_should_close(self.window)

    @staticmethod
    def set_time(time=0.0):
        glfw.set_time(time)

    @staticmethod
    def get_current_time():
        # TODO as always not sure how to implement this.
        glfw_time = glfw.get_time()
        glfw_timer_frequency = glfw.get_timer_frequency()
        # return glfw.get_time() * 1000 / glfw.get_timer_frequency()
        # return glfw.get_time() * 100000 / glfw.get_timer_frequency()
        return glfw.get_time() * 200000 / glfw.get_timer_frequency()
        # return glfw.get_time()
        # return Sys.getTime() * 1000 / Sys.getTimerResolution()
