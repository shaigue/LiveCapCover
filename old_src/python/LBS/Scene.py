import keyboard


class Camera:
    def __init__(self, scale):
        self.position = [0, 0, 0]
        self.pitch = 0      # high/low      (x axis)
        self.yaw = 0        # left/right    (y axis)
        self.roll = 0       # tilted        (z axis)
        self.scale = scale

    def move(self):
        if keyboard.is_pressed('i'):
            # z in
            self.position[2] -= 0.01 * self.scale
        if keyboard.is_pressed('o'):
            # z out
            self.position[2] += 0.01 * self.scale

        if keyboard.is_pressed('w'):
            # y up
            self.position[1] += 0.01 * self.scale
        if keyboard.is_pressed('s'):
            # y down
            self.position[1] -= 0.01 * self.scale

        if keyboard.is_pressed('d'):
            # x right
            self.position[0] += 0.01 * self.scale
        if keyboard.is_pressed('a'):
            # x left
            self.position[0] -= 0.01 * self.scale

        if keyboard.is_pressed('up'):
            self.pitch += 0.08 * self.scale
        if keyboard.is_pressed('down'):
            self.pitch -= 0.08 * self.scale
        if keyboard.is_pressed('left'):
            self.yaw += 0.08 * self.scale
        if keyboard.is_pressed('right'):
            self.yaw -= 0.08 * self.scale


class Scene:
    def __init__(self, entity, camera):
        # self.animatedModel = None
        self.entity = entity
        self.camera = camera
        self.lightDirection = None
        # self.lightDirection = new Vector3f(0, -1, 0)
        return

    # def getAnimatedModel(self) -> AnimatedModel:
    #     return self.animatedModel

    # return The direction of the light as a vector.
    # def getLightDirection(self) -> Vector3f:
    #     return self.lightDirection

    # def setLightDirection(self, lightDir: Vector3f):
    #     self.lightDirection.set(lightDir)
