# the camera class must not be changed
# the intrisics are fixed for the challenge
class Camera:
    def __init__(self, H=320, W=320, focal=277, model="PINHOLE"):
        self.H = H                      # image height
        self.W = W                      # image width
        self.focal = focal              # focal length
        self.fx = focal                 # focal length in x (same as focal)
        self.fy = focal                 # focal length in y (same as focal)
        self.cx = W / 2.0 - 0.5         # principal point x-coordinate
        self.cy = H / 2.0 - 0.5         # principal point y-coordinate
        self.model = model              # camera model

    def __repr__(self):
        return (f"Camera(model={self.model}, H={self.H}, W={self.W}, "
                f"fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})")
