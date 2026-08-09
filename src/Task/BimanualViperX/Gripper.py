class Gripper:

    def __init__(self, cfg):
        self.POSITION_OPEN = cfg.position.open
        self.POSITION_CLOSE = cfg.position.close
        self.JOINT_OPEN = cfg.joint.open
        self.JOINT_CLOSE = cfg.joint.close
        self.JOINT_MID = (self.JOINT_OPEN + self.JOINT_CLOSE)/2

    def position_normalize(self, x):
        return (x - self.POSITION_CLOSE) / (self.POSITION_OPEN - self.POSITION_CLOSE)

    def position_unnormalize(self, x):
        return x * (self.POSITION_OPEN - self.POSITION_CLOSE) + self.POSITION_CLOSE

    def joint_normalize(self, x):
        return (x - self.JOINT_CLOSE) / (self.JOINT_OPEN - self.JOINT_CLOSE)

    def joint_unnormalize(self, x):
        return x * (self.JOINT_OPEN - self.JOINT_CLOSE) + self.JOINT_CLOSE

    def velocity_normalize(self, x):
        return x / (self.POSITION_OPEN - self.POSITION_CLOSE)

    def pos2joint(self, x):
        return self.position_normalize(x) * (self.JOINT_OPEN - self.JOINT_CLOSE) + self.JOINT_CLOSE

    def joint2pos(self, x):
        return self.joint_normalize(x) * (self.POSITION_OPEN - self.POSITION_CLOSE) + self.POSITION_CLOSE
