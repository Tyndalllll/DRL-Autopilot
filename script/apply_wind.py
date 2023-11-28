from os import wait
import rospy
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates
import numpy as np

class wind_controller:
    def __init__(self) -> None:
        use_ros = False
        if use_ros:
            rospy.init_node('wind_controller', anonymous=True)
            # Publisher
            self.puber = rospy.Publisher('wind_force', Float32MultiArray, queue_size=1)

            # Subscriber
            rospy.Subscriber('gazebo/model_states', ModelStates, self.suber)

            # Frequency
            self.rate = rospy.Rate(100)

            # Initialize variables type
            self.wind_force = Float32MultiArray()

            self.wind_force.data = [0., 0., 0.]
        self.pos = np.array([0, 0, 0], dtype=float)
        self.wind_dist = 4
        self.width = 1.0
        # Buildings
        self.buildings = np.array([[-1.0, -1.0, 0.75], [0.0, 1.5, 0.6], [2.0, 0.0, 0.5]])

    def run(self):

        print(f"Start to apply wind")
        while not rospy.is_shutdown():
            if self.Within_Wind_Zone(self.pos):
                self.wind_force.data = [2., 0., 0.]
            else:
                self.wind_force.data = [0., 0., 0.]

            print(f"wind force: [{self.wind_force.data}]")
            self.puber.publish(self.wind_force)
            self.rate.sleep()

    def Within_Wind_Zone(self, pos):
        flag = True
        num = np.shape(self.buildings)[0]
        for i in range(num):
            if self.Behind_the_Building(self.buildings[i, :], pos):
                flag = False

        return flag

    def Behind_the_Building(self, pos_of_build, pos):
        relative_pos = pos - pos_of_build
        if (relative_pos[0] < 0) or (relative_pos[0]> self.wind_dist + self.width / 2) or (np.abs(relative_pos[1]) > self.width / 2) or (relative_pos[2] > pos_of_build[2]):
            return False
        else:
            wind_width = self.width / 2 * (self.wind_dist + self.width / 2 - relative_pos[0]) / self.wind_dist

            if (np.abs(relative_pos[1]) > wind_width):
                return False
            else:
                return True

    def suber(self, data):
        pose = data.pose[2]
        self.pos = np.array([pose.position.x, pose.position.y, pose.position.z])


if __name__ == "__main__":
    controller = wind_controller()
    controller.run()
