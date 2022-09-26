#!/usr/bin/env python3

import rospy

from network import suction_graspnet


if __name__ == "__main__":
    rospy.init_node("suction_graspnet", anonymous=True)
    
    # EXIT
