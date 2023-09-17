import numpy as np
import logging
import math


class Kinematics:
    def __init__(self, W, H, R, S, max_rpm, min_rpm):
        """Initializing the parameters

        Args:
            W (int): Width of the bot
            H (int): Height of the bot
            R (int): Wheel radius
            S (int): Max speed of the bot
            max_rpm (int): Maximum RPM of the bot
            min_rpm (int): Minimum RPM of the bot
        """
        self.W = W
        self.H = H
        self.R = R
        self.S = S
        self.vel = []
        self.max_rpm = max_rpm
        self.min_rpm = min_rpm
        self.position = [0, 0, 0]
        self.actual_position = [0, 0, 0]

        self.x = []
        self.y = []
        self.J_Inv = np.array([
            [1/self.R, -W/self.R],
            [1/self.R, W/self.R]
        ])
        self.J = np.array([
            [self.R/2, self.R/2],
            [-self.R/(2*W), self.R/(2*W)]
        ])
        logging.basicConfig(filename="server_log.log",
                            format='%(asctime)s %(message)s',
                            filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.l_rpm_val = []
        self.r_rpm_val = []

    def Ro(self, phi):
        """
        This method return the matrics[3x1] of the given angle

        Args:
            phi (int): It should be between PI to -PI

        Returns:
            Array : It return Array(3x1)
        """
        r = np.array([
            [np.cos(phi), np.sin(phi), 0],
            [0, 0, 1]
        ])
        return r

    def RT(self, phi):
        """
        This method return the matrics[1x3] of the given angle

        Args:
            phi (int): It should be between PI to -PI

        Returns:
            Array : It return Array(1x3)
        """
        r = np.array([
            [np.cos(phi), 0],
            [np.sin(phi), 0],
            [0, 1]
        ])
        return r

    def toTangent(self, RPM):
        """
        This method is used to get the tangent speed for the given RPM

        Args:
            RPM (int): Its the RPM which has to be converted

        Returns:
            Integer : Return the tangent speed
        """
        c = 2 * np.pi
        return c * RPM / 60

    def normalize(self, dphi):
        """
        This method is used to normalize the angle between PI to -PI

        Args:
            dphi (int): Takes input angle which has to be normalized

        Returns:
            int: Normalized angle is returned
        """
        if (dphi > np.pi):
            dphi = -(2*np.pi - dphi)
        elif (dphi < -np.pi):
            dphi = 2*np.pi + dphi

        return dphi

    def forwardKine(self, left_rpm, right_rpm, timestep):
        """
        This method is used to calculate the forward kinematics using given left wheel RPM and right wheel RPM
        and updates the position of end effector

        Args:
            left_rpm (int): RPM value
            right_rpm (int): RPM value
            timestep (int): Time in miliseconds
        """
        self.l_rpm_val.append(left_rpm)
        self.r_rpm_val.append(right_rpm)
        wheelVel = np.array([
            [self.toTangent(left_rpm)],
            [self.toTangent(right_rpm)]])

        Rt = self.RT(self.actual_position[2])

        Eta = np.round(
            np.array(np.dot(Rt, np.dot(self.J, wheelVel))), decimals=4)

        self.actual_position[0] += timestep*Eta[0][0]
        self.actual_position[1] += timestep*Eta[1][0]
        self.actual_position[2] += timestep*Eta[2][0]
        return self.actual_position

    def inverseKinematics(self, next_position):
        """
        This method is used to calculate the inverse kinematics using given next position 
        Set the 
        Args:
            next_position (list[x,y]): It takesnext position [x,y]

        Returns:
            list: List of velocities to reach that position
        """
        self.velocities = []
        self.logger.warning(f"Moving from {self.position} to {next_position}")

        x1, y1 = self.position[0:-1]
        x2, y2 = next_position

        next_angle = np.arctan2((y2-y1), (x2-x1))

        while abs(round(self.position[2], 5)-round(next_angle, 5)) > 0.0005:
            dphi = abs(self.normalize(self.position[2]-next_angle))
            time = 2*(0.0064*dphi)/0.01
            self.logger.info("Time given :"+str(time))

            self.logger.info("angle  : "+str(self.position[2]))
            self.logger.info("angle expected : "+str(next_angle))
            self.position[2] = self.normalize(
                self.move(0, 0, self.position[2], 0, 0, next_angle, time)[2])
            self.logger.info("angle reached : "+str(self.position[2])+"\n\n")

        distance = round(np.sqrt((y2-y1)**2+(x2-x1)**2), 2)
        time = (distance/(self.S))
        res = self.move(0, 0, 0, distance, 0, 0, time)
        self.logger.info(res)
        self.position[0] += res[0]*math.cos(self.position[2])
        self.position[1] += res[0]*math.sin(self.position[2])

        return self.velocities

    def move(self, x1, y1, prev_angle, x2, y2, next_angle, time):
        i = 0
        if x2 == 0:
            timestep = time/100
        else:
            timestep = time/10

        while i < time:
            dx = x2-x1
            dy = y2-y1
            dphi = next_angle - prev_angle
            dtime = abs(time-i)

            if (dphi > np.pi):
                dphi = 2*np.pi - dphi

            elif (dphi < -np.pi):
                dphi = 2*np.pi + dphi

            Botvel = np.array([[dx/dtime],
                              [dy/dtime],
                               [dphi/dtime]]).reshape(3, 1)

            Rot = self.Ro(prev_angle)

            Vs = 60*(np.array(np.dot(self.J_Inv, np.dot(Rot, Botvel))
                              ).reshape(2, 1))/(2*np.pi)

            absVs = np.abs(Vs)

            maxAbsV = np.max(absVs)

            if maxAbsV > self.max_rpm:
                Vs = [v * self.max_rpm / maxAbsV for v in Vs]

            if abs(Vs[0][0]) < self.min_rpm or abs(Vs[1][0]) < self.min_rpm:
                Vs[0][0] = 0
                Vs[1][0] = 0
                i += timestep

                continue

            wheelVel = np.array([
                [self.toTangent(Vs[0][0])],
                [self.toTangent(Vs[1][0])]])

            Rt = self.RT(prev_angle)

            Eta = np.round(
                np.array(np.dot(Rt, np.dot(self.J, wheelVel))), decimals=4)

            x1 += timestep*Eta[0][0]
            y1 += timestep*Eta[1][0]
            prev_angle += timestep*Eta[2][0]

            self.x += [x1]
            self.y += [y1]

            self.vel.append((Vs[0][0], Vs[1][0], timestep*1000))
            i += timestep

        return (x1, y1, prev_angle)
