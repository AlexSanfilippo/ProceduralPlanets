import glfw
from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians, atan2, sqrt
import glm as glm
from glm import vec3, mat4
from engine.simple_physics import find_intersection_ray_plane

class Camera:
    #todo: switch from Vector3 to glm vec3.  FollowCam already doing this!
    def __init__(self, camera_pos=[0.0, 0.0, 1.20], yaw=0, pitch=0, mouse_sensitivity=0.25):
        self.camera_pos = vec3(camera_pos)
        self.camera_front = vec3([0.0, 0.0, 1.0])
        self.camera_up = vec3([0.0, 1.0, 0.0])
        self.camera_right = vec3([1.0, 0.0, 0.0])

        self.mouse_sensitivity = mouse_sensitivity
        self.yaw = yaw
        self.pitch = pitch
        self.update_camera_vectors()

    def get_view_matrix(self):
        return matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 90:
                self.pitch = 90
            if self.pitch < -90:
                self.pitch = -90

        self.update_camera_vectors()

    def update_camera_vectors(self):
        front = vec3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.yaw)) * cos(radians(self.pitch))

        self.camera_front = glm.normalize(front)
        self.camera_right = glm.normalize(glm.cross(self.camera_front, vec3([0, 1, 0])))
        self.camera_up = glm.normalize(glm.cross(self.camera_right, self.camera_front))

    # Camera method for the WASD movement
    def process_keyboard(self, direction, velocity):
        if direction == "FORWARD":
            self.camera_pos += self.camera_front * velocity
        if direction == "BACKWARD":
            self.camera_pos -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_pos -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_pos += self.camera_right * velocity
        if direction == "PITCH_POS":
            self.pitch += velocity
            self.update_camera_vectors()
        if direction == "DOWN":
            self.camera_pos = list(vec3(self.camera_pos) - vec3(0.0, 1.5, 0.0))
            self.update_camera_vectors()
        if direction == "UP":
            self.camera_pos = list(vec3(self.camera_pos) + vec3(0.0, 1.5, 0.0))
            self.update_camera_vectors()

    def inherit_from_camera(self, other_camera):
        self.camera_pos = Vector3(other_camera.camera_pos)
        front = Vector3(other_camera.target_position) - self.camera_pos
        self.camera_front = vector.normalise(front)
        self.camera_right = vector.normalise(vector3.cross(self.camera_front, Vector3([0, 1, 0])))
        self.camera_up = vector.normalise(vector3.cross(self.camera_right, self.camera_front))


class FollowCamera(Camera):

    def __init__(self, camera_pos=[0.0, 0.0, 0.0], yaw=0, pitch=0, target_position=[0.0, 0.0, 0.0]):
        self.camera_pos = vec3(camera_pos)
        self.camera_front = vec3([0.0, 0.0, 1.0])
        self.camera_up = vec3([0.0, 1.0, 0.0])
        self.camera_right = vec3([1.0, 0.0, 0.0])
        self.target_position = vec3(target_position)
        self.offset = self.calculate_offset()
        self.offset_distance = glm.length(self.offset)
        self.mouse_sensitivity = 0.25 #0.25 default
        self.yaw = yaw #was -90
        self.pitch = pitch
        self.update_camera_vectors()
        self.angle = self.calclate_angle()

    def set_position(self, updated_position):
        self.camera_pos = vec3(updated_position)

    def rotate_camera_over_time(self, speed = 1.0):
        movement = glfw.get_time() * speed
        distance_multiplier = 0.70
        self.set_position([
            cos(movement) * self.offset_distance * distance_multiplier,
            cos(movement*0.5) * self.offset_distance * distance_multiplier,
            # self.camera_pos.y,
            # self.camera_pos.z,
            sin(movement*.77) * self.offset_distance * distance_multiplier
        ])
        self.look_at_target()
        self.update_camera_vectors()

    def calclate_angle(self):
        '''
        finds the angle on the XZ plane between the camera and the target
        '''
        return atan2(self.camera_pos.y - self.target_position.y, self.camera_pos.x - self.target_position.x)

    def calculate_offset(self):
        return self.camera_pos - self.target_position

    def update_position_by_target_position(self, target_position):
        self.target_position = target_position
        self.camera_pos = self.target_position + self.offset
        self.look_at_target()

    def look_at_target(self):
        """
        change camera pitch and yaw to look at a target object
        """
        return glm.lookAt(vec3(self.camera_pos), self.target_position, vec3(self.camera_up))
        #what does lookAt return?  how to adjust

    def update_camera_vectors(self):
        front = vec3([0.0, 0.0, 0.0])
        look_at = self.look_at_target()
        # front = look_at[2].xyz
        front = self.target_position - self.camera_pos
        self.camera_front = glm.normalize(front)
        self.camera_right = glm.normalize(glm.cross(self.camera_front, vec3([0, 1, 0])))
        self.camera_up = glm.normalize(glm.cross(self.camera_right, self.camera_front))

    def process_keyboard(self, direction, velocity):
        if direction == "YAW_CLOCKWISE":

            pos_from_center = self.camera_pos - self.target_position
            self.angle += 0.01
            distance_from_origin = sqrt(pow(pos_from_center.x,2) + pow(pos_from_center.z,2))
            pos_from_center.z = sin(self.angle)*distance_from_origin
            pos_from_center.x = cos(self.angle)*distance_from_origin
            self.camera_pos = pos_from_center + self.target_position
            self.update_camera_vectors()
            self.offset = self.calculate_offset()
        if direction == "YAW_COUNTERCLOCKWISE":

            pos_from_center = self.camera_pos - self.target_position
            self.angle -= 0.01
            distance_from_origin = sqrt(pow(pos_from_center.x, 2) + pow(pos_from_center.z, 2))
            pos_from_center.z = sin(self.angle) * distance_from_origin
            pos_from_center.x = cos(self.angle) * distance_from_origin
            self.camera_pos = pos_from_center + self.target_position
            self.update_camera_vectors()
            self.offset = self.calculate_offset()

    def inherit_from_camera(self, other_camera):
        self.camera_pos = vec3(other_camera.camera_pos)
        self.offset = self.camera_pos - self.target_position
        self.yaw = other_camera.yaw
        self.pitch = other_camera.pitch
        self.angle = self.calclate_angle()
        self.update_camera_vectors()

    def process_mouse_scroll(self, xoffset, yoffset):
        zoom_speed = 10.0
        front = glm.normalize(self.target_position - self.camera_pos)
        updated_camera_pos = glm.vec3(self.camera_pos)
        updated_camera_pos += front * yoffset * zoom_speed
        minimum_distance_camera_target = 1.0
        if glm.distance(self.target_position, updated_camera_pos) > minimum_distance_camera_target:
            self.camera_pos = updated_camera_pos
            self.offset = self.calculate_offset()
        else:
            # self.camera_pos = (-front * 2.001) + self.target_position

            # self.offset = self.calculate_offset()
            pass


    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        pass
        # xoffset *= self.mouse_sensitivity
        # yoffset *= self.mouse_sensitivity
        #
        # self.yaw += xoffset
        # self.pitch += yoffset
        #
        # if constrain_pitch:
        #     if self.pitch > 90:
        #         self.pitch = 90
        #     if self.pitch < -90:
        #         self.pitch = -90
        #
        # self.update_camera_vectors()

class SimulationCamera(FollowCamera):
    """
    Camera mode as seen in games likes banished, cities skylines, etc
    target to 'follow' (rotate,zoom around) is based on camera position and forward vector
    """

    def calculate_target(self):
        self.target_position = find_intersection_ray_plane(
            ray_origin=self.camera_pos,
            ray_direction=self.camera_front,
            plane=glm.vec4(0.0, 1.0, 0.0, 0.0)
        )

    def process_keyboard(self, direction, velocity):
        sim_front = vec3(self.camera_front.x, 0.0, self.camera_front.z)
        if direction == "FORWARD":
            self.camera_pos += sim_front * velocity * 2.5
            self.calculate_target()
        if direction == "BACKWARD":
            self.camera_pos += -sim_front * velocity * 2.5
            self.calculate_target()
        if direction == "LEFT":
            self.camera_pos += -self.camera_right * velocity * 2.5
            self.calculate_target()
        if direction == "RIGHT":
            self.camera_pos += self.camera_right * velocity * 2.5
            self.calculate_target()
        if direction == "YAW_CLOCKWISE":
            pos_from_center = self.camera_pos - self.target_position
            self.angle += 0.015 * velocity
            distance_from_origin = sqrt(pow(pos_from_center.x, 2) + pow(pos_from_center.z, 2))
            pos_from_center.z = sin(self.angle)*distance_from_origin
            pos_from_center.x = cos(self.angle)*distance_from_origin
            self.camera_pos = pos_from_center + self.target_position
            self.update_camera_vectors()
            self.offset = self.calculate_offset()
        if direction == "YAW_COUNTERCLOCKWISE":
            pos_from_center = self.camera_pos - self.target_position
            self.angle -= 0.015 * velocity
            distance_from_origin = sqrt(pow(pos_from_center.x, 2) + pow(pos_from_center.z, 2))
            pos_from_center.z = sin(self.angle) * distance_from_origin
            pos_from_center.x = cos(self.angle) * distance_from_origin
            self.camera_pos = pos_from_center + self.target_position
            self.update_camera_vectors()
            self.offset = self.calculate_offset()
        if direction == "DOWN":
            self.camera_pos -= vec3(0.0, 1.5, 0.0) * velocity * 2.0
            self.update_camera_vectors()
        if direction == "UP":
            self.camera_pos += vec3(0.0, 1.5, 0.0) * velocity * 2.0
            self.update_camera_vectors()


class RollableCamera(Camera):
    """
    A free-look camera with no pitch constraints and roll support.
    Roll rotates the camera_up and camera_right vectors around camera_front.
    Mouse look rotates the orientation vectors directly so that up/down/left/right
    movement is always relative to the camera's current rolled orientation.
    """

    def __init__(self, camera_pos=[0.0, 0.0, 1.20], yaw=0, pitch=0, mouse_sensitivity=0.25):
        self.roll = 0.0        # roll angle in degrees
        self._prev_roll = 0.0  # tracks applied roll so we only rotate by the delta
        super().__init__(camera_pos=camera_pos, yaw=yaw, pitch=pitch, mouse_sensitivity=mouse_sensitivity)

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=False):
        """
        Rotate orientation vectors directly using the camera's own axes so that
        mouse movement is always relative to the current rolled orientation.
          - xoffset rotates around camera_up   (yaw relative to camera)
          - yoffset rotates around camera_right (pitch relative to camera)
        """
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        # Rotate around camera_up for horizontal (yaw) mouse movement
        yaw_rot = glm.angleAxis(radians(-xoffset), self.camera_up)
        # Rotate around camera_right for vertical (pitch) mouse movement
        pitch_rot = glm.angleAxis(radians(-yoffset), self.camera_right)

        # Combine: apply pitch then yaw
        combined = yaw_rot * pitch_rot

        self.camera_front = glm.normalize(glm.vec3(combined * glm.vec4(self.camera_front, 0.0)))
        self.camera_right = glm.normalize(glm.vec3(combined * glm.vec4(self.camera_right, 0.0)))
        self.camera_up = glm.normalize(glm.vec3(combined * glm.vec4(self.camera_up, 0.0)))

    def update_camera_vectors(self):
        """
        Apply the roll delta (self.roll - self._prev_roll) by rotating
        camera_right and camera_up around the current camera_front.
        camera_front is left untouched — it is owned by process_mouse_movement.
        """
        delta_roll = radians(self.roll - self._prev_roll)
        self._prev_roll = self.roll

        roll_rot = glm.angleAxis(delta_roll, self.camera_front)
        self.camera_right = glm.normalize(glm.vec3(roll_rot * glm.vec4(self.camera_right, 0.0)))
        self.camera_up = glm.normalize(glm.vec3(roll_rot * glm.vec4(self.camera_up, 0.0)))

    # def get_view_matrix(self):
    #     return glm.lookAt(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

    def process_keyboard(self, direction, velocity):
        """Full keyboard handling using camera-relative axes for all directions."""
        if direction == "FORWARD":
            self.camera_pos += self.camera_front * velocity
        elif direction == "BACKWARD":
            self.camera_pos -= self.camera_front * velocity
        elif direction == "LEFT":
            self.camera_pos += self.camera_right * velocity
        elif direction == "RIGHT":
            self.camera_pos -= self.camera_right * velocity
        elif direction == "UP":
            self.camera_pos += self.camera_up * velocity
        elif direction == "DOWN":
            self.camera_pos -= self.camera_up * velocity
        elif direction == "ROLL_LEFT":
            self.roll -= velocity / 8.0
            self.update_camera_vectors()
        elif direction == "ROLL_RIGHT":
            self.roll += velocity / 8.0
            self.update_camera_vectors()

