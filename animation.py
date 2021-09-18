import numpy as np


class Animation:
    def __init__(self, bvh_data):
        self.bvh_data = bvh_data
        self.transforms = None
        positions = []
        for joint in bvh_data.joints:
            positions.append(joint.offset + [1.0])
        self.positions = np.array(positions)
        self.local_transforms = np.zeros(
            (self.bvh_data.frame_num, len(self.bvh_data.joints), 4, 4)
        )

    def compute_frame_transform(self):
        index = 0
        frame = np.array(self.bvh_data.frames)
        for i, joint in enumerate(self.bvh_data.joints):
            if joint.channels[0] == "3":
                # calculate rotations
                self.calculate_transforms(joint, frame[:, index : index + 3], i)
                index += 3
            elif joint.channels[0] == "6":
                # Calculate offsets as well as rotations
                self.calculate_transforms(joint, frame[:, index : index + 6], i)
                index += 6

    def calculate_transforms(self, joint, frame_data, joint_idx):
        # print(f"{joint.name} ----> {frame_data}")
        if joint.is_root():
            offset = frame_data[:, 0:3] + np.array(joint.offset)
            rotation = self.calculate_rotation(frame_data[:, 3:], joint.channels[4:])
            transform = np.dstack((rotation, np.zeros_like(offset)))
            final = np.array([[0, 0, 0, 1]] * frame_data.shape[0]).reshape(-1, 1, 4)
            transform = np.concatenate((transform, final), axis=1)
            translation = np.zeros_like(transform)
            translation[:, 0, 0] = 1
            translation[:, 1, 1] = 1
            translation[:, 2, 2] = 1
            translation[:, 3, 3] = 1
            translation[:, 0:3, 3] = offset
            # final = np.matmul(transform, translation)
            final = np.matmul(translation, transform)
            self.local_transforms[:, joint_idx, :, :] = final
            # print(final)
        else:
            offset = np.zeros((frame_data.shape[0], 3)) + np.array(joint.offset)
            rotation = self.calculate_rotation(frame_data, joint.channels[1:])
            transform = np.dstack((rotation, np.zeros_like(offset)))
            final = np.array([[0, 0, 0, 1]] * frame_data.shape[0]).reshape(-1, 1, 4)
            transform = np.concatenate((transform, final), axis=1)
            translation = np.zeros_like(transform)
            translation[:, 0, 0] = 1
            translation[:, 1, 1] = 1
            translation[:, 2, 2] = 1
            translation[:, 3, 3] = 1
            translation[:, 0:3, 3] = offset
            # final = np.matmul(transform, translation)
            final = np.matmul(translation, transform)
            parent_transform = self.local_transforms[:, joint.parent, :, :]
            self.local_transforms[:, joint_idx, :, :] = np.matmul(
                parent_transform, final
            )

    def calculate_rotation(self, rotation, channel):
        rot = np.zeros((rotation.shape[0], 3, 3))
        rot[:, 0, 0] = 1
        rot[:, 1, 1] = 1
        rot[:, 2, 2] = 1
        # return rot
        if channel == "ZYX":
            z_theta = rotation[:, 0] * np.pi / 180.0
            y_theta = rotation[:, 1] * np.pi / 180.0
            x_theta = rotation[:, 2] * np.pi / 180.0
        elif channel == "ZXY":
            z_theta = rotation[:, 0] * np.pi / 180.0
            x_theta = rotation[:, 1] * np.pi / 180.0
            y_theta = rotation[:, 2] * np.pi / 180.0
        else:  # XYZ
            x_theta = rotation[:, 0] * np.pi / 180.0
            y_theta = rotation[:, 1] * np.pi / 180.0
            z_theta = rotation[:, 2] * np.pi / 180.0

        fx_rotation = lambda x_angle: np.array(
            [
                [np.ones_like(x_angle), np.zeros_like(x_angle), np.zeros_like(x_angle)],
                [np.zeros_like(x_angle), np.cos(x_angle), -np.sin(x_angle)],
                [np.zeros_like(x_angle), np.sin(x_angle), np.cos(x_angle)],
            ]
        )

        fy_rotation = lambda y_angle: np.array(
            [
                [np.cos(y_angle), np.zeros_like(y_angle), np.sin(y_angle)],
                [np.zeros_like(y_angle), np.ones_like(y_angle), np.zeros_like(y_angle)],
                [-np.sin(y_angle), np.zeros_like(y_angle), np.cos(y_angle)],
            ]
        )

        fz_rotation = lambda z_angle: np.array(
            [
                [np.cos(z_angle), -np.sin(z_angle), np.zeros_like(z_angle)],
                [np.sin(z_angle), np.cos(z_angle), np.zeros_like(z_angle)],
                [np.zeros_like(z_angle), np.zeros_like(z_angle), np.ones_like(z_angle)],
            ]
        )

        x_rotation = fx_rotation(x_theta).transpose(2, 0, 1)
        y_rotation = fy_rotation(y_theta).transpose(2, 0, 1)
        z_rotation = fz_rotation(z_theta).transpose(2, 0, 1)

        if channel == "ZYX":
            rot = np.matmul(z_rotation, np.matmul(y_rotation, x_rotation))
        elif channel == "ZXY":
            rot = np.matmul(z_rotation, np.matmul(x_rotation, y_rotation))
        else:
            rot = np.matmul(x_rotation, np.matmul(y_rotation, z_rotation))
        # print(rot.shape)
        return rot
