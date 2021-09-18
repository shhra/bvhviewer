import moderngl_window as glw
from moderngl_window.conf import settings
from pathlib import Path
from bvhimporter import BVHdata
from animation import Animation
import numpy as np
import argparse
import time

# settings.WINDOW['class'] = "moderngl_window.context.glfw.Window"
# settings.WINDOW['gl_version'] = (4, 4)
# settings.WINDOW['title'] = "explorer"


class AnimViewer(glw.WindowConfig):
    gl_version = (4, 4)
    title = "explorer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Do program initialization
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_vert;
                in vec3 in_color;

                out vec3 v_color;

                void main() {
                    v_color = in_color;
                    gl_Position = vec4(in_vert, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 v_color;

                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            """,
        )
        self.animation = None
        self.pos = list()
        data = self.initialize_animation()
        self.vbo = None
        self.vao = None
        self.orig = self.pos
        self.i = 0

    def initialize_animation(self):
        data_path = "Projects/Study/PFNN/pfnn/data/animations/LocomotionFlat01_000.bvh"
        bvh_filepath = Path(Path.home(), data_path)
        bvh = BVHdata(bvh_filepath)
        positions = list()
        self.animation = Animation(bvh)
        self.animation.compute_frame_transform()
        for joint in bvh.joints:
            self.pos.append(joint.offset)


    def render(self, time, frametime):
        self.i += 1
        if self.i >= self.animation.bvh_data.frame_num:
            self.i = 0
        self.vao = self.animate(self.i)
        self.vao.render(mode=1)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Q:
                self.wnd.close()
                print("Process closed.")

    def animate(self, i):
        frame = self.animation.local_transforms[i]
        pos = np.array(self.pos)
        pos = np.concatenate((pos, np.ones(pos.shape[0]).reshape(-1, 1)), axis=1)
        result = frame @ pos.reshape(-1, 4, 1)
        result = result.reshape(-1, 4)
        divisor = np.concatenate([result[:, 3]] * 3, axis=0).reshape(-1, 3)
        data = result[:, 0:3] / divisor
        self.pos = data
        data *= 0.015
        colors = np.ones_like(data)
        data = np.concatenate((data, colors), axis=1).tolist()
        # print(data)
        positions = list()
        for idx, joint in enumerate(self.animation.bvh_data.joints):
            if joint.is_root():
                continue
            parent = data[joint.parent]
            current = data[idx]
            positions.append(current)
            positions.append(parent)

        data = np.array(positions)
        data = data.astype('f4').tobytes()
        self.vbo = self.ctx.buffer(data)
        # print(self.vbo)
        vao = self.vao
        vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_color')
        # time.sleep(self.animation.bvh_data.frame_info)
        return vao




def main():
    # data_path = "Projects/Study/PFNN/pfnn/data/animations/LocomotionFlat01_000.bvh"
    # bvh_filepath = Path(Path.home(), data_path)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", default=None)
    # args = ['--path', str(bvh_filepath)]
    # glw.parse_args(args, parser)
    # print(dir(glw))
    glw.run_window_config(AnimViewer)


if __name__ == "__main__":
    main()
