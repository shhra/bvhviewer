from pathlib import Path


class Node:
    """Represents a node in skeleton"""

    def __init__(self, name, parent_id, data):
        self.name = name
        self.parent = parent_id
        self.offset = data["offset"]
        self.channels = data["channels"]
        self.channel_len = len(self.channels)

    def is_root(self):
        return self.parent == -1

    def __repr__(self):
        return f"{self.name}:\n\tOffset: {self.offset}\
        \n\tChannels: {self.channels}\n\tparent: {self.parent}"


class BVHdata:
    """
    Reads a single biovision hierarchy file.

    It requires parsing the hierarchial data as well as motion data. Therefore, when we
    are reading the files, we store the motion data in
    a list where a single element of the list denotes the frame. We store the
    hierarchial data in a tree structure.
    """

    def __init__(self, filepath):
        self.data_path = Path(Path.home(), filepath)
        self.index = 0
        self.lines = list()
        self.joints = list()
        self.frame_num = 0
        self.frame_info = 0
        self.frames = list()
        self.frame_time = 0
        self.read()

    def read(self):
        with open(self.data_path, "r") as f:
            self.lines = f.readlines()
            if "HIERARCHY" in self.lines[self.index]:
                print("Parsing Hierarchy")
                self.index += 1
                self.parse_hierarchy(len(self.joints) - 1)
            print("Hierarchy Parsed")
            if "MOTION" in self.lines[self.index]:
                print("Parsing Motion")
                self.parse_motion()

    def parse_hierarchy(self, parent):
        if self.out_of_bound():
            return
        cur_line = self.lines[self.index]
        if "ROOT" in cur_line or "JOINT" in cur_line:
            return self.parse_root(parent)
        cur_line = self.lines[self.index]
        if "MOTION" in cur_line:
            return self.parse_motion()

    def parse_root(self, parent):
        """
        Creates a new node for the given joint location.
        Once the node is created it it goes on the parse the children
        or inner nodes depending if it JOINT or End Site
        """
        stack = list()
        stack.insert(0, parent)
        while len(stack) > 0 and not self.out_of_bound():
            parent = stack[0]
            name, data = self.fetch_data()
            node = Node(name, parent, data)
            self.joints.append(node)
            stack.insert(0, len(self.joints) - 1)
            if self.skip_end():
                while "}" in self.lines[self.index]:
                    self.index += 1
                    stack.remove(stack[0])
                    if self.out_of_bound():
                        self.index = self.index - 1
                        print("Error no Motion data found.")
                        return
                    if "MOTION" in self.lines[self.index]:
                        return

    def skip_end(self):
        if self.out_of_bound():
            return False
        if "End" in self.lines[self.index]:
            # parent = len(self.joints) - 1
            self.index += 2
            # print("Working in end.\n")
            # offset = self.lines[self.index][:-1].split(" ")
            # data = {"offset": offset[1:], "channels": ""}
            # node = Node("Dummy", parent, data)
            # self.joints.append(node)
            self.index += 2
            return True
        return False

    def out_of_bound(self):
        return self.index >= len(self.lines)

    def fetch_data(self):
        index = self.index
        if self.out_of_bound():
            return None, None
        name = self.lines[index][:-1].split(" ")[-1]
        index += 2
        offset = [float(x) for x in self.lines[index][:-1].split(" ")[1:]]
        index += 1
        channels = self.lines[index][:-1].split(" ")[1:]
        if '' in channels:
            channels.remove('')
        index += 1
        data = {"offset": offset, "channels": "".join(x[0] for x in channels)}
        self.index = index
        return name, data

    def parse_motion(self):
        index = self.index
        index += 1
        self.frame_num = int(self.lines[index][:-1].split(" ")[1])
        index += 1
        self.frame_info = float(self.lines[index][:-1].split(" ")[-1])
        index += 1
        i = 0
        while index <= self.frame_num:
            line = self.lines[index][:-2].split(" ")
            line = [float(x) for x in line]
            index += 1
            self.frames.append(line)
            # if i == 25:
            #     break
            i += 1
        self.frame_num = len(self.frames)


def main():
    data_path = "Projects/Study/PFNN/pfnn/data/animations/LocomotionFlat01_000.bvh"
    # data_path = "Projects/animated/test.bvh"
    bvh_filepath = Path(data_path)
    bvh = BVHdata(bvh_filepath)
    bvh.read()
    # for joint in bvh.joints:
    #     print(joint)
    print(bvh.frames)


if __name__ == "__main__":
    main()
