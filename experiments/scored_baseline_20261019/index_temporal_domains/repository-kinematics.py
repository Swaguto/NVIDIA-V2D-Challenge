"""Small fixed/revolute URDF kinematics shared by preparation and diagnostics."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class URDFKinematics:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.joints = list(self.root.findall("joint"))
        self.names = [
            j.attrib["name"] for j in self.joints if j.attrib["type"] != "fixed"
        ]
        self.limits = {
            j.attrib["name"]: (
                float(j.find("limit").attrib["lower"]),
                float(j.find("limit").attrib["upper"]),
            )
            for j in self.joints
            if j.attrib["type"] != "fixed"
        }
        children = {j.find("child").attrib["link"] for j in self.joints}
        roots = {r.attrib["name"] for r in self.root.findall("link")} - children
        if len(roots) != 1:
            raise ValueError("Expected a connected single-root hand URDF")
        self.root_name = roots.pop()

    def poses(self, root_pose, positions):
        base = np.eye(4)
        base[:3, :3] = Rotation.from_quat(root_pose[3:7]).as_matrix()
        base[:3, 3] = root_pose[:3]
        result = {self.root_name: base}
        pending = self.joints.copy()
        while pending:
            remaining = []
            for joint in pending:
                parent = joint.find("parent").attrib["link"]
                if parent not in result:
                    remaining.append(joint)
                    continue
                origin = joint.find("origin")
                transform = np.eye(4)
                if origin is not None:
                    transform[:3, 3] = np.fromstring(
                        origin.attrib.get("xyz", "0 0 0"), sep=" "
                    )
                    transform[:3, :3] = Rotation.from_euler(
                        "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                    ).as_matrix()
                if joint.attrib["type"] != "fixed":
                    axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                    motion = np.eye(4)
                    if joint.attrib["type"] == "revolute":
                        motion[:3, :3] = Rotation.from_rotvec(
                            axis * positions[joint.attrib["name"]]
                        ).as_matrix()
                    else:
                        raise ValueError(
                            "Only fixed/revolute hand joints are supported"
                        )
                    transform = transform @ motion
                result[joint.find("child").attrib["link"]] = result[parent] @ transform
            if len(remaining) == len(pending):
                raise ValueError("URDF joint graph is disconnected or cyclic")
            pending = remaining
        return result
