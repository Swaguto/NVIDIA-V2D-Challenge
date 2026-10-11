import tempfile
import unittest
from pathlib import Path
import numpy as np
from verify_motion_match import compare


class MotionMatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name); self.reference=root/'reference'; self.candidate=root/'candidate'
        self.write(self.reference); self.write(self.candidate)

    def write(self, root, position=0.0, angle=0.0, quaternion_sign=1):
        (root/'bundle').mkdir(parents=True,exist_ok=True); (root/'retarget').mkdir(exist_ok=True)
        frames=np.array([0]); valid=np.ones((1,2),dtype=bool); times=np.array([0.0])
        joints=np.zeros((1,2,21,3)); vertices=np.zeros((1,2,778,3))
        joints[...,0]=position; vertices[...,0]=position
        np.savez(root/'bundle/episode.npz',frame_indices=frames,timestamps_s=times,
                 hand_valid=valid,hand_directly_observed=valid,hand_joints_world=joints,hand_vertices_world=vertices)
        q=np.zeros((1,2,14)); q[:,:,6]=quaternion_sign; q[:,:,7]=angle
        np.savez(root/'retarget/retarget.npz',frame_indices=frames,timestamps_s=times,
                 robot_valid=valid,source_hand_valid=valid,robot_qpos=q)

    def test_identical_pass(self):
        self.assertEqual(compare(self.reference,self.candidate)['status'],'motion_matches')

    def test_position_drift_fails(self):
        self.write(self.candidate,position=0.00015)
        self.assertEqual(compare(self.reference,self.candidate)['status'],'failed')

    def test_joint_drift_fails(self):
        self.write(self.candidate,angle=0.002)
        self.assertEqual(compare(self.reference,self.candidate)['status'],'failed')

    def test_equivalent_quaternion_sign_passes(self):
        self.write(self.candidate,quaternion_sign=-1)
        self.assertEqual(compare(self.reference,self.candidate)['status'],'motion_matches')


if __name__=='__main__':unittest.main()
