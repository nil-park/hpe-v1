import os
from turtle import bgcolor
import trimesh
import pyrender
import numpy as np

DIR = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(DIR, 'data/template.obj')

class Renderer:

    def __init__(self, resolution):
        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        self.mesh = pyrender.Mesh.from_trimesh(trimesh.load(TEMPLATE_PATH), smooth=True)
        self.renderer = renderer

    def __call__(self, mat):
        camera = pyrender.camera.OrthographicCamera(xmag=0.13, ymag=0.13)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3000)
        light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]])
        scene = pyrender.Scene(ambient_light=[.05, .05, .05], bg_color=[0,0,0,0])
        m = np.eye(4)
        m[:3,:3] = mat[:3,:3]
        scene.add(self.mesh, pose=m)
        scene.add(camera, pose=np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 5],[0, 0, 0, 1]]))
        scene.add(light, pose=light_pose)
        color, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color
