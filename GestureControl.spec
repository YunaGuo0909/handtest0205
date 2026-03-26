# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for GestureControl
# Build: pyinstaller GestureControl.spec

import os
import mediapipe
import torch

# mediapipe 数据文件路径
mp_path = os.path.dirname(mediapipe.__file__)
# torch 路径 (for DLLs)
torch_path = os.path.dirname(torch.__file__)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('hand_landmarker.task', '.'),
        ('gesture_model.pt', '.'),
        ('gesture_data.csv', '.'),
        ('train_gesture.py', '.'),
        (os.path.join(mp_path, 'modules'), 'mediapipe/modules'),
        (os.path.join(mp_path, 'tasks', 'c'), 'mediapipe/tasks/c'),
        (os.path.join(torch_path, 'lib'), 'torch/lib'),
    ],
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.c',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.core',
        'mediapipe.tasks.python.core.mediapipe_c_bindings',
        'mediapipe.tasks.python.vision',
        'pynput',
        'pynput.keyboard',
        'pynput.keyboard._win32',
        'pynput.mouse',
        'pynput.mouse._win32',
        'cv2',
        'numpy',
        'torch',
        'torch.nn',
        'torch.utils',
        'torch.utils.data',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'scipy',
        'pandas',
        'pytest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GestureControl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 保留控制台窗口，方便看日志
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GestureControl',
)
