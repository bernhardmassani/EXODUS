# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build file for EXODUS v0.1.4:  pyinstaller EXODUS_main.spec
import os
from PyInstaller.utils.hooks import collect_all

# Images loaded at run time from the program folder (window icon, log-scale
# easter egg). icon.ico (Windows taskbar icon) is optional.
datas = [(f, '.') for f in ('icon.png', 'icon.ico', 'RSW.png') if os.path.exists(f)]
binaries = []
hiddenimports = []
tmp_ret = collect_all('pymatgen')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['EXODUS_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='EXODUS_main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'] if os.path.exists('icon.ico') else None,
)
