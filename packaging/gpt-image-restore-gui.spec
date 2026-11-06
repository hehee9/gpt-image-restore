from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
package = root / "gptimg_restore"
hidden = [name for name in collect_submodules("gptimg_restore")
          if "._gpu" not in name and ".gui.__main__" not in name]
datas = [(str(package / "blue_noise_thresholds.npy"), "gptimg_restore"),
         (str(package / "gui" / "web"), "gptimg_restore/gui/web")]
a = Analysis([str(package / "gui" / "app.py")], pathex=[str(root)],
             hiddenimports=hidden, datas=datas,
             excludes=["cupy", "cupyx", "gptimg_restore._gpu", "gptimg_restore._gpu_defocus",
                       "gptimg_restore._gpu_lattice", "gptimg_restore._gpu_mottle",
                       "gptimg_restore._gpu_periodic", "gptimg_restore._gpu_textgap"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="gpt-image-restore-gui", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="gpt-image-restore-gui")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="GPT Image Restore.app", bundle_identifier="com.hehee9.gpt-image-restore")
