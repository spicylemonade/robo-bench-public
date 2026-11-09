"""Fetch third-party robot assets (pinned upstream commits) into roboticsbench/assets/vendor.

Sources
-------
* Unitree G1 (29-DoF + Dex3 hands) MJCF, meshes, and the GR00T whole-body-control
  lower-body policies (Balance + Walk ONNX) from NVlabs/GR00T-WholeBodyControl.
  Code: Apache-2.0.  Policy weights: NVIDIA Open Model License (commercial use OK, attribution).
* Franka Emika Panda MJCF + meshes from google-deepmind/mujoco_menagerie (Apache-2.0).

Usage:  python -m roboticsbench.assets.fetch [--force]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

VENDOR = Path(os.environ.get("RBENCH_ASSETS_DIR") or (Path(__file__).resolve().parent / "vendor"))

GR00T_REPO = "NVlabs/GR00T-WholeBodyControl"
GR00T_SHA = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"
GR00T_G1 = "decoupled_wbc/sim2mujoco/resources/robots/g1"

MENAGERIE_REPO = "google-deepmind/mujoco_menagerie"
MENAGERIE_SHA = "c96a32d28fb5da84da38c1da4d749e7a13212855"

G1_MESHES = [
    "head_link.STL", "left_ankle_pitch_link.STL", "left_ankle_roll_link.STL", "left_elbow_link.STL",
    "left_hand_index_0_link.STL", "left_hand_index_1_link.STL", "left_hand_middle_0_link.STL",
    "left_hand_middle_1_link.STL", "left_hand_palm_link.STL", "left_hand_thumb_0_link.STL",
    "left_hand_thumb_1_link.STL", "left_hand_thumb_2_link.STL", "left_hip_pitch_link.STL",
    "left_hip_roll_link.STL", "left_hip_yaw_link.STL", "left_knee_link.STL",
    "left_shoulder_pitch_link.STL", "left_shoulder_roll_link.STL", "left_shoulder_yaw_link.STL",
    "left_wrist_pitch_link.STL", "left_wrist_roll_link.STL", "left_wrist_yaw_link.STL",
    "logo_link.STL", "pelvis.STL", "pelvis_contour_link.STL", "right_ankle_pitch_link.STL",
    "right_ankle_roll_link.STL", "right_elbow_link.STL", "right_hand_index_0_link.STL",
    "right_hand_index_1_link.STL", "right_hand_middle_0_link.STL", "right_hand_middle_1_link.STL",
    "right_hand_palm_link.STL", "right_hand_thumb_0_link.STL", "right_hand_thumb_1_link.STL",
    "right_hand_thumb_2_link.STL", "right_hip_pitch_link.STL", "right_hip_roll_link.STL",
    "right_hip_yaw_link.STL", "right_knee_link.STL", "right_shoulder_pitch_link.STL",
    "right_shoulder_roll_link.STL", "right_shoulder_yaw_link.STL", "right_wrist_pitch_link.STL",
    "right_wrist_roll_link.STL", "right_wrist_yaw_link.STL", "torso_link_rev_1_0.STL",
    "waist_roll_link_rev_1_0.STL", "waist_yaw_link_rev_1_0.STL",
]

PANDA_FILES = [
    "panda.xml", "LICENSE",
    *[f"assets/{n}" for n in (
        'finger_0.obj', 'finger_1.obj', 'hand.stl', 'hand_0.obj', 'hand_1.obj', 'hand_2.obj',
        'hand_3.obj', 'hand_4.obj', 'link0.stl', 'link0_0.obj', 'link0_1.obj', 'link0_10.obj',
        'link0_11.obj', 'link0_2.obj', 'link0_3.obj', 'link0_4.obj', 'link0_5.obj',
        'link0_7.obj', 'link0_8.obj', 'link0_9.obj', 'link1.obj', 'link1.stl', 'link2.obj',
        'link2.stl', 'link3.stl', 'link3_0.obj', 'link3_1.obj', 'link3_2.obj', 'link3_3.obj',
        'link4.stl', 'link4_0.obj', 'link4_1.obj', 'link4_2.obj', 'link4_3.obj', 'link5_0.obj',
        'link5_1.obj', 'link5_2.obj', 'link5_collision_0.obj', 'link5_collision_1.obj',
        'link5_collision_2.obj', 'link6.stl', 'link6_0.obj', 'link6_1.obj', 'link6_10.obj',
        'link6_11.obj', 'link6_12.obj', 'link6_13.obj', 'link6_14.obj', 'link6_15.obj',
        'link6_16.obj', 'link6_2.obj', 'link6_3.obj', 'link6_4.obj', 'link6_5.obj',
        'link6_6.obj', 'link6_7.obj', 'link6_8.obj', 'link6_9.obj', 'link7.stl', 'link7_0.obj',
        'link7_1.obj', 'link7_2.obj', 'link7_3.obj', 'link7_4.obj', 'link7_5.obj',
        'link7_6.obj', 'link7_7.obj',
    )],
]

# sha256 of LFS objects (from the upstream LFS pointers) -- policies must match exactly.
PINNED_SHA256 = {
    "g1/policy/GR00T-WholeBodyControl-Balance.onnx":
        "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18",
    "g1/policy/GR00T-WholeBodyControl-Walk.onnx":
        "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407",
}


def _jobs() -> list[tuple[str, str]]:
    """(url, relative destination) pairs."""
    raw = f"https://raw.githubusercontent.com/{GR00T_REPO}/{GR00T_SHA}/{GR00T_G1}"
    lfs = f"https://media.githubusercontent.com/media/{GR00T_REPO}/{GR00T_SHA}/{GR00T_G1}"
    jobs = [
        (f"{raw}/g1_gear_wbc.xml", "g1/g1_gear_wbc.xml"),
        (f"{raw}/g1_gear_wbc.yaml", "g1/g1_gear_wbc.yaml"),
        (f"{raw}/policy/NVIDIA%20Open%20Model%20License", "g1/policy/NVIDIA_Open_Model_License.txt"),
        (f"{lfs}/policy/GR00T-WholeBodyControl-Balance.onnx", "g1/policy/GR00T-WholeBodyControl-Balance.onnx"),
        (f"{lfs}/policy/GR00T-WholeBodyControl-Walk.onnx", "g1/policy/GR00T-WholeBodyControl-Walk.onnx"),
        (f"https://raw.githubusercontent.com/{GR00T_REPO}/{GR00T_SHA}/LICENSE", "g1/LICENSE_GR00T_WBC_Apache2.txt"),
    ]
    jobs += [(f"{lfs}/meshes/{m}", f"g1/meshes/{m}") for m in G1_MESHES]
    mraw = f"https://raw.githubusercontent.com/{MENAGERIE_REPO}/{MENAGERIE_SHA}/franka_emika_panda"
    jobs += [(f"{mraw}/{f}", f"panda/{f}") for f in PANDA_FILES]
    return jobs


def _download(url: str, dest: Path, force: bool) -> str:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "roboticsbench-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    head = tmp.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        tmp.unlink()
        raise RuntimeError(f"got an LFS pointer instead of content for {url}")
    tmp.replace(dest)
    return "downloaded"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assets_present() -> bool:
    return all((VENDOR / rel).exists() for _, rel in _jobs())


def fetch(force: bool = False, quiet: bool = False) -> None:
    jobs = _jobs()
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_download, url, VENDOR / rel, force): rel for url, rel in jobs}
        for fut in cf.as_completed(futs):
            fut.result()
    for rel, want in PINNED_SHA256.items():
        got = _sha256(VENDOR / rel)
        if got != want:
            raise RuntimeError(f"sha256 mismatch for {rel}: {got} != {want}")
    if not quiet:
        print(f"assets ready in {VENDOR} ({len(jobs)} files)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--dest", default=None, help="target directory (default: $RBENCH_ASSETS_DIR or package vendor/)")
    args = ap.parse_args(argv)
    if args.dest:
        global VENDOR
        VENDOR = Path(args.dest)
    fetch(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
