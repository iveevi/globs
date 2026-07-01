import argparse
import logging

from . import logsetup
from .viewer import Viewer


def main():
    parser = argparse.ArgumentParser(
        prog="glb-viewer", description="slangpy glTF/GLB scene viewer"
    )
    parser.add_argument("model", help="path to a .glb / .gltf file")
    parser.add_argument("--frames", type=int, default=0,
                        help="render N frames then exit (testing)")
    parser.add_argument("--screenshot", default="",
                        help="with --frames, write the last frame to this path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose logging (per-texture decode, etc.)")
    args = parser.parse_args()

    logsetup.setup(logging.DEBUG if args.verbose else logging.INFO)

    Viewer(args.model).run(max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
