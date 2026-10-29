"""Command Line Interface (CLI) for Image-to-3D Mesh Agent."""

import argparse
import os
import sys
from PIL import Image

from .config import AgentConfig
from .orchestrator import ImageTo3DAgent
from .preprocessing import load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image3d",
        description="Image-to-3D Mesh Agent: Intelligent 3D mesh generation with watchdog routing.",
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path to input image for single-view 3D reconstruction."
    )
    parser.add_argument("--front", type=str, help="Path to front viewpoint image.")
    parser.add_argument("--left", type=str, help="Path to left viewpoint image.")
    parser.add_argument("--back", type=str, help="Path to back viewpoint image.")
    parser.add_argument("--right", type=str, help="Path to right viewpoint image.")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./outputs", help="Output directory for generated assets."
    )
    parser.add_argument(
        "-n", "--name", type=str, default="mesh_output", help="Object identifier name for saved assets."
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=80, help="Watchdog timeout limit in seconds (default: 80s)."
    )
    parser.add_argument(
        "--gemini-key", type=str, default="", help="Google Gemini API key for Tier 2 cloud fallback."
    )
    parser.add_argument(
        "--max-faces", type=int, default=20000, help="Quadric decimation face target (default: 20000)."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    items = []
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        items.append(("front", load_image(args.input)))

    multi_keys = [("front", args.front), ("left", args.left), ("back", args.back), ("right", args.right)]
    for key, path in multi_keys:
        if path and os.path.exists(path):
            items.append((key, load_image(path)))

    if not items:
        parser.print_help()
        print("\nError: Please provide at least --input or --front image.", file=sys.stderr)
        sys.exit(1)

    config = AgentConfig(
        timeout_seconds=args.timeout,
        output_dir=args.output_dir,
        max_faces=args.max_faces,
        gemini_api_key=args.gemini_key,
    )

    agent = ImageTo3DAgent(config=config)
    print(f"Starting 3D reconstruction for '{args.name}'...")
    result = agent.run(items=items, name=args.name, api_key=args.gemini_key)

    report = result["report"]
    print("\n Reconstructed Successfully!")
    print(f"  • Backend Used : {report['backend']}")
    print(f"  • Latency      : {report['seconds']}s (Watchdog limit: {report['watchdog_timeout']}s)")
    print(f"  • Geometry     : {report['faces']:,} faces, {report['vertices']:,} vertices")
    print(f"  • GLB Asset    : {result['glb']}")
    print(f"  • OBJ Asset    : {result['obj']}")
    print(f"  • STL Asset    : {result['stl']}")
    print(f"  • Preview      : {result['preview']}")
    print(f"  • JSON Report  : {result['report_json']}")


if __name__ == "__main__":
    main()
