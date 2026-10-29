"""Programmatic Single-Image 3D Reconstruction Example."""

import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import AgentConfig, ImageTo3DAgent, load_image


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not image_path or not os.path.exists(image_path):
        print("No input image path provided or file does not exist.")
        print("Usage: python examples/run_single_image.py <path_to_image.png>")
        print("\nCreating a temporary demonstration shape instead...")
        from examples.generate_procedural_demo import create_synthetic_test_image
        input_image = create_synthetic_test_image(size=512)
        object_name = "demo_shape"
    else:
        print(f"Loading input image: {image_path}")
        input_image = load_image(image_path)
        object_name = os.path.splitext(os.path.basename(image_path))[0]

    # Configure the agent
    config = AgentConfig(
        timeout_seconds=80,
        max_faces=20000,
        output_dir="./outputs",
    )

    agent = ImageTo3DAgent(config=config)
    print(f"Initiating 3D reconstruction for '{object_name}'...")
    results = agent.run(items=[("front", input_image)], name=object_name)

    rep = results["report"]
    print("\n🎉 Reconstruction Complete!")
    print(f"  • Backend Engine: {rep['backend']}")
    print(f"  • Latency       : {rep['seconds']}s")
    print(f"  • Face Count    : {rep['faces']:,}")
    print(f"  • GLB File      : {results['glb']}")
    print(f"  • Preview Image : {results['preview']}")
    print(f"  • Report JSON   : {results['report_json']}")


if __name__ == "__main__":
    main()
