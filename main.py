import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal

GPU = "A100"
WORKFLOW_PATH = "test_sam.json"


image = (
    modal.Image.from_registry("ghcr.io/astral-sh/uv:python3.11-bookworm-slim")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.3.5")
    .run_commands(
        "comfy --skip-prompt install --nvidia --version 0.3.10"
    )
)

image = (
    image.run_commands(
        [
            "comfy node install comfyui-impact-pack",
            "comfy node install was-node-suite-comfyui",
            "comfy node install comfyui_segment_anything",
            "comfy node install comfyui_layerstyle",
            "comfy node install comfyui-in-context-lora-utils",
            "comfy node install efficiency-nodes-comfyui"
        ]
    )
)



def hf_download():
    from huggingface_hub import hf_hub_download

    shaper_filename = "DreamShaper_2.52.safetensors"
    shaper_model = hf_hub_download(
        repo_id="Lykon/DreamShaper",
        filename=shaper_filename,
        cache_dir="/cache",
    )
    
    sam_filename = "sam_vit_h_4b8939.pth"
    SAM_model = hf_hub_download(
        repo_id="segments-arnaud/sam_vit_h",
        filename=sam_filename,
        cache_dir="/cache",
    )
    
    subprocess.run(
        f"ln -s {shaper_model} /root/comfy/ComfyUI/models/checkpoints/{shaper_filename}",
        shell=True,
        check=True,
    )

    # Create the sams directory if it doesn't exist
    subprocess.run(
        "mkdir -p /root/comfy/ComfyUI/models/sams",
        shell=True,
        check=True,
    )
    
    subprocess.run(
        f"ln -s {SAM_model} /root/comfy/ComfyUI/models/sams/{sam_filename}",
        shell=True,
        check=True,
    )


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # Install huggingface hub with hf transfer support
    image.pip_install("huggingface-hub[hf-transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

image = image.add_local_file(
    Path(__file__).parent / f"{WORKFLOW_PATH}", f"/root/{WORKFLOW_PATH}"
)

app = modal.App(name="comfyui-api", image=image)


@app.function(
    allow_concurrent_inputs=10,
    max_containers=1,
    gpu=GPU,
    volumes={"/cache": vol},  # mount cached models
)
@modal.web_server(8000, startup_timeout=60)
# Service for ComfyUI
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)


# Run Workflow as API
@app.cls(
    allow_concurrent_inputs=10, scaledown_window=300, gpu=GPU, volumes={"/cache": vol}
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        # Starts comfyui server in background exactly once when first input received
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = WORKFLOW_PATH):
        # Runs comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200"
        subprocess.run(cmd, shell=True, check=True)

        # Completed workflows write output images to this dir
        output_dir = "/root/comfy/ComfyUI/output"

        # Lookup name of output image file based on workflow
        workflow = json.loads(Path(workflow_path).read_text())
        file_prefix = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
        ][0]["filename_prefix"]

        # Returns image as bytes
        for f in Path(output_dir).iterdir():
            if f.name.startswith(file_prefix):
                return f.read_bytes()

    @modal.web_endpoint(method="POST")
    def api(self, item: Dict):
        from fastapi import Response

        workflow_data = json.loads((Path(__file__).parent / WORKFLOW_PATH).read_text())

        # Insert prompt
        workflow_data["6"]["inputs"]["text"] = item["prompt"]

        # Give output image a uniue id per client request
        client_id = uuid.uuid4().hex
        workflow_data["9"]["inputs"]["filename_prefix"] = client_id

        # Save updated workflow to a new file
        new_workflow_file = f"{client_id}.json"
        json.dump(workflow_data, open(new_workflow_file, "w"))

        # Run inference on currently running container
        img_bytes = self.infer.local(new_workflow_file)

        return Response(img_bytes, media_type="image/png")
