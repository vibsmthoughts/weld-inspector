import base64
import binascii
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


WORKSPACE = "vibsm-thoughts"
WORKFLOW = "weld-surface-inspection-1792419902153"
MAX_BYTES = 10 * 1024 * 1024

TARGET_CLASSES = {
    "cracks",
    "porosity",
    "spatter",
    "undercut",
    "bead irregularity",
}

GUIDANCE = {
    "cracks": {
        "action": (
            "Hold the part for qualified assessment before service or further "
            "welding, especially for structural or pressure-containing parts."
        ),
        "repair": [
            "Have a qualified inspector confirm the indication and determine "
            "its extent using the examination method required for the part.",
            "If repair is permitted, a qualified repair procedure must specify "
            "defect removal and verification before rewelding.",
            "Reweld only under an approved procedure, then perform the "
            "required post-repair inspection.",
        ],
        "prevention": [
            "Review material compatibility, filler selection, restraint, "
            "joint preparation, and hydrogen control.",
            "Verify any required preheat and interpass-temperature controls "
            "against the approved welding procedure.",
        ],
    },
    "porosity": {
        "action": (
            "Confirm that visible cavities are weld porosity rather than "
            "surface contamination, corrosion, or another indication."
        ),
        "repair": [
            "Have a qualified reviewer assess size, distribution, and "
            "acceptability against the applicable requirements.",
            "If repair is required and allowed, remove the affected weld "
            "material as specified by the repair procedure.",
            "Correct the contamination or shielding issue before rewelding "
            "and reinspecting.",
        ],
        "prevention": [
            "Clean and dry the joint and consumables as required.",
            "Check shielding-gas delivery, leaks, drafts, nozzle condition, "
            "and consumable storage.",
        ],
    },
    "spatter": {
        "action": (
            "Determine whether the droplets are attached spatter and whether "
            "they affect fit, coating, sealing, or acceptance requirements."
        ),
        "repair": [
            "Allow safe cooling and use an approved cleaning method.",
            "Remove spatter only as permitted without gouging the base metal "
            "or damaging the weld profile.",
            "Inspect the cleaned surface for damage or previously hidden "
            "indications.",
        ],
        "prevention": [
            "Review arc stability, consumable condition, shielding, and "
            "equipment setup.",
            "Check technique and electrical settings against the qualified "
            "procedure rather than changing them from the photo alone.",
        ],
    },
    "undercut": {
        "action": (
            "Confirm the groove at the weld toe and measure it with an "
            "appropriate inspection tool; a photo cannot establish its depth."
        ),
        "repair": [
            "Have a qualified reviewer compare the measured condition with "
            "the applicable acceptance criteria.",
            "Where permitted, an approved procedure may specify blending "
            "or weld restoration.",
            "Verify remaining material thickness, final profile, and "
            "post-repair inspection requirements.",
        ],
        "prevention": [
            "Review travel speed, arc length, electrode angle, and heat input "
            "against the approved procedure.",
            "Check joint fit-up and technique at the weld toes.",
        ],
    },
    "bead irregularity": {
        "action": (
            "Review the bead profile against the joint drawing and welding "
            "procedure. An uneven appearance alone does not establish "
            "structural failure."
        ),
        "repair": [
            "Measure the required weld size and profile using suitable tools.",
            "Check for associated overlap, underfill, or other indications "
            "that need separate assessment.",
            "If repair is required, use an approved procedure for dressing, "
            "removal, or additional welding, followed by inspection.",
        ],
        "prevention": [
            "Review fit-up, access, travel consistency, torch or electrode "
            "angle, and deposition technique.",
            "Confirm equipment operation and procedure compliance.",
        ],
    },
}


class ResponseError(Exception):
    pass


def get_status_code(exc):
    """Inspect exception chains without displaying credentials or raw URLs."""
    current = exc
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status

        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status

        current = current.__cause__ or current.__context__

    return None


def show_request_error(exc):
    status = get_status_code(exc)

    messages = {
        400: "Roboflow rejected the request format or configuration.",
        401: (
            "Authentication failed. Check the private ROBOFLOW_API_KEY "
            "secret and restart the app."
        ),
        402: (
            "Roboflow reported a payment or inference-access restriction. "
            "Check the workspace's plan and usage."
        ),
        403: (
            "Access was denied. Confirm that the key belongs to "
            "'vibsm-thoughts' and can access the Workflow and its Model."
        ),
        404: (
            "The Workflow or a linked resource was not found or is not "
            "accessible to this API key."
        ),
        423: "Roboflow reported that workspace billing is locked or paused.",
        429: "Roboflow rate-limited the request. Wait before trying again.",
    }

    if status is not None:
        message = messages.get(
            status,
            "The inspection service failed to complete the request."
        )
        st.error(f"HTTP {status}: {message}")
    else:
        st.error(
            "Inspection could not be completed. A network, SDK, or service "
            "error occurred. Ask your developer to inspect sanitized "
            "server diagnostics."
        )

    st.caption(
        f"Diagnostic type: {type(exc).__name__}. "
        "No automatic retry was made by this app."
    )


def prepare_image(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")

    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Choose an image with no more than 25 megapixels.")

        image = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def parse_result(raw):
    if not isinstance(raw, list) or len(raw) != 1:
        raise ResponseError("Expected exactly one inspection result.")

    entry = raw[0]
    if not isinstance(entry, dict):
        raise ResponseError("The inspection result is not an object.")

    required = {"output_image", "predictions", "defect_region_count"}
    if not required.issubset(entry):
        raise ResponseError("The Workflow response is missing required outputs.")

    prediction_data = entry["predictions"]
    if not isinstance(prediction_data, dict):
        raise ResponseError("Invalid predictions output.")

    defects = prediction_data.get("predictions")
    if not isinstance(defects, list):
        raise ResponseError("The defect list is missing or invalid.")

    count = entry["defect_region_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or count != len(defects)
    ):
        raise ResponseError("The defect count does not match the results.")

    for defect in defects:
        if not isinstance(defect, dict):
            raise ResponseError("Invalid defect entry.")
        if defect.get("class") not in TARGET_CLASSES:
            raise ResponseError("An unexpected defect category was returned.")

    encoded = entry["output_image"]
    if isinstance(encoded, dict):
        encoded = encoded.get("value")

    if not isinstance(encoded, str) or not encoded:
        raise ResponseError("The annotated image is missing.")

    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]

    try:
        annotated = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(annotated)) as image:
            image.verify()
    except (
        binascii.Error,
        ValueError,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ResponseError("The annotated image could not be decoded.") from exc

    return entry, annotated


def inspect_image(image_bytes, key):
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "weld.jpg"
        image_path.write_bytes(image_bytes)

        raw = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW,
            images={"image": str(image_path)},
            use_cache=True,
        )

    return parse_result(raw)


st.set_page_config(
    page_title="Weld Inspector",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Weld Inspector")
st.write("Surface-defect screening and general corrective guidance.")

st.warning(
    "For human-reviewed visual screening only. This app does not certify "
    "weld acceptance, measure defect depth, or detect internal defects. "
    "Do not perform repairs solely from these results."
)

with st.expander("Inspection scope"):
    st.write(
        "Target categories: cracks, porosity, spatter, undercut, "
        "and bead irregularity."
    )
    st.write(
        "Corrosion, slag, shadows, and surface texture can cause incorrect "
        "findings. No visible defect detected does not mean defect-free."
    )
    st.write(
        "Images are sent to Roboflow for processing. Upload only images "
        "you are authorized to share."
    )

key = str(st.secrets.get("ROBOFLOW_API_KEY", "")).strip()
configured = bool(key) and key != "YOUR_ROBOFLOW_API_KEY"

if not configured:
    st.error(
        "Setup required: add your private Roboflow API key as "
        "ROBOFLOW_API_KEY in Replit Secrets, then restart this app."
    )

source = st.radio(
    "Image source",
    ["Upload image", "Use camera"],
    horizontal=True,
)

if source == "Use camera":
    uploaded = st.camera_input("Take a clear weld photo")
else:
    uploaded = st.file_uploader(
        "Upload a weld photo — JPG or PNG, up to 10 MB",
        type=["jpg", "jpeg", "png"],
    )

if uploaded is None:
    st.info("Upload or capture a weld photo to begin.")
    st.stop()

try:
    image_bytes = prepare_image(uploaded.getvalue())
except (ValueError, UnidentifiedImageError, OSError) as exc:
    st.error(f"Cannot use this image: {exc}")
    st.stop()

image_id = hashlib.sha256(image_bytes).hexdigest()

# Clear results when the selected image changes.
if st.session_state.get("image_id") != image_id:
    st.session_state["image_id"] = image_id
    st.session_state.pop("inspection", None)

st.image(image_bytes, caption="Selected image", use_container_width=True)
st.caption("Inspect makes an online inference request and may incur usage.")

if st.button(
    "Inspect weld",
    type="primary",
    disabled=not configured,
):
    st.session_state.pop("inspection", None)

    try:
        with st.spinner("Inspecting the weld surface…"):
            entry, annotated = inspect_image(image_bytes, key)

        st.session_state["inspection"] = {
            "entry": entry,
            "annotated": annotated,
            "time": datetime.now(timezone.utc).isoformat(),
        }
    except ResponseError as exc:
        st.error(f"Unexpected inspection response: {exc}")
    except Exception as exc:
        show_request_error(exc)

inspection = st.session_state.get("inspection")
if inspection is None:
    st.stop()

entry = inspection["entry"]
defects = entry["predictions"]["predictions"]
counts = Counter(defect["class"] for defect in defects)

st.subheader("Visual findings")
st.metric("Detected regions", entry["defect_region_count"])
st.caption(
    "Region counts are Model predictions—not measured defect severity "
    "or acceptance decisions."
)

st.image(
    inspection["annotated"],
    caption="Review each marked region for correctness",
    use_container_width=True,
)

if counts:
    st.table([
        {"Category": name, "Detected regions": count}
        for name, count in sorted(counts.items())
    ])

    st.subheader("Corrective guidance")
    st.info(
        "This is a fixed general reference selected by the predicted labels. "
        "It is not a photo-derived engineering repair procedure."
    )

    for name in sorted(counts):
        guidance = GUIDANCE[name]
        with st.expander(name.title(), expanded=True):
            st.markdown(f"**First action:** {guidance['action']}")

            st.markdown("**Repair considerations**")
            for item in guidance["repair"]:
                st.markdown(f"- {item}")

            st.markdown("**Prevention checks**")
            for item in guidance["prevention"]:
                st.markdown(f"- {item}")
else:
    st.info(
        "No target defect was detected. Image quality, hidden surfaces, "
        "and Model limitations may conceal defects. Continue the inspection "
        "required by your quality procedure."
    )

report = {
    "inspected_at_utc": inspection["time"],
    "workspace": WORKSPACE,
    "workflow": WORKFLOW,
    "image_sha256": image_id,
    "defect_region_count": entry["defect_region_count"],
    "counts_by_category": dict(counts),
    "predictions": entry["predictions"],
    "general_guidance": {
        name: GUIDANCE[name] for name in sorted(counts)
    },
    "human_review_status": "not_recorded",
    "notice": (
        "Unverified visual screening. Not a weld acceptance certificate "
        "or an approved repair procedure."
    ),
}

st.download_button(
    "Download annotated image",
    data=inspection["annotated"],
    file_name="weld-inspection.jpg",
    mime="image/jpeg",
)

st.download_button(
    "Download inspection report",
    data=json.dumps(report, indent=2),
    file_name="weld-inspection-report.json",
    mime="application/json",
)
