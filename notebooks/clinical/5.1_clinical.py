import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import exposure, img_as_float
import aipassport_config as cfg

st.markdown(
    """
An image is a matrix of numbers, not a picture, to a model. This subsection establishes where those
numbers come from and what happens when you change them.

1. **Where the numbers come from.** Every pixel in a radiograph records how much of the X-ray beam
   survived the tissue at that point. You will use an edge detector to find the places where that
   measurement changes sharply, and see how the same anatomy looks different on a CT-weighted view
   and an MRI-weighted one.
2. **Changing what you can see.** Four common adjustments — gamma, contrast rescaling, histogram
   equalization, and CLAHE — each make features easier to spot by moving the pixel values around.
   None of them add information that was not measured.

That distinction is the whole point of this subsection. Subsection 5.2 then handles real artifacts.
"""
)

st.warning(
    "**Privacy notice:** do not upload images containing Protected Health Information (PHI) or any "
    "sensitive personal data."
)

DEFAULT_IMAGE_PATH = "assets/images/content/Identifying Structures in X-Ray Imaging.png"


@st.cache_data
def load_default_radiograph():
    return cv2.imread(DEFAULT_IMAGE_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Image formation and structure
# ═══════════════════════════════════════════════════════════════════════════
st.header("1. Where Pixel Values Come From")

st.markdown(
    """
X-rays pass through the body and are absorbed — *attenuated* — to different degrees depending on tissue
density and atomic number. Bone absorbs most of the beam and reads bright; air passes it and reads dark;
soft tissue lands in between. The greyscale is a measurement.

Which means a **boundary** in the image is a boundary in attenuation — a place where the tissue changed.
That is what an edge detector finds.
"""
)

uploaded_file = st.file_uploader(
    "Upload a radiograph (optional)",
    type=["jpg", "jpeg", "png"],
    key="m5_form_upload",
    help="A standard JPG or PNG. Ensure no patient data is visible in the image.",
)

if uploaded_file:
    img_bgr = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    img_bgr = load_default_radiograph()

if img_bgr is None:
    st.error(
        f"Could not load the bundled radiograph at `{DEFAULT_IMAGE_PATH}`. Upload an image above to continue."
    )
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    st.subheader("1.1 Finding structure: Canny edge detection")
    st.markdown(
        "Canny keeps a gradient as an edge if it exceeds the **high** threshold, and keeps weaker gradients "
        "only where they connect to a strong one. The two thresholds are the sensitivity/noise trade-off "
        "made explicit."
    )

    edge_cols = st.columns(2)
    low_threshold = edge_cols[0].slider(
        "Low threshold (sensitivity)",
        0,
        200,
        100,
        help="Gradients below this are discarded outright. Lowering it admits more noise.",
        key="m5_form_canny_low",
    )
    high_threshold = edge_cols[1].slider(
        "High threshold (edge strength)",
        0,
        255,
        150,
        help="Gradients above this are always kept as strong edges.",
        key="m5_form_canny_high",
    )

    edges = cv2.Canny(gray, low_threshold, high_threshold)

    view_cols = st.columns([1, 1, 1])
    view_cols[0].image(img_rgb, caption="Original", width="stretch")
    view_cols[1].image(edges, caption="Canny edges", width="stretch")
    with view_cols[2]:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        fig_hist, ax_hist = plt.subplots(figsize=(4.5, 3.2))
        ax_hist.plot(hist, color=cfg.CHART_PRIMARY)
        ax_hist.set_title("Intensity histogram")
        ax_hist.set_xlabel("Pixel intensity (0–255)")
        ax_hist.set_ylabel("Pixel count")
        ax_hist.set_xlim([0, 256])
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        with st.expander("View chart data as text (accessible alternative)"):
            st.markdown(
                "Pixel counts grouped into sixteen intensity bands, darkest first. A peak tells you "
                "where most of the image sits on the black-to-white scale."
            )
            counts = hist.ravel().reshape(16, 16).sum(axis=1)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Intensity band": [f"{i * 16}–{i * 16 + 15}" for i in range(16)],
                        "Pixel count": counts.astype(int),
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    edge_fraction = float((edges > 0).mean())
    st.metric("Share of pixels marked as edge", f"{edge_fraction:.1%}")
    st.markdown(
        """
    Reading a histogram is how you diagnose an image before you diagnose a patient: intensities bunched at
    one end mean an under- or over-exposed acquisition, and no amount of modelling recovers detail that was
    never captured.

    **Consider:** why is edge detection alone insufficient for finding a fracture? Drop the low threshold and
    watch the edge fraction climb — the detector will happily outline trabecular texture, film grain, and
    soft-tissue boundaries with exactly the same confidence it gives a fracture line. It reports *where the
    gradient is*, and nothing about what the gradient means.
    """
    )

    st.subheader("1.2 The same tissue, two modalities' worth of appearance")
    st.markdown(
        "Contrast and brightness are the two knobs behind every window/level control. A high-contrast view "
        "separates dense structure the way a **CT** does; lifting brightness and moderating contrast reveals "
        "soft-tissue gradation the way an **MRI** does. Same acquired data, two different clinical questions."
    )

    mod_cols = st.columns(2)
    contrast = mod_cols[0].slider(
        "Contrast",
        1.0,
        3.0,
        1.0,
        help="Multiplies the distance between light and dark, separating tissue types. "
        "Starts at 1.0 — no change — so you can see what each step costs.",
        key="m5_form_contrast",
    )
    brightness = mod_cols[1].slider(
        "Brightness",
        -50,
        50,
        0,
        help="Shifts every value up or down, revealing detail in shadowed regions.",
        key="m5_form_brightness",
    )

    adjusted = cv2.convertScaleAbs(img_rgb, alpha=contrast, beta=brightness)

    cmp_cols = st.columns(2)
    cmp_cols[0].image(img_rgb, caption="Baseline (dense-structure focus)", width="stretch")
    cmp_cols[1].image(
        adjusted,
        caption=f"Adjusted (contrast {contrast}, brightness {brightness:+d})",
        width="stretch",
    )

    # Report the saturation this adjustment *added*. This radiograph's background is already
    # pure white, and that pre-existing 255 is not something the learner did.
    baseline_clipped = float((gray >= 255).mean())
    now_clipped = float((cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY) >= 255).mean())
    added = max(0.0, now_clipped - baseline_clipped)

    if added > 0.10:
        st.error(
            f"**Your adjustment has pushed a further {added:.1%} of the image to pure white** "
            f"({baseline_clipped:.1%} was already background). Those values are gone with no way back. This "
            "is the failure mode that makes aggressive enhancement dangerous: the image looks more confident "
            "and contains less."
        )
    elif added > 0.02:
        st.warning(
            f"A further {added:.1%} of the image is now saturated at 255 "
            f"(background was {baseline_clipped:.1%}). You are beginning to trade real detail for apparent "
            "contrast."
        )
    else:
        st.caption(
            f"Newly saturated: {added:.2%}. Nothing meaningful has been clipped yet — "
            f"{baseline_clipped:.1%} of this image was already pure-white background."
        )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Intensity operations
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. Changing What Is Visible")

st.markdown(
    """
The operations below are **display and preprocessing transforms**. They redistribute intensities to make
structure easier to see — for you or for a model. None of them adds information that was not acquired.
"""
)


@st.cache_data
def load_intensity_samples_clinical():
    """Bundled imaging samples, alpha stripped so every one is RGB or greyscale."""

    def read(path):
        image = io.imread(path)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]
        return image

    return {
        "Kidney MRI (low dynamic range)": read("assets/datasets/images/kidney_mri.jpg"),
        "Mammography (low contrast)": read("assets/datasets/images/breast.png"),
        "Low-contrast sample": read("assets/datasets/images/low_contrast2.jpg"),
        "Brightfield (blood smear)": read("assets/datasets/images/BloodSmear.png"),
        "Fluorescence (IF cells)": read("assets/datasets/images/IFCells.jpg"),
    }


images = load_intensity_samples_clinical()

def to_gray_for_stats(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


image_choice = st.selectbox(
    "Image:",
    list(images),
    help="The same operation behaves differently on each of these. The low-contrast and "
    "low-dynamic-range samples are where rescaling and CLAHE earn their keep; on a "
    "well-exposed brightfield image they mostly amplify noise.",
    key="m5_intensity_image",
)
img = images[image_choice]

st.caption(
    f"5th-95th percentile spread of **{image_choice}**: "
    f"{int(np.percentile(to_gray_for_stats(img), 95) - np.percentile(to_gray_for_stats(img), 5))} "
    "of 255 available levels. The narrower that is, the more an intensity operation has to work with."
)


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


OPERATIONS = [
    "Gamma",
    "Contrast rescaling",
    "Histogram equalization",
    "CLAHE",
]
# A keyed segmented_control rather than st.tabs: tab selection lives in the browser and is
# lost whenever a widget inside a tab triggers a rerun, which is what sent learners back to
# the first activity mid-edit. This selection is in session_state, so it survives.
operation = st.segmented_control(
    "Intensity operation",
    OPERATIONS,
    default=OPERATIONS[0],
    key="m5_intensity_op",
    required=True,
)
if operation == OPERATIONS[0]:
    st.subheader("Gamma correction")
    st.markdown(
        "A power-law curve applied to every pixel. Values **below 1** brighten the image by expanding the "
        "dark end; values **above 1** darken it by compressing the bright end. Pure black and pure white "
        "stay where they are."
    )
    gamma = st.slider(
        "Gamma",
        0.1,
        3.0,
        1.0,
        step=0.05,
        help="The exponent in the power-law mapping between stored value and displayed luminance.",
        key="m5_gamma",
    )
    corrected = exposure.adjust_gamma(img, gamma=gamma, gain=1)

    g_cols = st.columns(2)
    g_cols[0].image(img, caption="Original", width="stretch")
    g_cols[1].image(corrected, caption=f"Gamma = {gamma}", width="stretch")

    with st.expander("Reveal expected outcome"):
        st.write(
            "Gamma is most useful when the interesting structure sits in the mid-tones — which, in MRI, is "
            "usually where soft-tissue contrast lives. Push it very low and the image washes out; push it "
            "high and it goes muddy. The extremes never move: gamma redistributes the middle, it does not "
            "extend the range."
        )

if operation == OPERATIONS[1]:
    st.subheader("Contrast rescaling")
    st.markdown(
        "Pick a narrow intensity window and stretch it across the full range. Everything below the minimum "
        "is crushed to black, everything above the maximum to white — and the detail in between expands to "
        "fill the space. This is precisely the window/level control on a clinical workstation."
    )
    r_cols = st.columns(2)
    in_min = r_cols[0].slider("Window minimum", 0.0, 1.0, 0.55, key="m5_rescale_min")
    in_max = r_cols[1].slider("Window maximum", 0.0, 1.0, 0.7, key="m5_rescale_max")

    if in_min >= in_max:
        st.warning("The window minimum must be below the maximum — the range would otherwise be empty.")
    else:
        adjusted_img = exposure.rescale_intensity(
            img_as_float(img), in_range=(in_min, in_max), out_range=(0, 1)
        )
        a_cols = st.columns(2)
        a_cols[0].image(img, caption="Original", width="stretch")
        a_cols[1].image(
            adjusted_img, caption=f"Rescaled from [{in_min}, {in_max}]", width="stretch"
        )

    with st.expander("Reveal expected outcome"):
        st.write(
            "Hidden detail 'pops' because you gave a narrow band the whole dynamic range — but everything "
            "outside the window is now irrecoverably flat. You chose what to see, and therefore also what to "
            "discard. Two readers using different windows are looking at different images of the same patient."
        )

if operation == OPERATIONS[2]:
    st.subheader("Histogram equalization")
    st.markdown(
        "Rather than you choosing a window, equalization spreads the *most frequent* intensities out "
        "automatically, flattening the histogram."
    )
    num_bins = st.slider(
        "Histogram bins", 10, 256, 256, help="More bins show finer structure and look more jagged.",
        key="m5_eq_bins",
    )

    img_gray = to_gray(img)
    img_eq = cv2.equalizeHist(img_gray)

    fig_eq, ax_eq = plt.subplots(2, 2, figsize=(10, 7))
    ax_eq[0, 0].imshow(img_gray, cmap="gray")
    ax_eq[0, 0].axis("off")
    ax_eq[0, 0].set_title("Original")
    ax_eq[0, 1].hist(img_gray.ravel(), bins=num_bins, color=cfg.CHART_PRIMARY)
    ax_eq[0, 1].set_title("Original histogram")
    ax_eq[1, 0].imshow(img_eq, cmap="gray")
    ax_eq[1, 0].axis("off")
    ax_eq[1, 0].set_title("Equalized")
    ax_eq[1, 1].hist(img_eq.ravel(), bins=num_bins, color=cfg.CHART_SECONDARY)
    ax_eq[1, 1].set_title("Equalized histogram")
    fig_eq.tight_layout()
    st.pyplot(fig_eq)
    plt.close(fig_eq)

    with st.expander("View chart data as text (accessible alternative)"):
        st.markdown(
            "Equalization spreads the intensities out. Compare the two columns: the original clusters "
            "into a few bands, the equalized version is closer to an even count in every band."
        )
        bands = np.arange(0, 256, 16)
        st.dataframe(
            pd.DataFrame(
                {
                    "Intensity band": [f"{b}–{b + 15}" for b in bands],
                    "Original pixel count": np.histogram(img_gray, bins=np.append(bands, 256))[0],
                    "Equalized pixel count": np.histogram(img_eq, bins=np.append(bands, 256))[0],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Reveal expected outcome"):
        st.write(
            "The equalized histogram is flatter and wider, and the image looks starker. The cost is that "
            "equalization is **global**: it amplifies contrast everywhere, including noise in regions that "
            "were uniform for a good reason. That is the problem CLAHE exists to solve."
        )

if operation == OPERATIONS[3]:
    st.subheader("CLAHE — contrast-limited adaptive histogram equalization")
    st.markdown(
        "Equalization applied tile by tile instead of to the whole image, with a ceiling on how much any one "
        "tile's contrast may be amplified. Local detail improves without the noise blow-up — which is why "
        "CLAHE is the default enhancement in a great deal of medical imaging software."
    )
    c_cols = st.columns(2)
    clip_limit = c_cols[0].slider(
        "Clip limit", 1.0, 10.0, 2.0, help="The ceiling on per-tile amplification.", key="m5_clahe_clip"
    )
    tile_grid_size = c_cols[1].slider(
        "Tile grid size", 2, 32, 8, help="Smaller tiles mean more local adaptation.", key="m5_clahe_tile"
    )

    img_gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    clahe_img = clahe.apply(img_gray)

    cl_cols = st.columns(2)
    cl_cols[0].image(img_gray, caption="Original (greyscale)", width="stretch")
    cl_cols[1].image(
        clahe_img, caption=f"CLAHE (clip {clip_limit}, {tile_grid_size}×{tile_grid_size} tiles)",
        width="stretch",
    )

    with st.expander("Reveal expected outcome"):
        st.write(
            "Fine texture becomes visible across the whole field, not just in the well-exposed part. Raise "
            "the clip limit far enough and you will see the noise come back — which is the setting telling "
            "you exactly where enhancement stops being information and starts being invention. An enhanced "
            "image that looks more diagnostic than the acquisition supports is a patient-safety problem."
        )

st.markdown(
    """
---
**Key takeaways**

- A pixel value is a measurement of attenuation. The histogram is the fastest way to judge whether an
  acquisition is usable at all.
- An edge detector reports where the gradient is, not what it means. Structure-finding is not diagnosis.
- Every operation here is lossy in practice: windowing discards what falls outside the window, equalization
  amplifies noise, CLAHE bounds that amplification.
- **None of these adds information.** If the structure was not captured, no transform recovers it — and any
  transform that appears to has invented it.
- Whatever you apply here must be applied identically to every image a model ever sees, at training and at
  inference. An enhancement pipeline is part of the model.

**Resources:** [scikit-image exposure](https://scikit-image.org/docs/stable/api/skimage.exposure.html) ·
[OpenCV Canny tutorial](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
"""
)
