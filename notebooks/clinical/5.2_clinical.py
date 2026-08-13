import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform, img_as_ubyte
from skimage.util import random_noise
from skimage.morphology import closing, disk
from skimage.feature import graycomatrix, graycoprops

st.markdown(
    """
Subsection 5.1 established that a pixel is a measurement and that intensity operations change what is
visible, not what was captured. Now the work: real artifacts, real feature extraction, and the question that
matters more than any single filter — **is this pipeline consistent enough to trust?**

1. **Augmentation** — teach a model the shape of a structure rather than its position on the slide.
2. **Artifacts** — motion blur and sensor noise, and why the right denoiser depends on the artifact.
3. **Features** — directional and Sobel gradients, GLCM texture on malignant vs. benign tissue, and
   morphological closing. *GLCM* is the grey-level co-occurrence matrix: it counts how often each pair
   of brightness values sits next to each other, which is how you turn "this tissue looks coarse" into
   a number a model can use.
4. **Trust** — the consistency gate every imaging pipeline has to pass before anyone acts on its output.
"""
)

st.warning(
    "**Privacy notice:** this is an educational sandbox. Do not upload sensitive clinical data, "
    "personally identifiable information (PII), or Protected Health Information (PHI)."
)

BF_PATH = "assets/datasets/images/BloodSmear.png"
IF_PATH = "assets/datasets/images/IFCells.jpg"

image_source = st.selectbox(
    "Working image for activities 1–3:",
    ["Brightfield (blood smear)", "Fluorescence (IF cells)", "Upload an image"],
    help="Chosen once and used throughout, so you can compare operations on the same input.",
    key="m5_prep_source",
)

uploaded_file = None
if image_source == "Upload an image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="m5_prep_upload",
        help="JPG or PNG. No sensitive data.",
    )


@st.cache_data
def load_working_image(choice, upload):
    if upload is not None:
        return np.array(Image.open(upload).convert("RGB"))
    if choice == "Fluorescence (IF cells)":
        return np.array(Image.open(IF_PATH).convert("RGB"))
    if choice == "Brightfield (blood smear)":
        return np.array(Image.open(BF_PATH).convert("RGB"))
    return None


img = load_working_image(image_source, uploaded_file)

if img is None:
    st.info("Upload an image above to begin, or pick one of the bundled samples.")
else:
    ACTIVITIES = [
        "1. Normalization & Augmentation",
        "2. Artifacts & Denoising",
        "3. Feature Extraction",
        "4. Is the Pipeline Trustworthy?",
    ]
    # A keyed segmented_control rather than st.tabs: tab selection lives in the browser and is
    # lost whenever a widget inside a tab triggers a rerun, which is what sent learners back to
    # the first activity mid-edit. This selection is in session_state, so it survives.
    activity = st.segmented_control(
        "Activity",
        ACTIVITIES,
        default=ACTIVITIES[0],
        key="m5_prep_activity",
        required=True,
    )
    # ═══════════════════════════════════════════════════════════════════════
    # 1 — Normalization and augmentation
    # ═══════════════════════════════════════════════════════════════════════
    if activity == ACTIVITIES[0]:
        st.header("Activity 1: Normalization and Augmentation")
        st.markdown(
            "Normalization puts every image on the same numeric footing. Augmentation changes an image's "
            "orientation without changing what it contains — which forces a model to learn the *shape* of a "
            "structure instead of memorizing where on the slide it happened to sit."
        )

        aug_cols = st.columns(2)
        normalization_factor = aug_cols[0].slider(
            "Normalization factor", 0.0, 1.0, 1.0,
            help="Scales the [0,1] normalized image. Lowering it uniformly darkens.",
            key="m5_prep_norm",
        )
        rotation_angle = aug_cols[1].slider(
            "Rotation (degrees)", -30.0, 30.0, 0.0, key="m5_prep_rotate"
        )
        flip_cols = st.columns(2)
        flip_horizontal = flip_cols[0].checkbox("Flip horizontal", key="m5_prep_fliph")
        flip_vertical = flip_cols[1].checkbox("Flip vertical", key="m5_prep_flipv")

        processed = img.astype(np.float32) / 255.0
        processed = processed * normalization_factor
        processed = transform.rotate(processed, rotation_angle, mode="wrap")
        if flip_horizontal:
            processed = np.fliplr(processed)
        if flip_vertical:
            processed = np.flipud(processed)

        cols = st.columns(2)
        cols[0].image(img, channels="RGB", caption="Original", width="stretch")
        cols[1].image(
            processed, channels="RGB", caption="Normalized + augmented",
            width="stretch", clamp=True,
        )

        with st.expander("What to expect"):
            st.write(
                "**Normalization:** standardizing every image to a [0, 1] range keeps neural-network "
                "gradients stable — an image whose values run to 4095 and one that runs to 255 must not "
                "produce different weight updates for the same anatomy.\n\n"
                "**Augmentation:** rotating and flipping leaves the cells unchanged while changing every "
                "pixel coordinate. A model trained with augmentation cannot pass by learning 'the "
                "interesting thing is in the upper left'. Note the `mode='wrap'` edge behaviour above: "
                "rotation has to invent the corners somehow, and that invention is in your training data."
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 2 — Artifacts and denoising
    # ═══════════════════════════════════════════════════════════════════════
    if activity == ACTIVITIES[1]:
        st.header("Activity 2: Artifacts, and Matching the Fix to the Fault")

        st.subheader("2.1 Motion blur")
        st.markdown(
            "Camera shake or stage movement during acquisition smears the image along the direction of "
            "travel. Simulating it tells you what your pipeline will face."
        )
        blur_cols = st.columns(2)
        blur_length = blur_cols[0].slider(
            "Blur length", 3, 50, 20, help="How far the smear stretches — a longer or faster movement.",
            key="m5_prep_blur_len",
        )
        blur_angle = blur_cols[1].slider(
            "Blur angle (degrees)", 0, 180, 45, help="The direction of travel.", key="m5_prep_blur_angle"
        )

        kernel = np.zeros((blur_length, blur_length))
        kernel[int((blur_length - 1) / 2), :] = np.ones(blur_length)
        rot = cv2.getRotationMatrix2D((blur_length / 2 - 0.5, blur_length / 2 - 0.5), blur_angle, 1)
        kernel = cv2.warpAffine(kernel, rot, (blur_length, blur_length))
        kernel /= blur_length

        img_motion = cv2.filter2D(img, -1, kernel)
        mb_cols = st.columns(2)
        mb_cols[0].image(img, caption="Original", width="stretch")
        mb_cols[1].image(img_motion, caption="With motion blur", width="stretch")
        st.caption(
            "Motion blur destroys high-frequency detail irreversibly. Unlike noise, there is no filter that "
            "cleanly undoes it — the correct response is to reject the acquisition and repeat it."
        )

        st.subheader("2.2 Salt-and-pepper noise, and two ways to remove it")
        st.markdown(
            "A faulty sensor produces isolated pure-black and pure-white pixels. Two denoisers are on offer, "
            "and only one of them is right for this artifact."
        )
        noise_cols = st.columns([1, 1])
        noise_amount = noise_cols[0].slider(
            "Noise amount", 0.0, 0.2, 0.05, help="Fraction of pixels replaced by extremes.",
            key="m5_prep_noise",
        )
        filter_type = noise_cols[1].radio(
            "Denoising filter", ["Median", "Gaussian"], key="m5_prep_filter"
        )

        if filter_type == "Median":
            filter_strength = st.slider(
                "Kernel size (odd only)", 3, 11, 3, step=2, key="m5_prep_median_k"
            )
        else:
            filter_strength = st.slider(
                "Gaussian sigma", 0.5, 5.0, 1.0, step=0.5, key="m5_prep_gauss_sigma"
            )

        noisy = random_noise(img, mode="s&p", amount=noise_amount)
        noisy_u8 = img_as_ubyte(noisy)

        if filter_type == "Median":
            denoised = cv2.medianBlur(noisy_u8, filter_strength)
        else:
            denoised = cv2.GaussianBlur(noisy_u8, (5, 5), filter_strength)

        dn_cols = st.columns(3)
        dn_cols[0].image(img, caption="Original", width="stretch")
        dn_cols[1].image(noisy, caption="With salt-and-pepper noise", width="stretch", clamp=True)
        dn_cols[2].image(denoised, caption=f"Denoised ({filter_type})", width="stretch")

        mae = float(np.mean(np.abs(denoised.astype(float) - img.astype(float))))
        st.metric(
            f"Mean absolute error vs. original ({filter_type})",
            f"{mae:.2f}",
            help="Lower is a closer recovery of the clean image. Switch the filter and compare.",
        )

        with st.expander("What to expect"):
            st.write(
                "**Median** replaces each pixel with the middle value of its neighbours. An extreme outlier "
                "is never the median, so the noise vanishes and edges stay sharp.\n\n"
                "**Gaussian** replaces each pixel with a weighted *average*. An extreme value still "
                "contributes to that average, so the noise is spread into its neighbours rather than "
                "removed — the image looks muddy and the noise is still in it.\n\n"
                "Compare the error metric under each. This is the general principle: **the right filter is "
                "the one whose assumption matches the artifact.** Median assumes outliers; Gaussian assumes "
                "smooth additive noise."
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 3 — Feature extraction
    # ═══════════════════════════════════════════════════════════════════════
    if activity == ACTIVITIES[2]:
        st.header("Activity 3: Turning Pixels Into Features")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        st.subheader("3.1 Directional gradients and Sobel")
        st.markdown(
            "A gradient filter responds to change in one direction. Combining the horizontal and vertical "
            "responses gives an orientation-independent outline — the first step in cell counting or "
            "segmentation."
        )
        f_cols = st.columns(3)
        horiz_strength = f_cols[0].slider("Horizontal strength", 0.5, 5.0, 1.0, key="m5_prep_horiz")
        vert_strength = f_cols[1].slider("Vertical strength", 0.5, 5.0, 1.0, key="m5_prep_vert")
        sobel_strength = f_cols[2].slider("Sobel strength", 0.5, 5.0, 1.0, key="m5_prep_sobel")

        base_horiz = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
        base_vert = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
        img_horiz = cv2.filter2D(gray, cv2.CV_64F, horiz_strength * base_horiz)
        img_vert = cv2.filter2D(gray, cv2.CV_64F, vert_strength * base_vert)
        magnitude = cv2.normalize(
            np.sqrt(img_horiz**2 + img_vert**2), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        sobel = cv2.convertScaleAbs(sobel * sobel_strength)

        grad_cols = st.columns(5)
        grad_cols[0].image(gray, caption="Greyscale", width="stretch")
        grad_cols[1].image(cv2.convertScaleAbs(img_horiz), caption="Horizontal", width="stretch")
        grad_cols[2].image(cv2.convertScaleAbs(img_vert), caption="Vertical", width="stretch")
        grad_cols[3].image(magnitude, caption="Magnitude", width="stretch")
        grad_cols[4].image(sobel, caption="Sobel", width="stretch")
        st.caption(
            "The horizontal filter finds top and bottom boundaries; the vertical filter finds left and right. "
            "Neither alone is a complete outline — which is why magnitude, not either component, is the "
            "feature you would hand to a model."
        )

        st.divider()
        st.subheader("3.2 Texture: GLCM on malignant vs. benign tissue")
        st.markdown(
            "Some tissue differences are not edges at all — they are **texture**, the statistical "
            "relationship between neighbouring intensities. A grey-level co-occurrence matrix turns that "
            "into two numbers you can compare across slides."
        )

        tex_cols = st.columns([1, 1])
        data_mode = tex_cols[0].selectbox(
            "Pathology slide:", ["Malignant sample", "Benign sample"], key="m5_prep_slide"
        )
        dist = tex_cols[1].slider(
            "Pixel distance", 1, 10, 1, help="How far apart the co-occurring pixel pairs are.",
            key="m5_prep_glcm_dist",
        )
        use_norm = st.toggle(
            "Standardize intensity before analysis", value=True,
            help="Rescales to the full 0–255 range so a brighter scan is not mistaken for a different texture.",
            key="m5_prep_glcm_norm",
        )

        slide_path = (
            "assets/datasets/images/small_slide_BC.png"
            if data_mode == "Malignant sample"
            else "assets/datasets/images/small_slide_noBC.png"
        )

        @st.cache_data
        def load_slide(path):
            return np.array(Image.open(path).convert("L"))

        slide = load_slide(slide_path)
        if use_norm:
            slide = (
                (slide - slide.min()) / (slide.max() - slide.min() + 1e-5) * 255
            ).astype(np.uint8)

        glcm = graycomatrix(slide, distances=[dist], angles=[0], levels=256, symmetric=True, normed=True)
        contrast_val = float(graycoprops(glcm, "contrast")[0, 0])
        correlation_val = float(graycoprops(glcm, "correlation")[0, 0])

        glcm_cols = st.columns([2, 1])
        glcm_cols[0].image(slide, caption=f"{data_mode}", width="stretch")
        with glcm_cols[1]:
            st.metric("GLCM contrast", f"{contrast_val:.2f}", help="Local intensity variation. Higher = coarser.")
            st.metric(
                "GLCM correlation", f"{correlation_val:.4f}",
                help="How predictable a neighbour's value is. Higher = smoother, more organized.",
            )
        st.caption(
            "Switch between the two slides and record both numbers. Then change the pixel distance and note "
            "that the values move — a texture feature is only comparable across images if the acquisition "
            "*and* the parameters are identical. That is a pipeline requirement, not a footnote."
        )

        st.divider()
        st.subheader("3.3 Morphology: closing gaps in a segmentation")
        st.markdown(
            "Speckle and dropout leave holes in what should be a solid region. **Closing** — dilation then "
            "erosion — fills gaps smaller than its structuring element while leaving the region's outer "
            "boundary intact."
        )

        radius = st.slider(
            "Disk radius", 1, 15, 5, help="Gaps smaller than this are filled; larger ones survive.",
            key="m5_prep_radius",
        )

        @st.cache_data
        def load_ultrasound():
            return np.array(Image.open("assets/datasets/images/breast_US.png").convert("L"))

        us = load_ultrasound()
        if use_norm:
            us = ((us - us.min()) / (us.max() - us.min() + 1e-5) * 255).astype(np.uint8)

        closed = closing(us, disk(radius))

        m_cols = st.columns(3)
        m_cols[0].image(us, caption="Original (with speckle/gaps)", width="stretch")
        m_cols[1].image(closed, caption=f"Closed (radius {radius})", width="stretch")
        with m_cols[2]:
            fig_diff, ax_diff = plt.subplots(figsize=(4, 4))
            diff = closed.astype(float) - us.astype(float)
            im = ax_diff.imshow(diff, cmap="magma")
            plt.colorbar(im, ax=ax_diff, fraction=0.046)
            ax_diff.set_title("Action map")
            ax_diff.axis("off")
            st.pyplot(fig_diff)
            plt.close(fig_diff)

        changed = float((diff > 0).mean())
        st.metric("Share of pixels the operation altered", f"{changed:.1%}")
        st.caption(
            "The action map is the honest view: bright regions are pixels the operation **invented**. At a "
            "small radius that is defensible gap-filling; push the radius up and watch the altered share "
            "climb until you are no longer cleaning a lesion boundary but drawing one."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 4 — Trust
    # ═══════════════════════════════════════════════════════════════════════
    if activity == ACTIVITIES[3]:
        st.header("Activity 4: Is This Pipeline Consistent Enough to Trust?")

        st.markdown(
            """
        Everything above was a tool. This is the judgement the tools exist to support — and it is the part
        that decides whether a clinical imaging model is safe to deploy.

        Consider the application you would actually build: **fracture screening**, **pathology slide triage**,
        or **ultrasound lesion assessment**. Each one inherits every choice you made in activities 1–3: the
        normalization, the denoiser, the kernel size, the GLCM distance, the structuring element. A pipeline
        that is not reproducible does not have a performance number — it has a coincidence.
        """
        )

        application = st.selectbox(
            "Application you are evaluating:",
            ["Fracture screening", "Pathology slide triage", "Ultrasound lesion assessment"],
            key="m5_trust_application",
        )

        st.subheader("The consistency gate")
        checks = {
            "Scanner or acquisition protocol is documented": st.checkbox(
                "Scanner or acquisition protocol is documented", key="m5_trust_acquisition"
            ),
            "Preprocessing is applied identically at training and inference": st.checkbox(
                "Preprocessing is applied identically at training and inference", key="m5_trust_preprocessing"
            ),
            "Clinical labels use a shared, written definition": st.checkbox(
                "Clinical labels use a shared, written definition", key="m5_trust_labels"
            ),
            "Evaluation includes external or site-held-out data": st.checkbox(
                "Evaluation includes external or site-held-out data", key="m5_trust_evaluation"
            ),
            "Subgroup performance is reported, not just the aggregate": st.checkbox(
                "Subgroup performance is reported, not just the aggregate", key="m5_trust_subgroup"
            ),
            "There is a defined escalation path for uncertain outputs": st.checkbox(
                "There is a defined escalation path for uncertain outputs", key="m5_trust_escalation"
            ),
        }

        score = sum(checks.values())
        st.metric("Consistency score", f"{score}/{len(checks)}")

        if score == len(checks):
            st.success(
                f"All six gates pass. **{application}** has a defensible pipeline — every number it produces "
                "can be traced to a documented decision and reproduced by someone else."
            )
        elif score >= 4:
            st.warning(
                f"{len(checks) - score} gate(s) still open. **{application}** may perform well in "
                "development and fail silently at another site, and you would not be able to say why."
            )
        else:
            st.error(
                f"**{application}** is not ready to be evaluated, let alone deployed. With this many gates "
                "open, a good result is indistinguishable from a lucky one."
            )

        st.subheader("Consider this:")
        st.text_area(
            f"Which step in a {application.lower()} workflow is most likely to introduce inconsistency, "
            "and how would you detect it?",
            key="m5_trust_risk",
        )
        st.text_area(
            "What patient-safety risk appears if this pipeline fails silently — and what evidence would you "
            "require before deployment?",
            key="m5_trust_evidence",
        )
        st.text_area(
            "How would you standardize this workflow so a collaborating site reproduces your results?",
            key="m5_trust_standardize",
        )

        with st.expander("Expected considerations"):
            st.write(
                "Strong answers name image quality and acquisition protocol, subgroup performance, external "
                "validation at a site that did not contribute training data, fit with the existing clinical "
                "workflow, and a human review step for low-confidence outputs. Note that the first and last "
                "of those are not modelling problems at all."
            )

st.markdown(
    """
---
**Key takeaways**

- Augmentation teaches invariance; it also invents pixels at the edges. Know what your transform fabricates.
- **The right filter is the one whose assumption matches the artifact.** Median for outliers, Gaussian for
  smooth noise, and nothing at all for motion blur — reject and re-acquire.
- Edges, texture, and morphology answer different questions. A feature is only comparable across images when
  the acquisition and the parameters are identical.
- A morphological operation's action map shows you what it invented. Look at it before you trust it.
- The consistency gate is the deliverable. Without it, a performance number is a coincidence.

**Resources:** [scikit-image morphology](https://scikit-image.org/docs/stable/api/skimage.morphology.html) ·
[GLCM texture features](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html) ·
[OpenCV smoothing](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
"""
)
