import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform, img_as_ubyte
from skimage.util import random_noise

st.markdown(
    """
Subsection 5.1 established that a pixel is a measurement and that intensity operations change what is
visible, not what was captured. Now the work: real artifacts, real feature extraction, and the question that
matters more than any single filter — **is this pipeline consistent enough to trust?**

1. **Augmentation** — teach a model the shape of a structure rather than its position on the slide.
2. **Artifacts** — motion blur and sensor noise, and why the right denoiser depends on the artifact.
3. **Features** — directional and Sobel gradients, Canny edges, and Otsu's automatic threshold.
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
        st.subheader("3.2 Sobel vs. Canny vs. Otsu on a controlled image")
        st.markdown(
            """
        Real images confound two things: how good the algorithm is, and how noisy the input was. So here the
        input is **synthetic** — a bright rectangle on a dark field, with noise you control. Now a difference
        between the methods is a fact about the methods.
        """
        )

        p_cols = st.columns(3)
        noise_level = p_cols[0].slider(
            "Noise level", 1, 100, 50, help="Random intensity added to every pixel.",
            key="m5_prep_synth_noise",
        )
        kernel_size = p_cols[1].slider(
            "Sobel kernel size", 3, 11, 5, step=2,
            help="The derivative window. Larger is smoother but less precise.",
            key="m5_prep_synth_ksize",
        )
        blur_sigma = p_cols[2].slider(
            "Gaussian blur sigma (before Otsu)", 0.0, 5.0, 1.0, key="m5_prep_synth_sigma"
        )

        t_cols = st.columns(3)
        threshold1 = t_cols[0].slider(
            "Canny threshold 1", 0, 255, 100, help="Lower hysteresis bound.", key="m5_prep_synth_t1"
        )
        threshold2 = t_cols[1].slider(
            "Canny threshold 2", 0, 255, 200, help="Upper hysteresis bound.", key="m5_prep_synth_t2"
        )
        colormap = t_cols[2].selectbox(
            "Colormap", ["gray", "viridis", "plasma", "magma", "inferno"], key="m5_prep_synth_cmap"
        )

        @st.cache_data
        def make_synthetic_target(noise):
            base = np.zeros((100, 100), dtype=np.uint8)
            cv2.rectangle(base, (30, 30), (70, 70), 255, -1)
            rng = np.random.default_rng(42)
            grain = rng.integers(0, noise, (100, 100)).astype(np.uint8)
            return cv2.add(base, grain)

        synth = make_synthetic_target(noise_level)

        sob_x = cv2.Sobel(synth, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sob_y = cv2.Sobel(synth, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel_edges = cv2.normalize(
            cv2.magnitude(sob_x, sob_y), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        canny_edges = cv2.Canny(synth, threshold1, threshold2)
        blurred = cv2.GaussianBlur(synth, (5, 5), blur_sigma)
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        titles = ["Input image", "Sobel filter", "Canny edges", "Otsu threshold"]
        panels = [synth, sobel_edges, canny_edges, otsu_thresh]
        explanations = [
            "Raw input. The rectangle's true boundary is a perfect square — every departure from that in the "
            "other three panels is the method's error, not the data's.",
            "Sobel is sensitive to noise because it differentiates the intensity directly, and "
            "differentiation amplifies high frequencies. Raise the noise level and watch the whole field light up.",
            "Canny is more robust: it blurs first, then keeps only gradient maxima, then follows weak edges "
            "only where they connect to strong ones. That is why its output is thin and mostly closed.",
            "Otsu does not look for edges at all. It picks the intensity threshold that best separates the "
            "image into two classes — which works beautifully here and fails the moment the object and "
            "background overlap in intensity.",
        ]

        panel_cols = st.columns(4)
        for i, col in enumerate(panel_cols):
            with col:
                st.markdown(f"**{titles[i]}**")
                fig_p, ax_p = plt.subplots(figsize=(3, 3))
                ax_p.imshow(panels[i], cmap=colormap)
                ax_p.axis("off")
                st.pyplot(fig_p)
                plt.close(fig_p)
                if st.checkbox("Reveal logic", key=f"m5_prep_reveal_{i}"):
                    st.info(explanations[i])

        st.caption(
            "Turn the noise up to 100 and compare Sobel against Canny. Then set the blur sigma to 0 and watch "
            "Otsu's clean segmentation break apart — the pre-blur was doing more work than the thresholding was."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 4 — Trust
    # ═══════════════════════════════════════════════════════════════════════
    if activity == ACTIVITIES[3]:
        st.header("Activity 4: Is This Pipeline Consistent Enough to Trust?")

        st.markdown(
            """
        Everything above was a tool. This is the judgement the tools exist to support — and it is the part
        that decides whether an imaging result is publishable.

        Consider the application you would actually build: **microscopy cell analysis**, **tissue slide
        quality control**, or **image-based phenotype screening**. Each one inherits every choice you made in
        activities 1–3: the normalization, the denoiser, the kernel size, the threshold. A pipeline that is not
        reproducible does not have a performance number — it has a coincidence.
        """
        )

        application = st.selectbox(
            "Application you are evaluating:",
            [
                "Microscopy cell analysis",
                "Tissue slide quality control",
                "Image-based phenotype screening",
            ],
            key="m5_trust_application",
        )

        st.subheader("The consistency gate")
        checks = {
            "Acquisition settings are documented": st.checkbox(
                "Acquisition settings are documented", key="m5_trust_acquisition"
            ),
            "Preprocessing is applied identically at training and inference": st.checkbox(
                "Preprocessing is applied identically at training and inference", key="m5_trust_preprocessing"
            ),
            "Labels use a shared, written definition": st.checkbox(
                "Labels use a shared, written definition", key="m5_trust_labels"
            ),
            "Evaluation includes a held-out set from a different batch": st.checkbox(
                "Evaluation includes a held-out set from a different batch", key="m5_trust_evaluation"
            ),
            "Per-batch and per-plate performance is reported, not just the aggregate": st.checkbox(
                "Per-batch and per-plate performance is reported, not just the aggregate",
                key="m5_trust_subgroup",
            ),
            "Uncertain or failed images have a defined handling rule": st.checkbox(
                "Uncertain or failed images have a defined handling rule", key="m5_trust_escalation"
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
                f"{len(checks) - score} gate(s) still open. **{application}** may work on this batch and fail "
                "silently on the next, and you would not be able to say why."
            )
        else:
            st.error(
                f"**{application}** is not ready to be evaluated, let alone published. With this many gates "
                "open, a good result is indistinguishable from a lucky one."
            )

        st.subheader("Consider this:")
        st.text_area(
            f"Which step in a {application.lower()} workflow is most likely to introduce inconsistency, "
            "and how would you detect it?",
            key="m5_trust_risk",
        )
        st.text_area(
            "What conclusion could you draw wrongly if this pipeline failed silently — and what evidence "
            "would you require before believing a result?",
            key="m5_trust_evidence",
        )
        st.text_area(
            "How would you standardize this workflow so a collaborating lab reproduces your results?",
            key="m5_trust_standardize",
        )

        with st.expander("Expected considerations"):
            st.write(
                "Strong answers name image quality and acquisition settings, per-batch performance, "
                "validation on images the model has never seen from a batch it has never seen, and a rule "
                "for handling failed acquisitions. Note that the first and last of those are not modelling "
                "problems at all."
            )

st.markdown(
    """
---
**Key takeaways**

- Augmentation teaches invariance; it also invents pixels at the edges. Know what your transform fabricates.
- **The right filter is the one whose assumption matches the artifact.** Median for outliers, Gaussian for
  smooth noise, and nothing at all for motion blur — reject and re-acquire.
- Sobel differentiates and so amplifies noise; Canny blurs, thins, and links; Otsu ignores edges entirely and
  splits on intensity. They fail in different ways, which is why the synthetic target is worth the detour.
- The consistency gate is the deliverable. Without it, a performance number is a coincidence.

**Resources:** [OpenCV image gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html) ·
[OpenCV thresholding (Otsu)](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) ·
[scikit-image transform](https://scikit-image.org/docs/stable/api/skimage.transform.html)
"""
)
