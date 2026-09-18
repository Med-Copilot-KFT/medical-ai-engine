# Medical AI Engine

A modular engine for analyzing medical images and studies — built to handle modality-aware preprocessing, run specialized medical AI models, segment anatomy, flag abnormalities, localize findings, and turn all of that into reports a clinician can actually use.

> **Status:** Active R&D / prototype

---

## What this is

Medical images carry a huge amount of information, and clinicians don't always have time to dig through all of it manually. This project is our attempt at building a reusable engine that can take in different kinds of medical imaging studies and hand back structured, AI-assisted analysis.

Eventually, we want this to be an imaging intelligence layer that can:

- Figure out the modality and study type on its own
- Process medical images and DICOM studies
- Work with both 2D and 3D data
- Apply preprocessing that's specific to the modality in question
- Catch potential abnormalities
- Segment out anatomical structures and regions of interest
- Pinpoint where a finding actually is in the image/volume
- Give quantitative measurements when that makes sense
- Back up its findings with evidence, not just a verdict
- Report confidence/uncertainty honestly
- Summarize everything in a way a clinician can quickly read
- Plug into existing tools like MedCopilot

To be clear: this is meant to support clinical decisions, not replace them. A qualified healthcare professional still makes the final call — this just gives them a head start.

---

## Where we're focusing

We're building this around five main goals.

### 1. Handling multiple imaging modalities

We want one engine that can support:

- CT
- MRI
- X-ray
- Ultrasound
- Mammography
- Retinal imaging
- Pathology imaging
- Endoscopy
- Whatever else comes up as we grow

Every modality needs its own preprocessing and its own models — there's no single generic model that works well across all of them, so we're building the architecture around that reality instead of fighting it.

### 2. Actually understanding the image, not just describing it

We don't just want the system answering "what's in this image?" We want it answering the questions a clinician would actually ask:

- What anatomy am I looking at?
- Is there anything here that needs a closer look?
- Where exactly is the finding?
- Which model produced this result?
- How confident is it?
- What's the evidence behind it?
- Can we quantify it?
- Can we present this in a way that's useful, not just technically correct?

### 3. Keeping specialized models separate from the language layer

General vision-language models are great at describing images and explaining things in plain language, but we don't want them doing the actual diagnostic heavy lifting. So we split the pipeline into two halves:

**The medical AI inference** (segmentation, detection, classification, localization, measurement) — handled by specialized models built for that

**The explanation and reporting** — handled afterward by a language model that turns the structured output into something readable

### 4. Building this to be reused, not bolted on

This is being built as its own standalone engine, not glued to one specific app. That means it should be able to plug into:

- MedCopilot
- Web apps
- Mobile apps
- Clinical research tools
- Internal AI services
- Whatever imaging workflows come next

That's also why this lives in its own repo, separate from the main MedCopilot codebase — so the imaging layer can move at its own pace.

### 5. Moving from research to something production-ready

The rough path we're following:

Research prototype → validated models → structured inference services → API integration → app integration → production deployment

Right now, most of the effort is going into getting the core pipeline solid and validated.

---

## What's working right now

The first full pipeline we've got running end-to-end:

**3D CT scan → preprocessing → pretrained medical segmentation model → 3D spleen segmentation → mapping back to original space → quantitative evaluation → visualization**

Built on MONAI and PyTorch.

---

## The current pipeline, step by step

```text
                 Medical CT Study
                        │
                        ▼
                DICOM / NIfTI Input
                        │
                        ▼
              Medical Image Loading
                        │
                        ▼
              Orientation Standardization
                        │
                        ▼
                Voxel Resampling
                        │
                        ▼
              Intensity Normalization
                        │
                        ▼
              3D Medical AI Model
                        │
                        ▼
             Sliding Window Inference
                        │
                        ▼
               3D Segmentation Mask
                        │
                        ▼
          Resampling to Original CT Grid
                        │
                        ▼
             Quantitative Evaluation
                        │
                        ▼
            Visualization / Verification
```

---

## What's actually in the stack

### Core development

| Technology | What it's doing here |
|---|---|
| Python 3.11 | The backbone of the whole pipeline |
| PyTorch | Running the model inference |
| MONAI | Our medical imaging framework — handles the 3D processing |
| MONAI Model Zoo | Where we're pulling pretrained medical imaging models from |
| MONAI Label | Annotation and AI-assisted labeling, still in research mode |
| NumPy | Array and numerical work across the image pipeline |
| NiBabel | Loading NIfTI files and handling spatial metadata/volumes |
| SciPy | General scientific computing, image-processing ops |
| OpenCV | Computer vision and image processing |
| Pillow (PIL) | Basic image loading and metadata |
| Tesseract OCR | Pulling text out of images when needed |
| psutil | Keeping an eye on memory/resource use at runtime |

### Medical imaging side

- **DICOM** — the standard clinical imaging format we're working with
- **NIfTI** — how we handle 3D imaging volumes
- **Medical Segmentation Decathlon** — our starting benchmark dataset
- **3D CT processing** — volumetric preprocessing and inference
- **Sliding-window inference** — lets us process big 3D volumes without blowing up memory

### AI and multimodal side

| Technology | Role |
|---|---|
| PyTorch | Runs the neural nets |
| MONAI | Models and preprocessing for medical images |
| 3D U-Net | The architecture behind our spleen CT segmentation |
| Google Gemini | Handles multimodal image understanding, and will drive the explanation/reporting layer down the line |

### Image processing tools

OpenCV, Pillow, Tesseract OCR, NumPy, NiBabel

### How we build and test

VS Code, a Python virtual environment (`venv`), terminal/zsh, cURL, Thunder Client, Git, GitHub

---

## How it all fits together

```text
                    Medical Imaging Data
                            │
                 ┌──────────┴──────────┐
                 │                     │
               DICOM                 NIfTI
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    Python Processing
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           NiBabel        NumPy        OpenCV
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     MONAI Pipeline
                            │
                    PyTorch Inference
                            │
                     3D U-Net Model
                            │
                            ▼
                  Medical AI Prediction
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          Segmentation             Evaluation
                │                       │
                ▼                       ▼
          Visualization            Dice Score

                            │
                            ▼
                    Gemini Multimodal
                    Reasoning / Reporting
```

