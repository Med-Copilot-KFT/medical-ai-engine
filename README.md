# medical-ai-engine
Medical imaging AI engine for multimodal analysis, medical image preprocessing, 3D segmentation, abnormality detection, and clinician-oriented decision support.
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





# Technologies Used

## Core Development

| Technology | Purpose |
|---|---|
| Python 3.11 | Core development and medical AI pipeline |
| PyTorch | Deep learning model inference |
| MONAI | Medical imaging AI framework and 3D medical image processing |
| MONAI Model Zoo | Pretrained medical imaging models |
| MONAI Label | Medical imaging annotation and AI-assisted labeling research |
| NumPy | Numerical and array-based medical image processing |
| NiBabel | NIfTI medical image loading, spatial metadata, and volume handling |
| SciPy | Scientific computing and image-processing operations |
| OpenCV | Image processing and computer vision operations |
| Pillow (PIL) | Standard image loading and metadata extraction |
| Tesseract OCR | Text extraction from images |
| psutil | Runtime memory/resource monitoring |

## Medical Imaging

- **DICOM** — Medical imaging data standard and clinical imaging workflow
- **NIfTI** — 3D medical imaging volume format
- **Medical Segmentation Decathlon** — Initial benchmark dataset
- **3D CT Processing** — Volumetric preprocessing and inference
- **Sliding-Window Inference** — Memory-efficient processing of large 3D volumes

## AI / Multimodal Analysis

| Technology | Role |
|---|---|
| PyTorch | Neural network execution |
| MONAI | Medical imaging models and preprocessing |
| 3D U-Net | Spleen CT segmentation architecture |
| Google Gemini | Multimodal image understanding and future explanation/reporting layer |

## Image Processing

- OpenCV
- Pillow
- Tesseract OCR
- NumPy
- NiBabel

## Development & Testing

- VS Code
- Python Virtual Environment (`venv`)
- Terminal / zsh
- cURL
- Thunder Client
- Git
- GitHub



# Technology Architecture

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


That tells the story much better than simply saying:

> Python, PyTorch, MONAI, Gemini, OpenCV...

---

### One correction I'd make

Don't list **DICOM** as if you've already completed the entire DICOM pipeline if that part is still under development.

We can separate technologies into:

**Implemented**
- Python
- PyTorch
- MONAI
- MONAI Model Zoo
- NiBabel
- NumPy
- OpenCV
- Pillow
- Tesseract
- Gemini
- NIfTI
- 3D CT inference

**In development**
- DICOM series processing
- Modality classification
- Model registry
- API layer
- Additional CT/MRI/X-ray models

That distinction will make the README much more credible to your boss.

And yes, I think the **technology section should be fairly prominent**, because your README is effectively documenting the engineering work you've done, not just advertising the project. GitHub also recommends scannable structure using headings, tables, lists, and visuals, which fits this format well. :contentReference[oaicite:1]{index=1}

If you want, I can also give you a **single polished final `README.md` combining everything we discussed**—overview + objectives + technologies + architecture + exactly what you implemented + results + roadmap—so you can paste it into VS Code as one file.
