# Project 01 — Build Your First AI: Image Classifier

> **🛠️ Stack for this lesson** — Python / Jupyter notebook · runs in Google Colab.
> 📥 Template: [/learn/ai25/template/project-01-image-classifier](/learn/ai25/template/project-01-image-classifier)

The Foundation Arc capstone. Two deliverables in one notebook: (A) a CNN you trained from scratch on CIFAR-10, (B) a transfer-learned ResNet-18 fine-tuned on a custom 3-class dataset *you* collect. The notebook ships with the data loaders, model skeletons, and evaluation harness. The training, the dataset curation, and the per-class diagnostics are yours.

**Time:** ~3 hours (Session 7) · **Concepts:** 04 (CNNs) + 06 (Transfer Learning)

---

## What You'll Build

**Part A — CNN from scratch on CIFAR-10**

| # | TODO |
|---|------|
| A1 | Build a CNN with at least 3 conv layers and reach **≥ 70% test accuracy** in 10 epochs |
| A2 | Add data augmentation (`RandomCrop`, `RandomHorizontalFlip`) and re-train; report change in test accuracy |
| A3 | Plot per-class accuracy on the test set and identify your worst class |

**Part B — Transfer learning on a custom dataset**

| # | TODO |
|---|------|
| B1 | Collect ≥ 50 images for each of 3 classes you choose; organise as `custom_data/{class}/img.jpg` |
| B2 | Replace ResNet-18's classification head with a 3-class linear layer; freeze the backbone |
| B3 | Fine-tune for ≤ 10 epochs and reach **≥ 85% test accuracy** |
| B4 | Unfreeze the last conv block, lower the LR, re-train for ≤ 5 epochs; report new accuracy |

## Run It

1. Open `project-01-image-classifier.ipynb` in **[Google Colab](https://colab.research.google.com/)**.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run cells top-to-bottom for Part A; in Part B you'll upload your custom dataset before running.

Total runtime ≈ 15 minutes on T4 with both parts.

## Verify

- [ ] Part A test accuracy ≥ 70% with augmentation
- [ ] Part A per-class accuracy plotted
- [ ] Part B custom dataset directory contains ≥ 150 images across 3 class folders
- [ ] Part B head-only fine-tune reaches ≥ 85%
- [ ] Part B unfrozen fine-tune is reported (improvement or no-improvement, both are valid findings)

## Investigation

> **📓 Where this lives now:** open your lesson note for this activity.
> Click **Take Notes** at the top of the lesson page if you don't already have a note —
> it creates one titled `<Lesson title> — <COURSE>` with a scaffold pre-filled from
> the lesson headings. Add an H2 section titled **`Investigation`** to that note and put
> your work there. The note persists across devices and is queryable by the AI Tutor.
>
> 💡 **Tip:** open the AI Tutor while viewing your note and ask
> *"help me draft my Investigation findings for [topic]"*. The Tutor will append a draft to your note;
> you can drag it into the **Investigation** section if it lands elsewhere.

**Prompts to answer in your note:**

CIFAR-10 has ~6000 images per class — your custom dataset has ~50. The performance gap between Part A and Part B is the lesson.

- Plot per-class accuracy for both Part A and Part B. Where does Part B do *better* than Part A? Why?
- Count parameters in your Part A CNN vs the ResNet-18 backbone. The ratio matters. Document why a 50-image-per-class fine-tune of a giant pretrained model can beat a from-scratch model with 600× more data.
- For your worst-performing custom class, inspect 5 misclassified test images. Are they ambiguous or did the model fail on a clear case?

## Stretch

Pick one:
- Replace ResNet-18 with `efficientnet_b0`. Smaller, similar accuracy?
- Add Grad-CAM visualisations on 3 test images per class. Where is the model "looking"?
- Train for 30 epochs with cosine-annealed learning rate.

## Grading Rubric

| Component | Weight |
|-----------|--------|
| Part A: CNN from scratch reaches ≥ 70% with augmentation | 25% |
| Part B: Custom dataset of ≥ 150 images, organised correctly | 15% |
| Part B: Head-only fine-tune reaches ≥ 85% | 25% |
| Investigation writeup with per-class analysis | 20% |
| Reflection (3 prompts, specific evidence) | 15% |

## 🪞 Reflect on Your Work

Answer in 2-3 sentences each, in this README under your TODO commits. Your tutor reads these as part of grading.

1. **What did you learn that you didn't know before?** Name the most surprising thing — a bug you hit, a syntax quirk, a way the simulator and real device differ.
2. **How did you collaborate with AI?** If you used Claude / ChatGPT / Cursor / Copilot, what part of the work did *you* contribute — the prompt, the verification, the design decision, the bug-fix? If you didn't use AI, what was the hardest thing to figure out alone?
3. **How do you know your code works?** Describe one specific thing you did to confirm — a test you ran, a screenshot you took, a behavior you watched on the device.

> AI is a great collaborator. Owning your thinking, verifying the output, and explaining your design choices is what *learning* looks like in this course.

## Submit

When the Verify checklist is green, head to **[/learn/ai25/certification](/learn/ai25/certification)** and submit your notebook link plus your custom dataset (zipped). Your Investigation section will be graded from your Knowledge Notebook.

<!-- claude-template-fix: readme-v3 -->

<!-- claude-template-fix: notes-migration-v1 -->
