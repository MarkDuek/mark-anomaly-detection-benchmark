---
title: Protocols
description: Evaluation procedures and data splits.
---

Protocols define when a model may fit and which dataset split it may score. The selected protocol must match the model's declared execution mode.

| Protocol | Model mode | Setup splits | Test split | Fit behavior |
| --- | --- | --- | --- | --- |
| [Inductive](/protocols/inductive/) | `inductive` | `train`, `validation` | `test` | Fit once on `train`; score later splits without refitting. |
| [Transductive](/protocols/transductive/) | `transductive` | `validation` | `test` | Fit and score each split as an independent batch. |

The experiment runner rejects incompatible model and protocol combinations. It calibrates threshold selection from validation scores, then applies that calibration to test scores without using test labels.

## Available Protocols

<div class="documentation-map">
  <a href="/protocols/inductive/"><strong>Inductive</strong><span>Learn from training data and score held-out splits.</span></a>
  <a href="/protocols/transductive/"><strong>Transductive</strong><span>Decompose validation and test batches independently.</span></a>
</div>
