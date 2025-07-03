# IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization
<img src="flowchart.jpg" alt="IAP Teaser" width="700"/>

## Abstract:

<p>
        Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of
        computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios
        or fail to produce contextually coherent adversarial patches,
        causing them to be easily noticeable by human examiners
        and insufficiently stealthy against automatic patch defenses.
        In this paper, we introduce IAP, a novel attack framework
        that generates highly invisible adversarial patches based on
        perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper
        location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of
        patch location to both victim model prediction and human
        visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes
        color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks
        and model architectures demonstrate that IAP consistently
        achieves competitive attack success rates in targeted settings
        with significantly improved patch invisibility compared to
        existing baselines. In addition to being highly imperceptible
        to humans, IAP is shown to be stealthy enough to render
        several state-of-the-art patch defenses ineffective.
</p>

## Highlights:

<ul>
        <li>We propose a novel imperceptible adversarial patch attack method using perceptibility-aware optimization.</li>
        <li>We acheive state-of-the-art attack efficacy and imperceptibility on ImageNet and VGGFace dataset</li>
        <li>We successfully bypass multiple state-of-the-art adversarial patch defense methods</li>
      </ul>

Official Code and instructions for our paper IAP. The code will be released soon.
