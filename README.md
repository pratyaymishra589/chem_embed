# Context-Aware Contrastive Learning for Chemical Reactivity

This repository contains the codebase for identifying and predicting chemical reactivity using a custom **Message Passing Neural Network (MPNN)**. By leveraging **Supervised Contrastive Learning (SupCon)** and a novel **1-Hop Masked Readout** technique, the model successfully distinguishes between true reaction centers (Positives) and identical, but unreactive, local structures (Hard Negatives) by learning from long-range global steric and electronic effects.

## Overview

Chemical reactivity is traditionally modeled by looking at local functional groups. However, identical functional groups often fail to react due to long-range effects (e.g., bulky substituents or distant electron-withdrawing groups). 

This project solves this by:
1. **Mining Hard Negatives:** Automatically extracting unreactive local substructures from the Open Reaction Database (ORD) that perfectly match the templates of true reacting centers.
2. **Global to Local Graph Processing:** Using a GINEConv-based MPNN to pass chemical messages across the entire molecule (up to 5 bonds away) before compressing the output down to *only* the 1-hop reaction center.
3. **Contrastive Representation Learning:** Pushing the "Positives" and "Hard Negatives" apart in a 64-dimensional latent space, forcing the network to understand global structural context.

## Results

The model's success is validated using **t-SNE** manifold mapping on the final 64D embeddings. 
* **Global Structure:** Distinct chemical rules form isolated clusters (islands), proving the network learns foundational functional group identities.
* **Local Separation:** Within the same reaction rule, identical local sub-structures are successfully split into "Positives" (reacting) and "Hard Negatives" (unreactive) sub-clusters based purely on their distant, global context.
