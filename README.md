# AIstage-segmentation

## Summary

## Experimental Results
### U-Net (ResNet35 Encoder) Reference Model
mIOU : 0.4495 [{'Backgroud': 0.9472}, {'General trash': 0.3215}, {'Paper': 0.6834}, {'Paper pack': 0.2977}, {'Metal': 0.2478}, {'Glass': 0.4147}, {'Plastic': 0.3806}, {'Styrofoam': 0.5683}, {'Plastic bag': 0.7865}, {'Battery': 0.0}, {'Clothing': 0.297}]
### Added Focal Loss
mIOU : 0.4538 [{'Backgroud': 0.9411}, {'General trash': 0.3232}, {'Paper': 0.6823}, {'Paper pack': 0.3066}, {'Metal': 0.2617}, {'Glass': 0.4105}, {'Plastic': 0.3398}, {'Styrofoam': 0.5991}, {'Plastic bag': 0.7805}, {'Battery': 0.0}, {'Clothing': 0.3476}]
### Added Focal Loss with balanced alpha
mIOU : 0.4754 [{'Backgroud': 0.9321}, {'General trash': 0.3059}, {'Paper': 0.6511}, {'Paper pack': 0.265}, {'Metal': 0.2548}, {'Glass': 0.3026}, {'Plastic': 0.3004}, {'Styrofoam': 0.5791}, {'Plastic bag': 0.7449}, {'Battery': 0.6648}, {'Clothing': 0.2286}]
### Added Dilated Convolution layer instead of Maxpooling
mIOU : 0.4267 [{'Backgroud': 0.9413}, {'General trash': 0.3164}, {'Paper': 0.6711}, {'Paper pack': 0.1828}, {'Metal': 0.3001}, {'Glass': 0.3042}, {'Plastic': 0.314}, {'Styrofoam': 0.6276}, {'Plastic bag': 0.7797}, {'Battery': 0.0}, {'Clothing': 0.2567}]
### Dilated Convolution layer + Focal Loss
mIOU : 0.3917 [{'Backgroud': 0.9108}, {'General trash': 0.2568}, {'Paper': 0.5211}, {'Paper pack': 0.1715}, {'Metal': 0.2372}, {'Glass': 0.3014}, {'Plastic': 0.2004}, {'Styrofoam': 0.3829}, {'Plastic bag': 0.6843}, {'Battery': 0.4051}, {'Clothing': 0.2373}]
### CRF + Focal loss (balanced)
mIOU : 0.3865 [{'Backgroud': 0.8933}, {'General trash': 0.2986}, {'Paper': 0.5106}, {'Paper pack': 0.2364}, {'Metal': 0.1952}, {'Glass': 0.3568}, {'Plastic': 0.2525}, {'Styrofoam': 0.5125}, {'Plastic bag': 0.6614}, {'Battery': 0.2326}, {'Clothing': 0.1018}]

## Instructions

## Approach

### Data
Oversampling
### Model
U-Net based
Dilated Convolution
### Loss
Focal Loss
Weighted Alpha for Imbalance dataset
### Post-processing
CRF