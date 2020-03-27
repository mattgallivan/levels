# Learning Notes for Image to Level Generation

## Overall Approach
Currently thinking that we'll want to subdivide the problem into:

- Loading images in
- foreach image, extract features (likely patch/pixel colour histograms initially)
- foreach set of features, generate levels (initially a comparator that matches the closest sprite foreach game in the library) 

## Feature Extraction

Possible features include:

 - Colour representation of the pixels or tiles (patch of pixels) via RGB histogram
 - Edges via edge detection

## Comparing Images

### Comparing Colours
Initially the thought was to look at comparing colour histograms, but when I started comparing images via histograms for the pitch presentation, I realized that mixed colours like yellow (green + red) can have channels of, say, green that are stronger than a pure-green image.  Thus, I looked into *comparing images* ([1](https://stackoverflow.com/questions/9018016/how-to-compare-two-colors-for-similarity-difference)), which led to a Wiki article on [Color Difference](https://en.wikipedia.org/wiki/Color_difference) and along similar lines, led me to this recommended method:
https://www.compuphase.com/cmetric.htm

It's used by a few people on the stackoverflow page, and I recall reading it as a recommended formulation.

That all said, this only compares one pixel colour to another.  We'll want to compare a bunch of them together.

### Histogram Comparison

This site has a straightforward method and explanation of the techniques:
https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

Will likely use this method (want to check on how binning is done).

