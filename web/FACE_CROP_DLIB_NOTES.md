# DeepfakeBench dlib face cropping

Backend auto requires dlib in live mode. Mock mode may fall back to Haar,
with a warning. Explicit haar remains available for development.
The installed preprocess.py is loaded once without chdir; dlib HOG detector
and 81-landmark predictor are cached. Signature and nested scale=1.3 are
checked. The upstream implementation returns (BGR crop, landmarks, mask).
Output must be uint8 256x256; PNG is encoded using cv2.imwrite. Video samples
up to 32 distinct frames using endpoint-inclusive linspace.

No model, calibrator, WFS, threshold, or verdict logic is changed.
Using the same function is not proof of parity with historical PNG bytes.
Server verification still required: actual preprocess.py/predictor hashes,
the real dlib integration test, a known historical frame/crop comparison,
and live readiness plus API provenance. No quality improvement is claimed.
Nothing is pushed or deployed by this installer.
