# VisionAssist

A macOS smart-glasses prototype for visually impaired users. VisionAssist combines
local object detection, camera-relative directions, estimated depth, offline text
reading, and optional ChatGPT scene descriptions.

The camera preview targets **30 FPS**. Object detection checks the latest frame
**once per second**, automatically, and speaks when objects enter, leave, or
change direction. Local processing does not need an OpenAI API key.

## Features and controls

Focus the camera window before pressing a key.

| Key | Action |
| --- | --- |
| Automatic | Detect objects and announce meaningful changes every second |
| Space | Ask ChatGPT to describe the current camera image |
| L | Connect your ChatGPT account through browser sign-in |
| R | Read visible English text offline using Apple Vision |
| D | Describe which detected object appears closest |
| V | Show or hide the relative-depth snapshot |
| M or m | Show or hide the black-and-white change mask |
| Escape | Close the app and stop its workers and speech |

The standard detector recognizes 365 categories. The change mask highlights
changed pixels; the detector still receives the original color image. Depth and
OCR run on separate background workers so they do not block object detection.

## Requirements

- macOS with camera access; Apple Vision provides offline OCR.
- Xcode Command Line Tools and CMake 3.20 or newer.
- OpenCV with its C++ development libraries.
- Python 3 for optional cloud descriptions and one-time model preparation.

On a Mac using Homebrew:

```sh
xcode-select --install
brew install cmake opencv python
```

## Build and run

Run these commands from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/VisionAssist
```

**This checkout already has the local model weights.** Fresh GitHub clones need
the model preparation below before launching the camera app. Weights are excluded
from Git to keep the repository small.

In CLion, open the root `CMakeLists.txt`, select the `VisionAssist` target, and set
the run working directory to the repository root. Existing local builds can still
use `cmake-build-debug`. Allow camera access when macOS asks.

Shared camera/audio settings are in [`config/default.txt`](config/default.txt).
An alternate local configuration can be selected with:

```sh
./build/VisionAssist --config path/to/settings.txt
```

## Prepare the local models

The app runs ONNX models directly in C++. Python model tools are used only for
this preparation step, not on every video frame.

```sh
python3 -m venv .model-tools
.model-tools/bin/python3 -m pip install -r requirements/models.txt
.model-tools/bin/python3 tools/models/prepare_objects365.py --variant n
.model-tools/bin/python3 tools/models/prepare_depth.py
```

These commands download pretrained checkpoints, export fixed-size ONNX files,
and validate them with OpenCV. They do not download the full training datasets.
The exports, labels, provenance, and hashes live under `models/`.

| Task | Model | Input |
| --- | --- | --- |
| Objects | YOLO26n, pretrained on Objects365 | 640 × 640 |
| Depth | YOLO26n-depth | 512 × 512 |
| Text | Apple Vision fast English OCR | Captured camera image |

## Optional ChatGPT and OpenAI setup

Cloud descriptions happen only when you press **Space**. All AI settings and
credentials belong in **`.local/`**, which is excluded from Git and the source
archive. The public settings file contains no AI credentials.

```sh
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements/cloud.txt
mkdir -p .local
chmod 700 .local
cp config/openai.example.conf .local/openai.conf
chmod 600 .local/openai.conf
```

If your local settings already exist, keep them instead of copying the example.
Existing credentials and sign-in sessions were preserved during the refactor.

The default mode uses browser sign-in with your ChatGPT account. Press **L**, or:

```sh
.venv/bin/python3 python/chatgpt_auth.py login
```

For API mode, set `ai_auth_mode=api_key` in `.local/openai.conf` and supply
`OPENAI_API_KEY`, or place your key in `.local/openai_api_key` with private file
permissions. API billing is separate from a ChatGPT subscription. The model
setting also belongs in `.local/openai.conf`.

## Folder guide

```text
VisionAssist/
├── src/                    C++ implementations, grouped by feature
│   ├── app/                Startup, settings, and camera orchestration
│   ├── vision/             Detection, tracking, change mask, and depth
│   ├── text/               OCR and the small Apple framework adapter
│   ├── audio/              Serialized speech
│   ├── ai/                 Explicit ChatGPT requests
│   ├── runtime/            Background specialist workers
│   └── ui/                 Camera overlays and frame pacing
├── include/visionassist/   Matching .hpp interfaces in the same feature groups
├── python/                 ChatGPT authentication and image-request helpers
├── tests/cpp/              Native tests and installed-model smoke checks
├── tests/python/           Mocked cloud/authentication tests
├── tools/models/           One-time model export scripts
├── tools/package_source.py Credential check and clean source ZIP creation
├── models/                 ONNX weights (ignored), labels, and model manifests
├── assets/                 Friendly object labels and category resources
├── config/                 Shared settings and a credential-free AI example
├── requirements/           Separate cloud and model-tool dependencies
├── docs/                   Codebase learning guide
└── .local/                 Private settings, sessions, key, and camera snapshot
```

Start reading [`src/app/main.cpp`](src/app/main.cpp), then
[`src/app/VisionApplication.cpp`](src/app/VisionApplication.cpp).
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains each module and the frame flow.

## Tests

```sh
ctest --test-dir build --output-on-failure
PYTHONPATH=python .venv/bin/python3 -m unittest discover -s tests/python -v
```

There are four native regression tests and 28 Python tests. Cloud tests use
mocked requests and fake credentials; they do not use your ChatGPT allowance or
send camera images. Apple OCR tests need access to native macOS Vision services.

Optional installed-model checks, using a private saved frame:

```sh
./build/detector_smoke models/objects365_detector.onnx models/objects365.names .local/snapshot.jpg
./build/depth_smoke models/depth_estimator.onnx .local/snapshot.jpg
```

## Upload to GitHub

`.gitignore` excludes `.local/`, legacy credential locations, virtual environments,
model weights, camera captures, IDE settings, generated builds, and `dist/`.
Use Git to add source files; keep private data outside manual uploads.

For a ready-to-upload source archive:

```sh
python3 tools/package_source.py --output dist/VisionAssist-source.zip
```

The packager uses an explicit source-file list, excludes model binaries and
private files, and checks the public files for API keys and saved session tokens
without printing their values. Unzip the archive and upload its contents, or
initialize Git in the extracted folder. Model preparation is required on a fresh
clone. The archive contains no personal camera images or credentials.

## Practical limits

- Object categories and predictions are imperfect; the default ignores `person`
  for spoken object updates. Change `ignore` in the shared config if needed.
- Monocular depth is an **uncalibrated estimate**, not a measured distance in meters.
  The app declines ambiguous closest-object comparisons.
- OCR reads English and works best when the camera is steady and text is clear.
- Speech is serialized; long requested descriptions can delay later callouts.
- This prototype has not been validated as a navigation or mobility aid.

Ultralytics models use its [AGPL-3.0 / Enterprise licensing options](https://www.ultralytics.com/license).
See the [Objects365 model documentation](https://docs.ultralytics.com/datasets/detect/objects365),
[depth documentation](https://docs.ultralytics.com/tasks/depth), and
[Apple Vision OCR documentation](https://developer.apple.com/documentation/vision/recognizing-text-in-images)
for the underlying components.
