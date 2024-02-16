# Lane Departure Warning System
### Overview
This repository contains the source code and resources for a Lane Departure Warning System. The system is designed to detect and alert the driver when the vehicle deviates from its lane without the use of turn signals. It utilizes computer vision techniques to process real-time input from a camera mounted on the vehicle.
### Features
* Lane detection algorithm for identifying lane boundaries.
* Real-time processing of camera feed.
* Warning alerts for lane departure events.
* Easily configurable parameters for fine-tuning.

## Getting started
### Prerequisites
* Python 3.x
* OpenCV
* NumPy
* Pytorch
### Installation
1. Clone the repository
<pre>
  git clone https://github.com/AnkithShetty397/Lane_Departure_Warning_System.git
</pre>
2. Install dependencies
<pre>
  pip install -r requirements.txt
</pre>
3. Run the system
<pre>
  python ./edge_device/deploy_model.py
</pre>

### Configuration
* Adjust the hyperparameters for training the model in src/Trainer.ipynb
* Different parameters related to deploying the system can be adjusted in edge_device/deploy_model.py


