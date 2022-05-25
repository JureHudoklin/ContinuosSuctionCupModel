# ContinuousSuctionCupModel

## Requirements
   - python3 >= 3.6
   - tensorflow >= 2.3.0

## Installation
1. Install Triemsh Visualize Wrapper
   - Clone the repository from: https://github.com/JureHudoklin/trimesh_visualize
   - Follow the Installation procedure in the repository
3. Install PointNet++ 
   - Go to pointnet2-tensorlow2 subfolder: ``` cd pointnet2-tensorflow2 ```
   - Install the poitnet by running:
   ```
   chmod u+x tf_ops/compile_ops.sh
   tf_ops/compile_ops.sh
   ```
5. Install other requirements
   - run: ```pip install -r requirements.txt```

## Usage
Below are a few examples of how the code can be used

### Continious Suction Cup Model
- To test the Continious Suction Cup models a set of functions is available in "suction_model/suction_cup_main.py":
   -  ```python test_contact_point(file_loc, test_point=np.array([0, 0, 0]), display_contact=True):```

### Grasp Dataset
### Suction Grasp Net

## Examples

- Example video if Suction Grasp Net (press on the image to view):
   [![IMAGE ALT TEXT](http://img.youtube.com/vi/conTv7kHwe8/0.jpg)](http://www.youtube.com/watch?v=conTv7kHwe8 "Suction Grasp Net + Contact Grasp Net - Bin Picking")
   
- Model analysis performed by the Continious Suction Cup Model:
- 
    
