
### Image synthesis

##### ESIPE-IMAC 3, UGE

### Assignment and project

#### April 4th, 2021

----

### Chosen subjects

- Support loading multiple models.
- Support ambient light.

### Usage

From the project's root directory :

    build/bin/gltf-viewer viewer [path_to_model_1][...]

### Build

From the project's root directory :

    mkdir build
    cd build
    cmake ..
    make  

### Known issues

- Cannot not load models at run-time from given path.

### Main takeaways

- Learnt about the glTF format and what data it can hold.
- Learnt how model deserialisation to OpenGL data can be used to easily display complex meshes.
- Learnt how to use Dear ImGui and why it is convenient to do so.
