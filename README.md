LogicSplat

LogicSplat is a neuro-symbolic framework for structured scene understanding from monocular RGB video. The system converts raw video frames into a symbolic 3D scene graph, where objects are represented as nodes and their spatial or physical relationships are represented as edges.

Unlike traditional deep learning pipelines that produce only pixel-level predictions, LogicSplat builds a structured representation of the scene, enabling interpretable reasoning about object interactions and physical relationships.

Motivation

Modern computer vision models are excellent at detecting objects but struggle with explicit reasoning about relationships.

For example, most models can detect a cup and a table, but they cannot explicitly represent or reason that:

the cup is supported by the table

the cup is above the table

the table occludes part of another object

LogicSplat addresses this gap by combining neural perception with symbolic reasoning, allowing scenes to be represented in a form that machines can reason about logically.

Key Features

• Object detection from RGB video frames
• Construction of a scene graph representation
• Encoding of spatial and physical relationships such as:

support

containment

adjacency

occlusion

instability

• Bridge between deep learning perception and symbolic reasoning

System Pipeline

The LogicSplat pipeline follows these stages:

Video Input
Monocular RGB video is provided as input.

Object Detection
Neural models detect objects and estimate their positions.

Spatial Reasoning Module
Relationships between objects are inferred using geometric constraints.

Scene Graph Construction
Objects become nodes and relationships become edges in a structured graph.

Symbolic Representation
The final output is a symbolic scene graph that supports reasoning.

Project Structure
LogicSplat
│
├── data/              # input datasets and videos
├── src/               # core implementation
├── models/            # trained models
├── notebooks/         # experiments and visualization
├── outputs/           # generated scene graphs
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone https://github.com/Debjyoti-G77/logicsplat.git
cd logicsplat

Install dependencies:

pip install -r requirements.txt
Example Use Case

Input:
A video containing objects such as a cup placed on a table next to a book.

Output scene graph:

cup → supported_by → table  
book → adjacent_to → cup  
table → supports → cup

This structured representation allows machines to reason about physical interactions within the scene.

Future Work

Integration with 3D reconstruction pipelines

Temporal reasoning across video frames

Physical stability prediction

Applications in robotics and embodied AI

License

This project is released under the MIT License.
