from pydantic_graph import Graph
from beagle.data_generation.studentv2.state import StudentState
from beagle.data_generation.studentv2.deps import StudentDeps
from beagle.data_generation.studentv2.nodes import (
    MarkovNode, StrategistNode, ExecutorNode, EnvironmentNode,
    OffTopicNode, AssistanceNode, TutorNode
)

# Define the graph with all nodes including interrupt state nodes
student_graph = Graph(nodes=[
    MarkovNode,
    OffTopicNode,
    AssistanceNode,
    TutorNode,
    StrategistNode,
    ExecutorNode,
    EnvironmentNode
])
