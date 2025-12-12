import onnx
import os
import sys

# Load the ONNX model
model_path = os.path.join(os.path.dirname(__file__), "..", "ai_and_mad_project", "alzheimer_coatnet.onnx")

# Try alternative path if first one fails
if not os.path.exists(model_path):
    model_path = "alzheimer_coatnet.onnx"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

print(f"Loading model from: {model_path}")
print("=" * 80)

try:
    # Load and check the model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    print("\n📊 MODEL INFORMATION")
    print("=" * 80)
    print(f"Model IR Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    
    # Get graph information
    graph = model.graph
    print(f"\n📈 GRAPH INFORMATION")
    print("=" * 80)
    print(f"Graph Name: {graph.name}")
    print(f"Number of Nodes: {len(graph.node)}")
    print(f"Number of Inputs: {len(graph.input)}")
    print(f"Number of Outputs: {len(graph.output)}")
    
    # Input information
    print(f"\n📥 INPUT DETAILS")
    print("=" * 80)
    for input_tensor in graph.input:
        print(f"Name: {input_tensor.name}")
        print(f"Type: {input_tensor.type.tensor_type.elem_type}")
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})" 
                 for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
        print()
    
    # Output information
    print(f"\n📤 OUTPUT DETAILS")
    print("=" * 80)
    for output_tensor in graph.output:
        print(f"Name: {output_tensor.name}")
        print(f"Type: {output_tensor.type.tensor_type.elem_type}")
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})" 
                 for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
        print()
    
    # Initializers (weights/constants)
    print(f"\n⚙️  INITIALIZERS (WEIGHTS/CONSTANTS)")
    print("=" * 80)
    print(f"Number of Initializers: {len(graph.initializer)}")
    for init in graph.initializer[:5]:  # Show first 5
        shape = list(init.dims)
        data_type = init.data_type
        print(f"Name: {init.name}")
        print(f"Shape: {shape}")
        print(f"Data Type: {data_type}")
        print()
    
    if len(graph.initializer) > 5:
        print(f"... and {len(graph.initializer) - 5} more initializers")
    
    # Operators used
    print(f"\n🔧 OPERATORS USED")
    print("=" * 80)
    op_types = {}
    for node in graph.node:
        op_type = node.op_type
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    for op_type, count in sorted(op_types.items()):
        print(f"{op_type}: {count}")
    
    print(f"\nTotal unique operators: {len(op_types)}")
    
    print("\n" + "=" * 80)
    print("✅ ONNX Model Analysis Complete!")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
