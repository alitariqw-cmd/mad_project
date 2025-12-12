import onnx
import os
import sys

# Load the ONNX model
model_path = "C:\\Users\\01-135231-091\\Desktop\\Brain and Anzheimea App\\ai_and_mad_project\\alzheimer_coatnet.onnx"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

print(f"Loading model from: {model_path}")
print("File size: 191 MB")
print("=" * 80)

try:
    # Load the model (skip checker for large files)
    print("Loading ONNX model...")
    model = onnx.load(model_path)
    
    print("✅ Model loaded successfully!")
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
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(f"dynamic({dim.dim_param})")
        print(f"Shape: {shape}")
        print()
    
    # Output information
    print(f"\n📤 OUTPUT DETAILS")
    print("=" * 80)
    for output_tensor in graph.output:
        print(f"Name: {output_tensor.name}")
        print(f"Type: {output_tensor.type.tensor_type.elem_type}")
        shape = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(f"dynamic({dim.dim_param})")
        print(f"Shape: {shape}")
        print()
    
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
