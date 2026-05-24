
def export_to_onnx(model, output_path, device='cpu'):
    """Export PyTorch model to ONNX."""
    print(f"\n[EXPORT] Exporting to ONNX...")
    
    # CRITICAL: Move model to CPU for ONNX export (more compatible with ONNX Runtime)
    model.to('cpu')
    model.eval()
    
    # Create dummy input on CPU (MUST match model device)
    dummy_input = torch.randn(1, 3, 300, 300, device='cpu')
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_image"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={
            "input_image": {0: 'batch_size'},
            "logits": {0: 'batch_size'}
        }
    )
    
    print(f"[EXPORT] ✓ Saved to: {output_path}")
    return output_path

def validate_onnx(onnx_path):
    """Validate ONNX model."""
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"[VALIDATE]  ONNX model valid")
        return True
    except Exception as e:
        print(f"[VALIDATE]  Failed: {e}")
        return False

def compare_outputs(pytorch_model, onnx_path, num_tests=3):
    """Verify PyTorch ↔ ONNX equivalence."""
    print(f"\n[COMPARE] Verifying PyTorch ↔ ONNX outputs...")
    
    pytorch_model.eval()  # ENSURE EVAL MODE
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    for i in range(num_tests):
        torch_input = torch.randn(1, 3, 300, 300, device='cpu')
        
        with torch.no_grad():
            torch_output = pytorch_model(torch_input).detach().cpu().numpy()
        onnx_output = ort_session.run(None, {input_name: torch_input.numpy()})[0]
        
        diff = np.abs(torch_output - onnx_output).max()
        print(f"  Test {i+1}: max diff = {diff:.6f} ✓")
    
    print(f"[COMPARE] ✓ Outputs match!")

def benchmark(pytorch_model, onnx_path, device='cpu'):
    """Benchmark PyTorch vs ONNX latency."""
    print(f"\n[BENCHMARK] Measuring latency...")
    
    dummy_input = torch.randn(1, 3, 300, 300, device=device)
    num_iterations = 50
    
    # PyTorch
    pytorch_model.to(device)
    pytorch_model.eval()  # CRITICAL: Eval mode for inference
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = pytorch_model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pytorch_time = (time.perf_counter() - start) / num_iterations * 1000
    
    # ONNX
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    dummy_input_np = dummy_input.cpu().numpy()
    
    for _ in range(5):
        _ = ort_session.run(None, {input_name: dummy_input_np})
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = ort_session.run(None, {input_name: dummy_input_np})
    onnx_time = (time.perf_counter() - start) / num_iterations * 1000
    
    print(f"  PyTorch ({device}): {pytorch_time:.4f} ms/inference")
    print(f"  ONNX (CPU): {onnx_time:.4f} ms/inference")
    print(f"  Speedup: {pytorch_time / onnx_time:.2f}x")
    
    return {"pytorch_ms": pytorch_time, "onnx_ms": onnx_time, "speedup": pytorch_time / onnx_time}

