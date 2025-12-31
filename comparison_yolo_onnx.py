from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def benchmark_model(model, image_path, num_runs=10):
    """Benchmark model inference speed"""
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(image_path)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1000 / avg_time
    return avg_time, fps, times

def visualize_comparison(image_path="image2.jpg"):
    """Compare PyTorch and ONNX model predictions with performance analysis"""
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load models
    print("🔍 Loading models...")
    pt_model = YOLO("yolo11n.pt")
    onnx_model = YOLO("yolo11n.onnx")
    
    # Get predictions
    print("🧠 Running predictions...")
    pt_results = pt_model(image_path)[0]
    onnx_results = onnx_model(image_path)[0]
    
    # Extract boxes and confidence scores
    pt_boxes = pt_results.boxes.xyxy.cpu().numpy() if pt_results.boxes else []
    pt_conf = pt_results.boxes.conf.cpu().numpy() if pt_results.boxes else []
    
    onnx_boxes = onnx_results.boxes.xyxy.cpu().numpy() if onnx_results.boxes else []
    onnx_conf = onnx_results.boxes.conf.cpu().numpy() if onnx_results.boxes else []
    
    # Benchmark models
    print("⚡ Benchmarking models...")
    pt_time, pt_fps, _ = benchmark_model(pt_model, image_path, 5)
    onnx_time, onnx_fps, _ = benchmark_model(onnx_model, image_path, 5)
    
    # Create figure with 3x2 layout
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Original Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 2. PyTorch Predictions
    ax2 = plt.subplot(2, 3, 2)
    pt_vis = image_rgb.copy()
    for box, conf in zip(pt_boxes, pt_conf):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(pt_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(pt_vis, f'{conf:.2f}', (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    ax2.imshow(pt_vis)
    ax2.set_title(f'PyTorch Model\n{len(pt_boxes)} detections')
    ax2.axis('off')
    
    # 3. ONNX Predictions
    ax3 = plt.subplot(2, 3, 3)
    onnx_vis = image_rgb.copy()
    for box, conf in zip(onnx_boxes, onnx_conf):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(onnx_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(onnx_vis, f'{conf:.2f}', (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    ax3.imshow(onnx_vis)
    ax3.set_title(f'ONNX Model\n{len(onnx_boxes)} detections')
    ax3.axis('off')
    
    # 4. Performance Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    models = ['PyTorch', 'ONNX']
    speeds = [pt_time, onnx_time]
    colors = ['green', 'blue']
    
    bars = ax4.bar(models, speeds, color=colors, alpha=0.7)
    ax4.set_ylabel('Inference Time (ms)')
    ax4.set_title('Speed Comparison (Lower is Better)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{speed:.1f}ms', ha='center', va='bottom')
    
    # 5. Detection Count Comparison
    ax5 = plt.subplot(2, 3, 5)
    detections = [len(pt_boxes), len(onnx_boxes)]
    bars2 = ax5.bar(models, detections, color=colors, alpha=0.7)
    ax5.set_ylabel('Number of Detections')
    ax5.set_title('Detection Count Comparison')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, detections):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # 6. FPS Comparison
    ax6 = plt.subplot(2, 3, 6)
    fps_values = [pt_fps, onnx_fps]
    bars3 = ax6.bar(models, fps_values, color=colors, alpha=0.7)
    ax6.set_ylabel('FPS')
    ax6.set_title('Frames Per Second (Higher is Better)')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, fps in zip(bars3, fps_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{fps:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.jpg', dpi=150, bbox_inches='tight')
    
    # Determine Winner
    print("\n" + "="*50)
    print("🏆 WINNER ANALYSIS")
    print("="*50)
    
    # Speed comparison
    if pt_time < onnx_time:
        speed_winner = "PyTorch"
        speed_diff = f"{(onnx_time - pt_time):.1f}ms faster"
    else:
        speed_winner = "ONNX"
        speed_diff = f"{(pt_time - onnx_time):.1f}ms faster"
    
    # Detection comparison
    if len(pt_boxes) > len(onnx_boxes):
        detection_winner = "PyTorch"
        detection_diff = f"{len(pt_boxes) - len(onnx_boxes)} more detections"
    elif len(onnx_boxes) > len(pt_boxes):
        detection_winner = "ONNX"
        detection_diff = f"{len(onnx_boxes) - len(pt_boxes)} more detections"
    else:
        detection_winner = "Tie"
        detection_diff = "Same number of detections"
    
    # FPS comparison
    if pt_fps > onnx_fps:
        fps_winner = "PyTorch"
        fps_diff = f"{pt_fps - onnx_fps:.1f} FPS higher"
    else:
        fps_winner = "ONNX"
        fps_diff = f"{onnx_fps - pt_fps:.1f} FPS higher"
    
    # Overall winner
    scores = {"PyTorch": 0, "ONNX": 0}
    if pt_time < onnx_time: scores["PyTorch"] += 1
    else: scores["ONNX"] += 1
    
    if len(pt_boxes) > len(onnx_boxes): scores["PyTorch"] += 1
    elif len(onnx_boxes) > len(pt_boxes): scores["ONNX"] += 1
    
    if pt_fps > onnx_fps: scores["PyTorch"] += 1
    else: scores["ONNX"] += 1
    
    if scores["PyTorch"] > scores["ONNX"]:
        overall_winner = "PyTorch"
    elif scores["ONNX"] > scores["PyTorch"]:
        overall_winner = "ONNX"
    else:
        overall_winner = "Tie"
    
    # Print results
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"{'='*40}")
    print(f"PyTorch Model:")
    print(f"  • Inference Time: {pt_time:.1f} ms")
    print(f"  • FPS: {pt_fps:.1f}")
    print(f"  • Detections: {len(pt_boxes)}")
    print(f"  • Avg Confidence: {np.mean(pt_conf):.3f if len(pt_conf) > 0 else 0:.3f}")
    
    print(f"\nONNX Model:")
    print(f"  • Inference Time: {onnx_time:.1f} ms")
    print(f"  • FPS: {onnx_fps:.1f}")
    print(f"  • Detections: {len(onnx_boxes)}")
    print(f"  • Avg Confidence: {np.mean(onnx_conf):.3f if len(onnx_conf) > 0 else 0:.3f}")
    
    print(f"\n🏅 COMPARISON RESULTS:")
    print(f"{'='*40}")
    print(f"Speed: {speed_winner} ({speed_diff})")
    print(f"Detections: {detection_winner} ({detection_diff})")
    print(f"FPS: {fps_winner} ({fps_diff})")
    print(f"\n🏆 OVERALL WINNER: {overall_winner}!")
    
    # IoU analysis if both detected objects
    if len(pt_boxes) > 0 and len(onnx_boxes) > 0:
        iou_scores = []
        for i in range(min(len(pt_boxes), len(onnx_boxes))):
            iou = calculate_iou(pt_boxes[i], onnx_boxes[i])
            iou_scores.append(iou)
        
        avg_iou = np.mean(iou_scores) if iou_scores else 0
        print(f"\n🔍 IoU Analysis (first {len(iou_scores)} detections):")
        print(f"  • Average IoU: {avg_iou:.3f}")
        if avg_iou > 0.7:
            print("  • ✅ Excellent alignment between models")
        elif avg_iou > 0.5:
            print("  • ⚠️  Good alignment with minor differences")
        else:
            print("  • ❗ Significant differences in detections")
    
    print(f"\n💾 Results saved as:")
    print(f"  • Comparison Analysis: model_comparison_analysis.jpg")
    print(f"  • PyTorch Output: pytorch_predictions.jpg")
    print(f"  • ONNX Output: onnx_predictions.jpg")
    
    # Save individual results
    pt_results.save("pytorch_predictions.jpg")
    onnx_results.save("onnx_predictions.jpg")
    
    plt.show()

if __name__ == "__main__":
    visualize_comparison("image.png")