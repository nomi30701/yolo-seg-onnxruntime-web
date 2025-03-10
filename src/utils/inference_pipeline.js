import * as ort from "onnxruntime-web/webgpu";
import { preProcess_dynamic, applyNMS, Colors } from "./img_preprocess";

export const inference_pipeline = async (
  input_el,
  session,
  config,
  overlay_el
) => {
  const src_mat = cv.imread(input_el);

  // pre process
  const [src_mat_preProcessed, div_width, div_height] =
    preProcess_dynamic(src_mat);
  const xRatio = src_mat.cols / div_width;
  const yRatio = src_mat.rows / div_height;

  src_mat.delete();

  const input_tensor = new ort.Tensor("float32", src_mat_preProcessed.data32F, [
    1,
    3,
    div_height,
    div_width,
  ]);
  src_mat_preProcessed.delete();

  // inference
  const start = performance.now();
  const { output0, output1 } = await session.run({
    images: input_tensor,
  });
  const end = performance.now();
  input_tensor.dispose();

  // post process
  const NUM_PREDICTIONS = output0.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;
  const NUM_MASK_WEIGHTS = 32;

  const predictions = output0.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const scores_data = predictions.subarray(
    NUM_PREDICTIONS * NUM_BBOX_ATTRS,
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES)
  );
  const mask_weights_data = predictions.subarray(
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES)
  );

  const proto_mask = output1.data;
  const MASK_CHANNELS = output1.dims[1];
  const MASK_HEIGHT = output1.dims[2];
  const MASK_WIDTH = output1.dims[3];
  output0.dispose();
  output1.dispose();

  const results = [];
  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    let maxScore = 0;
    let class_idx = -1;

    for (let c = 0; c < NUM_SCORES; c++) {
      const score = scores_data[i + c * NUM_PREDICTIONS];
      if (score > maxScore) {
        maxScore = score;
        class_idx = c;
      }
    }
    if (maxScore <= config.score_threshold) continue;

    const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
    const x = bbox_data[i] * xRatio - 0.5 * w;
    const y = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    const mask_weights = new Float32Array(NUM_MASK_WEIGHTS);
    for (let c = 0; c < NUM_MASK_WEIGHTS; c++) {
      mask_weights[c] = mask_weights_data[i + c * NUM_PREDICTIONS];
    }

    results.push({
      bbox: [x, y, w, h],
      class_idx,
      score: maxScore,
      mask_weights,
    });
  }

  // nms
  const selected_indices = applyNMS(
    results,
    results.map((r) => r.score),
    config.iou_threshold
  );
  const filtered_results = selected_indices.map((i) => results[i]);

  // mask process
  const proto_mask_mat = cv.matFromArray(
    MASK_CHANNELS, // 32
    MASK_HEIGHT * MASK_WIDTH, // 128 * 128
    cv.CV_32F,
    proto_mask
  );

  // weights x proto_mask
  const NUM_FILTERED_RESULTS = filtered_results.length;
  const mask_weights = filtered_results
    .map((r) => Array.from(r.mask_weights))
    .flat();
  const mask_weights_mat = cv.matFromArray(
    NUM_FILTERED_RESULTS,
    MASK_CHANNELS,
    cv.CV_32F,
    mask_weights
  );
  const weights_mul_proto_mat = new cv.Mat();
  cv.gemm(
    mask_weights_mat,
    proto_mask_mat,
    1.0,
    new cv.Mat(),
    0.0,
    weights_mul_proto_mat
  );

  proto_mask_mat.delete();
  mask_weights_mat.delete();

  // sigmoid
  const mask_sigmoid_mat = new cv.Mat();
  const ones_mat = cv.Mat.ones(weights_mul_proto_mat.size(), cv.CV_32F);
  cv.multiply(weights_mul_proto_mat, ones_mat, mask_sigmoid_mat, -1);
  cv.exp(mask_sigmoid_mat, mask_sigmoid_mat);
  cv.add(mask_sigmoid_mat, ones_mat, mask_sigmoid_mat);
  cv.divide(ones_mat, mask_sigmoid_mat, mask_sigmoid_mat);

  ones_mat.delete();
  weights_mul_proto_mat.delete();

  // same size overlay mat
  const overlay_mat = new cv.Mat(
    overlay_el.height,
    overlay_el.width,
    cv.CV_8UC4,
    new cv.Scalar(0, 0, 0, 0)
  );

  for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
    // reshape
    const mask = mask_sigmoid_mat.row(i).data32F;
    const mask_mat = cv.matFromArray(MASK_HEIGHT, MASK_WIDTH, cv.CV_32F, mask);

    // upsample to overlay size
    const mask_resized_mat = new cv.Mat();
    cv.resize(
      mask_mat,
      mask_resized_mat,
      new cv.Size(overlay_el.width, overlay_el.height),
      cv.INTER_LINEAR
    );

    // threshold
    const mask_binary_mat = new cv.Mat();
    const mask_binary_u8_mat = new cv.Mat();
    cv.threshold(mask_resized_mat, mask_binary_mat, 0.5, 1, cv.THRESH_BINARY);
    mask_binary_mat.convertTo(mask_binary_u8_mat, cv.CV_8U);

    // crop and color
    const [x, y, w, h] = filtered_results[i].bbox;
    const x1 = Math.max(0, x);
    const y1 = Math.max(0, y);
    const x2 = Math.min(overlay_el.width, x + w);
    const y2 = Math.min(overlay_el.height, y + h);

    const roi = mask_binary_u8_mat.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
    const color = Colors.getColor(filtered_results[i].class_idx, 0.6);
    const color_scalar = new cv.Scalar(
      color[0],
      color[1],
      color[2],
      color[3] * 255
    );
    const mask_colored_mat = new cv.Mat(
      roi.rows,
      roi.cols,
      cv.CV_8UC4,
      color_scalar
    );
    mask_colored_mat.copyTo(
      overlay_mat.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)),
      roi
    );

    roi.delete();
    mask_resized_mat.delete();
    mask_binary_mat.delete();
    mask_mat.delete();
    mask_colored_mat.delete();
  }
  mask_sigmoid_mat.delete();


  // draw masks
  const imgData = new ImageData(
    new Uint8ClampedArray(overlay_mat.data),
    overlay_el.width,
    overlay_el.height
  );
  const ctx = overlay_el.getContext("2d");
  ctx.putImageData(imgData, 0, 0);

  overlay_mat.delete();
  return [filtered_results, (end - start).toFixed(2)];
};
