import classes from "./yolo_classes.json";
import { Colors } from "./img_preprocess";

/**
 * Draw bounding boxes in overlay canvas.
 * @param {Array[Object]} predictions - Bounding boxes, class and score objects
 * @param {HTMLCanvasElement} overlay_el - Show boxes in overlay canvas element.
 */
export async function draw_bounding_boxes(predictions, overlay_el) {
  const ctx = overlay_el.getContext("2d");

  // Calculate diagonal length of the canvas
  const diagonalLength = Math.sqrt(
    Math.pow(overlay_el.width, 2) + Math.pow(overlay_el.height, 2)
  );
  const lineWidth = diagonalLength / 250;

  // Draw boxes and labels
  predictions.forEach((predict) => {
    // Get color for the class
    const borderColor = Colors.getColor(predict.class_idx, 0.8);
    const rgbaBorderColor = `rgba(${borderColor[0]}, ${borderColor[1]}, ${borderColor[2]}, ${borderColor[3]})`;

    const [x1, y1, width, height] = predict.bbox;

    // Draw border
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = rgbaBorderColor;
    ctx.strokeRect(x1, y1, width, height);

    // Draw text and background
    ctx.fillStyle = rgbaBorderColor;
    ctx.font = "16px Arial";
    const text = `${classes.class[predict.class_idx]} ${predict.score.toFixed(
      2
    )}`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = parseInt(ctx.font, 10);

    // Calculate the Y position for the text
    let textY = y1 - 5;
    let rectY = y1 - textHeight - 4;

    // Check if the text will be outside the canvas
    if (rectY < 0) {
      // Adjust the Y position to be inside the canvas
      textY = y1 + textHeight + 5;
      rectY = y1 + 1;
    }

    ctx.fillRect(x1 - 1, rectY, textWidth + 4, textHeight + 4);
    ctx.fillStyle = "white";
    ctx.fillText(text, x1, textY);
  });
}
