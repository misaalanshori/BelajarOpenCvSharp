using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace BelajarOpenCvSharp
{
    internal class Program
    {
        public readonly record struct Point2D(float X, float Y);
        public readonly record struct BoundingBox(float X, float Y, float Width, float Height);

        public record PalmDetection(
            BoundingBox Box,
            IReadOnlyList<Point2D> Keypoints, // The 7 decoded coarse palm points
            float Score
        );

        private static readonly List<Anchor> Anchors = BlazePalmAnchors.Generate(192);
        static void Main(string[] args)
        {
            using var camera = new VideoCapture(1);
            var modelPath = Path.Combine(AppContext.BaseDirectory, "model", "033_Hand_Detection_and_Tracking-full-model_float32.onnx");
            using var onnxSession = new InferenceSession(modelPath);

            var landmarkModelPath = Path.Combine(AppContext.BaseDirectory, "model", "hand_landmark.onnx");
            using var landmarkSession = new InferenceSession(landmarkModelPath);


            //// Check input structure
            //foreach (var input in landmarkSession.InputMetadata)
            //{
            //    Console.WriteLine($"Input: {input.Key}, Dimensions: [{string.Join(", ", input.Value.Dimensions)}]");
            //}

            //// Check output structure
            //foreach (var output in landmarkSession.OutputMetadata)
            //{
            //    Console.WriteLine($"Output: {output.Key}, Dimensions: [{string.Join(", ", output.Value.Dimensions)}]");
            //}

            //return;

            while (!camera.IsOpened()) continue;

            int width = (int)camera.Get(VideoCaptureProperties.FrameWidth);
            int height = (int)camera.Get(VideoCaptureProperties.FrameHeight);

            camera.AutoExposure = 1.0;

            //Mat currentFrame = new(
            //    rows: height,
            //    cols: width,
            //    type: MatType.CV_8UC3,
            //    s: Scalar.Black
            //    );
            Mat currentFrame = new();
            Mat rgbFrame = new();
            Mat resizedFrame = new();
            while (camera.Read(currentFrame))
            {
                if (currentFrame.Empty())
                {
                    continue;
                }
                Console.WriteLine("Frame!");
                Cv2.CvtColor(currentFrame, rgbFrame, ColorConversionCodes.BGR2RGB);
                Cv2.Resize(rgbFrame, resizedFrame, new Size(192, 192));

                var inputTensor = new DenseTensor<float>(new[] { 1, 3, 192, 192 });
                for (int y = 0; y < 192; y++)
                {
                    for (int x = 0; x < 192; x++)
                    {
                        var pixel = resizedFrame.At<Vec3b>(y, x);
                        inputTensor[0, 0, y, x] = pixel.Item0 / 255.0f;
                        inputTensor[0, 1, y, x] = pixel.Item1 / 255.0f;
                        inputTensor[0, 2, y, x] = pixel.Item2 / 255.0f;
                    }
                }

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_1", inputTensor) };

                using var results = onnxSession.Run(inputs);

                float scoreThreshold = 0.5f;

                List<int> acceptedAnchors = new();

                var rawBoxes = results.First(r => r.Name == "Identity").AsTensor<float>().ToArray();
                var rawScores = results.First(r => r.Name == "Identity_1").AsTensor<float>().ToArray();

                List<Rect2d> candBoxes = new();
                List<float> candScores = new();
                List<List<Point2D>> candKeypoints = new();
                for (int i = 0; i < 2016; i++)
                {
                    float score = 1.0f / (1.0f + MathF.Exp(-rawScores[i]));
                    if (score < scoreThreshold) continue;
                    Console.WriteLine("OOh!");

                    int offset = i * 18;
                    var anchor = Anchors[i];

                    float cx = rawBoxes[offset + 0] / 192f + anchor.XCenter;
                    float cy = rawBoxes[offset + 1] / 192f + anchor.YCenter;
                    float w = rawBoxes[offset + 2] / 192f;
                    float h = rawBoxes[offset + 3] / 192f;

                    float x = cx - (w / 2f);
                    float y = cy - (h / 2f);

                    List<Point2D> keypoints = new();
                    for (int j = 0; j < 7; j++)
                    {
                        int keyOffset = offset + 4 + (j * 2);
                        float keyX = rawBoxes[keyOffset] / 192f + anchor.XCenter;
                        float keyY = rawBoxes[keyOffset + 1] / 192f + anchor.YCenter;
                        keypoints.Add(new Point2D(keyX, keyY));
                    }

                    candBoxes.Add(new(x,y,w,h));
                    candScores.Add(score);
                    candKeypoints.Add(keypoints);
                    // Console.WriteLine($"{x}, {y}, {w}, {h} -> {score}");
                }

                var resultBoxes = candBoxes.Select(box => new Rect(
                    (int)(box.X * width), (int)(box.Y * height),
                    (int)(box.Width * width), (int)(box.Height * height)
                    )).ToArray();

                

                Cv2.Dnn.NMSBoxes(resultBoxes, candScores, scoreThreshold, 0.3f, out int[] indices);

                List<Mat> hands = new();

                int handIdx = 0;
                foreach (var idx in indices)
                {
                    var box = resultBoxes[idx];
                    //var frameCopy = new Mat();
                    //currentFrame.CopyTo(frameCopy);
                    
                    var keypoints = candKeypoints[idx];
                    
                    var wrist = keypoints[0];
                    var middle = keypoints[2];

                    // math for rotation
                    /*
                     - so first we get those keypoints multiplied by real dimensions
                     - we get the delta between the position of the middle knuckle point
                       and wrist point, this gets us vx and vy, which are vectors
                     - so then we calculate the actual straight line distance (fancy term: euclidian distance) (in pixels) between the wrist and the middle knuckle (vLen)
                     - then we normalize our vx and vy by using vLen, this apparently mean dirX and dirY contains only directions
                     */
                    float middleX = middle.X * width;
                    float wristX = wrist.X * width;
                    float middleY = middle.Y * height;
                    float wristY = wrist.Y * height;

                    float vx = middleX - wristX;
                    float vy = middleY - wristY;
                    float vLen = MathF.Sqrt(vx * vx + vy * vy);

                    // so apparently the 1e-6f is an "epsilon", a really small number (0.000001) that is there to just prevent division by zero, not **technically** part of the formula
                    float dirX = vx / (vLen + 1e-6f); 
                    float dirY = vy / (vLen + 1e-6f);

                    // get rotation
                    /*
                     - then we divide pi by 2 and subtract the 2-arg arctangent of -1*dirY and dirX
                       according to wikipedia atan2(y, x) is used to convert from rect coords to polar coords, whatever that means (okay i actually know what that means, kinda, but still)
                       but basically that means we get the angle in radians (which we then turn into degrees)
                       and dirY is flipped because opencv coordinate system stuff
                     */
                    float rotRad = MathF.PI / 2.0f - MathF.Atan2(-dirY, dirX);
                    float rotDeg = rotRad * (180.0f / MathF.PI);

                    // resize the box
                    /*
                     - so first we get just the widest dimension of the palm bounding box
                       and multiply it by 2.6 to ensure it covers the entire hand 
                     - we also figure out the palm's center by taking the box coordinates,
                       which I think would be top left right? and then adding half the width/height
                       okay this one shouldnt be too confusing
                     */
                    float palmSizePx = MathF.Max(box.Width, box.Height);
                    float handBoxSizePx = palmSizePx * 2.6f;

                    float palmCenterX = box.X + box.Width / 2.0f;
                    float palmCenterY = box.Y + box.Height / 2.0f;

                    // shift the box
                    /*
                     - we shift the box by half the palm's max size
                     - we offset the palmCenter coords by amount of pixel shift (shiftPx) multiplied by the direction (dirX and dirY) 
                     */
                    float shiftPx = 0.5f * palmSizePx;
                    float handCenterX = palmCenterX + (shiftPx * dirX);
                    float handCenterY = palmCenterY + (shiftPx * dirY);

                    Point2f center = new(handCenterX, handCenterY);
                    Size targetSize = new(224, 224);

                    float scale = 224.0f / handBoxSizePx;

                    // basically this sets up the math to move the image around
                    // just refer to this gemini link https://share.gemini.google/DnvbPvzkOo8f
                    Mat affineMatrix = Cv2.GetRotationMatrix2D(center, rotDeg, scale);

                    // this part here modifies the matrix made above to adjust the center point? idk
                    double tX = affineMatrix.At<Double>(0, 2) + (112.0 - center.X);
                    double tY = affineMatrix.At<Double>(1, 2) + (112.0 - center.Y);
                    affineMatrix.Set(0, 2, tX); 
                    affineMatrix.Set(1, 2, tY);

                    var cropppedHand = new Mat();
                    Cv2.WarpAffine(currentFrame, cropppedHand, affineMatrix, targetSize, InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);

                    var handInputTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
                    for (int y = 0; y < 224; y++)
                    {
                        for (int x = 0; x < 224; x++)
                        {
                            var pixel = cropppedHand.At<Vec3b>(y, x);
                            // what if we just flip the BGR to RGB here?
                            handInputTensor[0, y, x, 0] = pixel.Item2 / 255.0f;
                            handInputTensor[0, y, x, 1] = pixel.Item1 / 255.0f;
                            handInputTensor[0, y, x, 2] = pixel.Item0 / 255.0f;
                        }
                    }

                    var landmarkInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_1", handInputTensor) };

                    using var landmarkResults = landmarkSession.Run(landmarkInputs);

                    var rawLandmarks = landmarkResults.First(r => r.Name == "Identity").AsTensor<float>().ToArray();

                    List<Point2D> landmarks = new();
                    
                    for (int j = 0; j < 21; j++)
                    {
                        int offset = j * 3;

                        float landmarkX = rawLandmarks[offset];
                        float landmarkY = rawLandmarks[offset+1];
                        Cv2.Circle(cropppedHand, new Point((int)landmarkX, (int)landmarkY), 5, Scalar.Yellow);
                    }

                    try
                    {
                        var handFrame = cropppedHand;
                        Cv2.ImShow($"hand{handIdx}", handFrame);
                    }
                    catch
                    {

                    }

                    using var invAffine = new Mat();
                    Cv2.InvertAffineTransform(affineMatrix, invAffine);

                    for (int i = 0; i < 21; i++)
                    {
                        int offset = i * 3;
                        float lx = rawLandmarks[offset];
                        float ly = rawLandmarks[offset + 1];

                        double m00 = invAffine.At<double>(0, 0);
                        double m01 = invAffine.At<double>(0, 1);
                        double m02 = invAffine.At<double>(0, 2);
                        double m10 = invAffine.At<double>(1, 0);
                        double m11 = invAffine.At<double>(1, 1);
                        double m12 = invAffine.At<double>(1, 2);

                        float ox = (float)(m00 * lx + m01 * ly + m02);
                        float oy = (float)(m10 * lx + m11 * ly + m12);
                        Cv2.Circle(currentFrame, new Point(ox, oy), 5, Scalar.Yellow);
                    }

                    Cv2.Rectangle(currentFrame, box, Scalar.Green, 1);
                    foreach (var kp in keypoints)
                    {
                        Cv2.Circle(currentFrame, new Point(kp.X * width, kp.Y * height), 5, Scalar.Red);
                    }
                    
                    handIdx++;
                }



                //Cv2.CvtColor(currentFrame, currentFrame, ColorConversionCodes.BGR2GRAY);
                Cv2.ImShow("Video Feed", currentFrame);
                if (Cv2.WaitKey(1) == 27) break;
            }

            Cv2.DestroyAllWindows();
        }
    }
}
