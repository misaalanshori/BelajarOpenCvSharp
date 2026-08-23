using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace BelajarOpenCvSharp
{
    internal class Program
    {
        private static readonly List<Anchor> Anchors = BlazePalmAnchors.Generate(192);
        static void Main(string[] args)
        {
            using var camera = new VideoCapture(1);
            var modelPath = Path.Combine(AppContext.BaseDirectory, "model", "033_Hand_Detection_and_Tracking-full-model_float32.onnx");
            using var onnxSession = new InferenceSession(modelPath);

            //// Check output structure
            //foreach (var output in onnxSession.OutputMetadata)
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

                float scoreThreshold = 0.8f;

                List<int> acceptedAnchors = new();

                var rawBoxes = results.First(r => r.Name == "Identity").AsTensor<float>().ToArray();
                var rawScores = results.First(r => r.Name == "Identity_1").AsTensor<float>().ToArray();

                List<Rect2d> candBoxes = new();
                List<float> candScores = new();
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

                    candBoxes.Add(new(x,y,w,h));
                    candScores.Add(score);
                    Console.WriteLine($"{x}, {y}, {w}, {h} -> {score}");
                }

                var resultBoxes = candBoxes.Select(box => new Rect(
                    (int)(box.X * width), (int)(box.Y * height),
                    (int)(box.Width * width), (int)(box.Height * height)
                    )).ToArray();

                Cv2.Dnn.NMSBoxes(resultBoxes, candScores, scoreThreshold, 0.3f, out int[] indices);
                
                foreach (var box in indices.Select(idx => resultBoxes[idx]))
                {

                    Cv2.Rectangle(currentFrame, box, Scalar.Green, 1);
                }



                //Cv2.CvtColor(currentFrame, currentFrame, ColorConversionCodes.BGR2GRAY);
                Cv2.ImShow("Video Feed", currentFrame);
                if (Cv2.WaitKey(1) == 27) break;
            }

            Cv2.DestroyAllWindows();
        }
    }
}
